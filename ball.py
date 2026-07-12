from __future__ import annotations

import math
import os
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
import torch

from BallTrack.model import BallTrackerNet

_TORCH_THREADS_CONFIGURED = False
MODEL_WIDTH = 640
MODEL_HEIGHT = 360
REFERENCE_WIDTH = 1280.0
REFERENCE_HEIGHT = 720.0


def load_model(model_path: Path, device: torch.device, use_compile: bool = False):
    """Load the upstream TrackNet architecture with application-owned weights."""
    model = BallTrackerNet()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return torch.compile(model) if use_compile else model


def _euclidean(first, second) -> float:
    if first[0] is None or second[0] is None:
        return -1.0
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _postprocess(feature_map, scale: float = 2.0):
    heatmap = (feature_map.reshape((MODEL_HEIGHT, MODEL_WIDTH)) * 255).astype(np.uint8)
    _, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7,
    )
    if circles is None or len(circles) != 1:
        return None, None
    return circles[0][0][0] * scale, circles[0][0][1] * scale


def _input_triplet(current, previous, pre_previous) -> np.ndarray:
    images = [cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT)) for frame in (current, previous, pre_previous)]
    return np.transpose(np.concatenate(images, axis=2).astype(np.float32) / 255.0, (2, 0, 1))


@torch.inference_mode()
def infer_model_batched(frames, model, device, batch_size: int = 32, use_amp: bool = True):
    """Infer ball centers in bounded batches while retaining three-frame context."""
    distances = [-1.0, -1.0]
    track = [(None, None), (None, None)]
    samples: list[np.ndarray] = []
    frame_indices: list[int] = []

    def flush() -> None:
        if not samples:
            return
        tensor = torch.from_numpy(np.stack(samples)).to(device=device, dtype=torch.float32)
        with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            output = model(tensor).argmax(dim=1).detach().cpu().numpy()
        for prediction, frame_index in zip(output, frame_indices, strict=True):
            x_value, y_value = _postprocess(prediction)
            height, width = frames[frame_index].shape[:2]
            point = (
                (None, None)
                if x_value is None or y_value is None
                else (float(x_value) * width / REFERENCE_WIDTH, float(y_value) * height / REFERENCE_HEIGHT)
            )
            track.append(point)
            distances.append(_euclidean(track[-1], track[-2]))
        samples.clear()
        frame_indices.clear()

    for frame_index in range(2, len(frames)):
        samples.append(_input_triplet(frames[frame_index], frames[frame_index - 1], frames[frame_index - 2]))
        frame_indices.append(frame_index)
        if len(samples) >= batch_size:
            flush()
    flush()
    return track, distances


def remove_outliers(track, distances, max_distance: float = 100):
    outliers = list(np.where(np.asarray(distances) > max_distance)[0])
    for index in outliers.copy():
        next_distance = distances[index + 1] if index + 1 < len(distances) else None
        if next_distance is not None and (next_distance > max_distance or next_distance == -1):
            track[index] = (None, None)
            if index in outliers:
                outliers.remove(index)
        elif index > 0 and distances[index - 1] == -1:
            track[index - 1] = (None, None)
    return track


def split_track(track, max_gap: int = 4, max_distance_gap: float = 80, min_track: int = 5):
    groups = [
        (missing, sum(1 for _ in values)) for missing, values in groupby(0 if p[0] is not None else 1 for p in track)
    ]
    cursor = 0
    start = 0
    result = []
    for group_index, (missing, length) in enumerate(groups):
        if missing == 1 and 0 < group_index < len(groups) - 1:
            distance = _euclidean(track[cursor - 1], track[cursor + length])
            if length >= max_gap or distance / max(length, 1) > max_distance_gap:
                if cursor - start > min_track:
                    result.append([start, cursor])
                start = cursor + length - 1
        cursor += length
    if len(track) - start > min_track:
        result.append([start, len(track)])
    return result


def interpolation(coordinates):
    x_values = np.asarray([point[0] if point[0] is not None else np.nan for point in coordinates], dtype=float)
    y_values = np.asarray([point[1] if point[1] is not None else np.nan for point in coordinates], dtype=float)
    for values in (x_values, y_values):
        valid = np.where(~np.isnan(values))[0]
        missing = np.where(np.isnan(values))[0]
        if len(valid) >= 2:
            values[missing] = np.interp(missing, valid, values[valid])
    return list(zip(x_values, y_values, strict=True))


def track_ball(frames, extrapolation=True, model_path=None, device_name=None):
    global _TORCH_THREADS_CONFIGURED
    project_root = Path(__file__).resolve().parent

    model_path = Path(model_path) if model_path else project_root / "weights" / "tracknet_model.pt"

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not _TORCH_THREADS_CONFIGURED:
        cpu_threads = max(1, (os.cpu_count() or 4) - 1)
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        _TORCH_THREADS_CONFIGURED = True

    use_compile = False
    model = load_model(model_path, device, use_compile=use_compile)

    # ball_track, dists = infer_model(frames, model, device)
    ball_track, dists = infer_model_batched(
        frames,
        model,
        device=device,
        batch_size=8,
        use_amp=False,
    )
    ball_track = remove_outliers(ball_track, dists)

    if extrapolation:
        subtracks = split_track(ball_track)
        for start, end in subtracks:
            ball_subtrack = ball_track[start:end]
            ball_track[start:end] = interpolation(ball_subtrack)

    return ball_track


def draw_track(frames, ball_track, trace=7):
    """
    Draw ball tracking trail on frames.

    Returns:
        processed_frames: list of frames with drawings
    """
    processed_frames = []

    for num in range(len(frames)):
        frame = frames[num].copy()

        for i in range(trace):
            idx = num - i
            if idx < 0:
                break

            if _has_valid_ball_point(ball_track[idx]):
                x = int(ball_track[idx][0])
                y = int(ball_track[idx][1])

                cv2.circle(
                    frame,
                    (x, y),
                    radius=0,
                    color=(0, 0, 255),
                    thickness=max(1, 10 - i),
                )
            else:
                break

        processed_frames.append(frame)

    return processed_frames


def _has_valid_ball_point(point):
    if point is None or point[0] is None or point[1] is None:
        return False
    return np.isfinite(point[0]) and np.isfinite(point[1])
