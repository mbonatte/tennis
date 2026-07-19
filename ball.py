from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from BallTrack.model import BallTrackerNet
from tennis_analyzer.checkpoints import normalize_state_dict
from tennis_analyzer.errors import VideoProcessingError
from tennis_analyzer.pipeline.ball_track import euclidean as _euclidean
from tennis_analyzer.pipeline.ball_track import postprocess_ball_track
from tennis_analyzer.pipeline.temporal import TemporalContextBuffer

_TORCH_THREADS_CONFIGURED = False
MODEL_WIDTH = 640
MODEL_HEIGHT = 360
REFERENCE_WIDTH = 1280.0
REFERENCE_HEIGHT = 720.0
TEMPORAL_WINDOW_FRAMES = 3
TEMPORAL_CONTEXT_FRAMES = TEMPORAL_WINDOW_FRAMES - 1


def load_model(model_path: Path, device: torch.device, use_compile: bool = False):
    """Load the upstream TrackNet architecture with application-owned weights."""
    model = BallTrackerNet()
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(normalize_state_dict(checkpoint))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise VideoProcessingError("The ball-tracking checkpoint is incompatible or unreadable") from exc
    model.to(device)
    model.eval()
    return torch.compile(model) if use_compile else model


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


class BallTracker:
    """Own one TrackNet model and its cross-chunk temporal source-frame state."""

    temporal_context_frames = TEMPORAL_CONTEXT_FRAMES

    def __init__(self, model, device: torch.device, *, batch_size: int = 8, use_amp: bool = False):
        if batch_size <= 0:
            raise ValueError("ball batch size must be positive")
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.use_amp = use_amp
        self._context = TemporalContextBuffer(self.temporal_context_frames)

    @classmethod
    def from_checkpoint(cls, model_path, device_name=None, *, batch_size=8, use_amp=False):
        global _TORCH_THREADS_CONFIGURED
        device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not _TORCH_THREADS_CONFIGURED:
            cpu_threads = max(1, (os.cpu_count() or 4) - 1)
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(1)
            _TORCH_THREADS_CONFIGURED = True
        model = load_model(Path(model_path), device, use_compile=False)
        return cls(model, device, batch_size=batch_size, use_amp=use_amp)

    def process_chunk(self, frames):
        if not frames:
            return []
        combined, overlap = self._context.prepend(frames)
        effective_batch = min(self.batch_size, max(1, len(combined) - self.temporal_context_frames))
        track, _ = infer_model_batched(
            combined,
            self.model,
            self.device,
            batch_size=effective_batch,
            use_amp=self.use_amp,
        )
        result = track[overlap:]
        if len(result) != len(frames):
            raise VideoProcessingError("Ball inference returned an unexpected number of frame results")
        return result

    def close(self):
        self._context.clear()
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def track_ball(frames, extrapolation=True, model_path=None, device_name=None, batch_size=8):
    """Compatibility helper for one in-memory frame sequence.

    The production pipeline owns a :class:`BallTracker` directly so its model
    and temporal state survive every source chunk.
    """
    project_root = Path(__file__).resolve().parent
    model_path = Path(model_path) if model_path else project_root / "weights" / "tracknet_model.pt"
    tracker = BallTracker.from_checkpoint(model_path, device_name, batch_size=batch_size)
    try:
        return postprocess_ball_track(tracker.process_chunk(frames), extrapolation=extrapolation)
    finally:
        tracker.close()


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
