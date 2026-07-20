from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
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


def _configured_cpu_threads() -> int:
    """Use an explicit container-friendly limit when one is configured."""
    configured = os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    affinity = getattr(os, "sched_getaffinity", None)
    available = len(affinity(0)) if affinity is not None else (os.cpu_count() or 1)
    return max(1, available - 1)


@dataclass
class BallStageTimings:
    """Cumulative, monotonic timings for one job-scoped ball tracker."""

    frames: int = 0
    batches: int = 0
    resize_seconds: float = 0.0
    input_seconds: float = 0.0
    transfer_seconds: float = 0.0
    inference_seconds: float = 0.0
    postprocess_seconds: float = 0.0

    def to_log_fields(self) -> dict[str, float | int]:
        total = sum(
            (
                self.resize_seconds,
                self.input_seconds,
                self.transfer_seconds,
                self.inference_seconds,
                self.postprocess_seconds,
            )
        )
        return {
            "ball_frames": self.frames,
            "ball_batches": self.batches,
            "ball_resize_seconds": round(self.resize_seconds, 3),
            "ball_input_seconds": round(self.input_seconds, 3),
            "ball_transfer_seconds": round(self.transfer_seconds, 3),
            "ball_inference_seconds": round(self.inference_seconds, 3),
            "ball_postprocess_seconds": round(self.postprocess_seconds, 3),
            "ball_measured_seconds": round(total, 3),
            "ball_frames_per_second": round(self.frames / total, 3) if total else 0.0,
        }


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


def _resized_frames(frames) -> np.ndarray:
    """Resize each source frame once; temporal windows share these pixels."""
    resized = np.empty((len(frames), MODEL_HEIGHT, MODEL_WIDTH, 3), dtype=np.uint8)
    for index, frame in enumerate(frames):
        cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT), dst=resized[index])
    return resized


def _fill_input_batch(resized: np.ndarray, start: int, count: int, output: np.ndarray) -> np.ndarray:
    """Fill N TrackNet triplets without intermediate concatenated frame arrays."""
    batch = output[:count]
    # The model expects current, previous, then pre-previous BGR channels.
    batch[:, 0:3] = np.moveaxis(resized[start + 2 : start + count + 2], -1, 1)
    batch[:, 3:6] = np.moveaxis(resized[start + 1 : start + count + 1], -1, 1)
    batch[:, 6:9] = np.moveaxis(resized[start : start + count], -1, 1)
    np.multiply(batch, 1.0 / 255.0, out=batch)
    return batch


@torch.inference_mode()
def infer_model_batched(
    frames, model, device, batch_size: int = 32, use_amp: bool = True, timings: BallStageTimings | None = None
):
    """Infer ball centers in bounded batches while retaining three-frame context."""
    distances = [-1.0, -1.0]
    track = [(None, None), (None, None)]
    if len(frames) <= TEMPORAL_CONTEXT_FRAMES:
        return track[: len(frames)], distances[: len(frames)]

    started = time.perf_counter()
    resized = _resized_frames(frames)
    if timings:
        timings.resize_seconds += time.perf_counter() - started
    batch_buffer = np.empty((batch_size, 9, MODEL_HEIGHT, MODEL_WIDTH), dtype=np.float32)

    def flush(start: int, count: int) -> None:
        started = time.perf_counter()
        inputs = _fill_input_batch(resized, start, count, batch_buffer)
        if timings:
            timings.input_seconds += time.perf_counter() - started
        started = time.perf_counter()
        tensor = torch.from_numpy(inputs).to(device=device, dtype=torch.float32)
        if timings:
            timings.transfer_seconds += time.perf_counter() - started
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            output = model(tensor).argmax(dim=1).detach().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if timings:
            timings.inference_seconds += time.perf_counter() - started
            timings.batches += 1
        started = time.perf_counter()
        for offset, prediction in enumerate(output):
            frame_index = start + offset + TEMPORAL_CONTEXT_FRAMES
            x_value, y_value = _postprocess(prediction)
            height, width = frames[frame_index].shape[:2]
            point = (
                (None, None)
                if x_value is None or y_value is None
                else (float(x_value) * width / REFERENCE_WIDTH, float(y_value) * height / REFERENCE_HEIGHT)
            )
            track.append(point)
            distances.append(_euclidean(track[-1], track[-2]))
        if timings:
            timings.postprocess_seconds += time.perf_counter() - started

    samples = len(frames) - TEMPORAL_CONTEXT_FRAMES
    for start in range(0, samples, batch_size):
        flush(start, min(batch_size, samples - start))
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
        self.timings = BallStageTimings()

    @classmethod
    def from_checkpoint(cls, model_path, device_name=None, *, batch_size=8, use_amp=False):
        global _TORCH_THREADS_CONFIGURED
        device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not _TORCH_THREADS_CONFIGURED:
            cpu_threads = _configured_cpu_threads()
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
        infer_kwargs = {"batch_size": effective_batch, "use_amp": self.use_amp}
        # Keep the small adapter seam compatible with existing test/dummy
        # inference functions while production inference receives timings.
        parameters = inspect.signature(infer_model_batched).parameters.values()
        if any(parameter.name == "timings" or parameter.kind == parameter.VAR_KEYWORD for parameter in parameters):
            infer_kwargs["timings"] = self.timings
        track, _ = infer_model_batched(combined, self.model, self.device, **infer_kwargs)
        result = track[overlap:]
        if len(result) != len(frames):
            raise VideoProcessingError("Ball inference returned an unexpected number of frame results")
        self.timings.frames += len(result)
        return result

    def timing_fields(self) -> dict[str, float | int]:
        return self.timings.to_log_fields()

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
