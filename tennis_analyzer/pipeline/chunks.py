from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tennis_analyzer.errors import VideoProcessingError


@dataclass(frozen=True)
class FrameChunk:
    """A contiguous, non-overlapping range of decoded source frames.

    ``start_frame`` is the zero-based global index of ``frames[0]``. Every
    consumer must emit exactly one result per frame in the same order.
    Model-specific temporal overlap is added by that model's adapter and is
    never included in this source-frame contract.
    """

    start_frame: int
    frames: list[np.ndarray]

    @property
    def end_frame(self) -> int:
        return self.start_frame + len(self.frames)


def iter_frame_chunks(
    path: Path, size: int, *, on_decode: Callable[[int, float], None] | None = None
) -> Iterator[FrameChunk]:
    """Decode a video incrementally and always release its capture handle."""
    if size <= 0:
        raise ValueError("chunk size must be positive")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError("OpenCV could not decode the input video")

    start = 0
    read_any = False
    try:
        while True:
            started = time.perf_counter()
            frames: list[np.ndarray] = []
            for _ in range(size):
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                break
            read_any = True
            if on_decode is not None:
                on_decode(len(frames), time.perf_counter() - started)
            yield FrameChunk(start, frames)
            start += len(frames)
    finally:
        capture.release()

    if not read_any:
        raise VideoProcessingError("The video contains no readable frames")
