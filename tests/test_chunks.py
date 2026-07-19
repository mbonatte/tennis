from pathlib import Path

import numpy as np
import pytest

from tennis_analyzer.errors import VideoProcessingError
from tennis_analyzer.pipeline.chunks import iter_frame_chunks


class FakeCapture:
    def __init__(self, frames, *, opened=True, fail_after=None):
        self.frames = iter(frames)
        self.opened = opened
        self.fail_after = fail_after
        self.read_count = 0
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.fail_after is not None and self.read_count >= self.fail_after:
            return False, None
        self.read_count += 1
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self):
        self.released = True


def _frames(count):
    return [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(count)]


@pytest.mark.parametrize(
    ("frame_count", "chunk_size", "expected"),
    [
        (1, 1, [(0, 1)]),
        (2, 1, [(0, 1), (1, 1)]),
        (4, 4, [(0, 4)]),
        (5, 4, [(0, 4), (4, 1)]),
        (10, 4, [(0, 4), (4, 4), (8, 2)]),
    ],
)
def test_frame_chunks_have_stable_global_indices(monkeypatch, frame_count, chunk_size, expected):
    capture = FakeCapture(_frames(frame_count))
    monkeypatch.setattr("tennis_analyzer.pipeline.chunks.cv2.VideoCapture", lambda _path: capture)

    chunks = list(iter_frame_chunks(Path("video.mp4"), chunk_size))

    assert [(chunk.start_frame, len(chunk.frames)) for chunk in chunks] == expected
    assert [int(frame[0, 0, 0]) for chunk in chunks for frame in chunk.frames] == list(range(frame_count))
    assert capture.released


def test_frame_reader_releases_when_consumer_fails(monkeypatch):
    capture = FakeCapture(_frames(3))
    monkeypatch.setattr("tennis_analyzer.pipeline.chunks.cv2.VideoCapture", lambda _path: capture)

    with pytest.raises(RuntimeError, match="inference"):
        for _chunk in iter_frame_chunks(Path("video.mp4"), 2):
            raise RuntimeError("inference failed")

    assert capture.released


def test_frame_reader_rejects_empty_video(monkeypatch):
    capture = FakeCapture([])
    monkeypatch.setattr("tennis_analyzer.pipeline.chunks.cv2.VideoCapture", lambda _path: capture)

    with pytest.raises(VideoProcessingError, match="no readable frames"):
        list(iter_frame_chunks(Path("video.mp4"), 2))

    assert capture.released


def test_frame_reader_rejects_nonpositive_chunk_size():
    with pytest.raises(ValueError, match="positive"):
        list(iter_frame_chunks(Path("video.mp4"), 0))
