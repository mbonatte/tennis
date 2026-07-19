from pathlib import Path

import numpy as np
import pytest

court = pytest.importorskip("court")
torch = court.torch


def _frames(count):
    return [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(count)]


def test_court_tracker_loads_once_and_reuses_detector(monkeypatch):
    created = []

    class FakeDetector:
        def __init__(self, path, device):
            created.append((Path(path), str(device)))

        def infer_model(self, frames):
            return [None] * len(frames), [None] * len(frames)

    monkeypatch.setattr(court, "CourtDetectorNet", FakeDetector)

    tracker = court.CourtTracker.from_checkpoint(Path("court.pt"), "cpu")
    first = tracker.process_chunk(_frames(2))
    second = tracker.process_chunk(_frames(1))
    tracker.close()

    assert created == [(Path("court.pt"), "cpu")]
    assert len(first[0]) == 2 and len(second[0]) == 1
    assert tracker.detector is None


def test_court_tracker_rejects_misaligned_results():
    class BadDetector:
        def infer_model(self, frames):
            return [None], [None]

    tracker = court.CourtTracker(BadDetector(), torch.device("cpu"))

    with pytest.raises(Exception, match="unexpected number"):
        tracker.process_chunk(_frames(2))
