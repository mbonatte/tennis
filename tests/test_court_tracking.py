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

        def infer_model(self, frames, batch_size=1):
            assert batch_size == 1
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
        def infer_model(self, frames, batch_size=1):
            return [None], [None]

    tracker = court.CourtTracker(BadDetector(), torch.device("cpu"))

    with pytest.raises(Exception, match="unexpected number"):
        tracker.process_chunk(_frames(2))


def test_court_overlay_draws_in_place_and_reuses_cached_template():
    court.get_court_img.cache_clear()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    result = court.draw_court_overlay_in_place(frame, None)
    court.draw_court_overlay_in_place(frame, None)

    assert result is frame
    assert court.get_court_img.cache_info().misses == 1
    assert court.get_court_img.cache_info().hits == 1


def test_court_overlay_projects_prior_bounces(monkeypatch):
    calls = []
    monkeypatch.setattr(court, "_draw_projected_point", lambda *args, **kwargs: calls.append((args, kwargs)))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    homography = np.eye(3, dtype=np.float32)

    court.draw_court_overlay_in_place(frame, homography, bounce_history=[((12, 18), homography)])

    assert calls[0][0][2] == (12, 18)
    assert calls[0][1]["thickness"] == -1
