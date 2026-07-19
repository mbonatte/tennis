from pathlib import Path

import numpy as np
import pytest

ball = pytest.importorskip("ball")
torch = ball.torch


def _frames(count):
    return [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(count)]


@pytest.mark.parametrize("frame_count", [1, 2, 3, 4, 5, 8, 9])
@pytest.mark.parametrize("chunk_size", [1, 2, 4, 8])
def test_ball_tracker_preserves_temporal_context_and_result_length(monkeypatch, frame_count, chunk_size):
    calls = []

    def fake_infer(frames, model, device, batch_size, use_amp):
        values = [int(frame[0, 0, 0]) for frame in frames]
        calls.append(values)
        track = [(None, None)] * min(2, len(values))
        track.extend((float(value), float(value)) for value in values[2:])
        return track, [-1.0] * len(values)

    monkeypatch.setattr(ball, "infer_model_batched", fake_infer)
    tracker = ball.BallTracker(object(), torch.device("cpu"), batch_size=8)
    frames = _frames(frame_count)

    results = []
    for start in range(0, frame_count, chunk_size):
        results.extend(tracker.process_chunk(frames[start : start + chunk_size]))

    assert len(results) == frame_count
    assert calls[0] == list(range(min(chunk_size, frame_count)))
    for previous, current in zip(calls, calls[1:], strict=False):
        assert current[:2] == previous[-2:] if len(previous) >= 2 else current[: len(previous)] == previous
    assert [point[0] for point in results[2:]] == [float(index) for index in range(2, frame_count)]


def test_ball_tracker_loads_checkpoint_once_for_many_chunks(monkeypatch):
    loaded = []
    model = object()
    monkeypatch.setattr(ball, "load_model", lambda path, device, use_compile=False: loaded.append(path) or model)
    monkeypatch.setattr(
        ball,
        "infer_model_batched",
        lambda frames, model, device, batch_size, use_amp: ([(None, None)] * len(frames), [-1.0] * len(frames)),
    )

    tracker = ball.BallTracker.from_checkpoint(Path("ball.pt"), "cpu")
    tracker.process_chunk(_frames(3))
    tracker.process_chunk(_frames(3))
    tracker.close()

    assert loaded == [Path("ball.pt")]
    assert tracker.model is None


def test_global_postprocessing_interpolates_across_former_chunk_boundary():
    raw = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (None, None), (4.0, 4.0), (5.0, 5.0), (6.0, 6.0)]

    processed = ball.postprocess_ball_track(raw)

    assert processed[3] == (3.0, 3.0)


def test_chunked_and_unchunked_global_postprocessing_are_equivalent():
    raw = [(float(index), float(index)) for index in range(8)]
    raw[3] = (None, None)

    assert ball.postprocess_ball_track(raw[:4] + raw[4:]) == ball.postprocess_ball_track(raw)


def test_global_postprocessing_normalizes_nonfinite_missing_values():
    raw = [(None, None), (float("nan"), float("nan")), (float("inf"), 2.0)]

    assert ball.postprocess_ball_track(raw) == [(None, None), (None, None), (None, None)]
