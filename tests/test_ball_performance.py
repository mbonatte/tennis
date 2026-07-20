import numpy as np

import ball


def test_reusable_batch_input_matches_legacy_temporal_channel_order():
    frames = [np.full((18, 24, 3), index * 10, dtype=np.uint8) for index in range(5)]
    resized = ball._resized_frames(frames)
    output = np.empty((3, 9, ball.MODEL_HEIGHT, ball.MODEL_WIDTH), dtype=np.float32)

    prepared = ball._fill_input_batch(resized, 0, 3, output)

    for index in range(3):
        legacy = np.transpose(
            np.concatenate([resized[index + 2], resized[index + 1], resized[index]], axis=2).astype(np.float32) / 255.0,
            (2, 0, 1),
        )
        np.testing.assert_allclose(prepared[index], legacy)


def test_preprocessing_resizes_each_source_frame_once(monkeypatch):
    frames = [np.zeros((18, 24, 3), dtype=np.uint8) for _ in range(5)]
    calls = []
    original_resize = ball.cv2.resize

    def counted_resize(*args, **kwargs):
        calls.append(1)
        return original_resize(*args, **kwargs)

    monkeypatch.setattr(ball.cv2, "resize", counted_resize)
    ball._resized_frames(frames)

    assert len(calls) == len(frames)
