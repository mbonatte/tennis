import numpy as np

from tennis_analyzer.pipeline.stages import PlayerStage


def test_player_stage_does_not_accumulate_raw_model_results():
    raw_results = []

    class Tracker:
        def track_frame(self, frame):
            raw = object()
            raw_results.append(raw)
            return [int(frame[0, 0, 0])], raw

        def track_frames(self, frames):
            raise AssertionError("chunk-level raw results must not be accumulated")

        def close(self):
            pass

        def draw(self, frame, tracks, *, copy_frame):
            assert copy_frame is False
            return frame

    frames = [np.full((1, 1, 3), index, dtype=np.uint8) for index in range(3)]

    tracks = PlayerStage(Tracker()).process_chunk(frames)

    assert tracks == [[0], [1], [2]]
    assert len(raw_results) == 3
    assert PlayerStage(Tracker()).draw(frames[0], tracks[0]) is frames[0]
