import json

import numpy as np

from tennis_analyzer.pipeline.artifact import read_artifact, write_artifact
from tennis_analyzer.pipeline.court_calibration import create_static_calibration
from tennis_analyzer.pipeline.service import recompute_court_dependent_events


class _Stats:
    def to_dict(self):
        return {
            "shot_events": [{"frame": 1, "player_role": "bottom_player"}],
            "bounce_events": [{"frame": 2, "in_out": "in", "court_region": "singles_court"}],
            "player_stats": {"bottom_player": {"shots": 1}},
            "average_ball_speed_kmh": 10.0,
            "max_ball_speed_kmh": 20.0,
        }


def test_court_calibration_recomputes_saved_court_events_without_models(monkeypatch, tmp_path):
    artifact_path = tmp_path / "analysis-artifact.json"
    result_path = tmp_path / "result.json"
    write_artifact(
        artifact_path,
        {
            "frame_count": 3,
            "metadata": {"fps": 30},
            "analysis_options": {"statistics": True},
            "ball_track": [[10, 10], [11, 11], [12, 12]],
            "homographies": [None, None, None],
            "court_keypoints": [None, None, None],
            "player_tracks": [[], [], []],
            "bounces": [2],
            "scene_cuts": [],
            "summary": {"frames_processed": 3},
        },
    )
    result_path.write_text(json.dumps({"shots": [], "bounces": [], "summary": {}}), encoding="utf-8")
    called = {}

    def fake_stats(ball_track, bounces, fps, *, homography_matrices, player_tracks, scene_cuts):
        called.update(bounces=bounces, fps=fps, homographies=homography_matrices)
        return _Stats()

    monkeypatch.setattr("analysis.compute_match_stats", fake_stats)
    calibration = create_static_calibration(0, [[10, 10], [300, 10], [300, 220], [10, 220]])

    recompute_court_dependent_events(artifact_path, result_path, calibration)

    updated = read_artifact(artifact_path)
    assert called["bounces"] == {2}
    assert called["fps"] == 30
    assert len(called["homographies"]) == 3
    assert all(np.isfinite(homography).all() for homography in called["homographies"])
    assert updated["bounces"] == [2]
    assert updated["bounce_events"][0]["court_region"] == "singles_court"
    assert json.loads(result_path.read_text(encoding="utf-8"))["bounces"][0]["in_out"] == "in"
