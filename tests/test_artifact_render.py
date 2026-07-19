from pathlib import Path

import cv2

from tennis_analyzer.pipeline.artifact import read_artifact, write_artifact
from tennis_analyzer.pipeline.service import render_from_artifact
from tennis_analyzer.schemas import VisualizationOptions


def test_artifact_round_trip_is_json_safe(tmp_path: Path):
    path = tmp_path / "analysis.json"
    write_artifact(path, {"frame_count": 2, "ball_track": [(1.5, 2.5), (None, None)]})

    restored = read_artifact(path)

    assert restored["schema_version"] == 1
    assert restored["ball_track"] == [[1.5, 2.5], [None, None]]


def test_render_from_artifact_streams_without_models(sample_video: Path, tmp_path: Path):
    capture = cv2.VideoCapture(str(sample_video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    artifact = tmp_path / "analysis.json"
    write_artifact(
        artifact,
        {
            "frame_count": frame_count,
            "ball_track": [[20 + index, 30] for index in range(frame_count)],
            "homographies": [None] * frame_count,
            "court_keypoints": [None] * frame_count,
            "player_tracks": [[] for _ in range(frame_count)],
            "bounces": [4],
            "summary": {"shot_count": 1, "bounce_count": 1},
        },
    )

    output = render_from_artifact(
        sample_video,
        artifact,
        tmp_path / "render",
        VisualizationOptions(ball_trail=True, bounce_markers=True, frame_number=True),
    )

    rendered = cv2.VideoCapture(str(output))
    assert rendered.isOpened()
    assert int(rendered.get(cv2.CAP_PROP_FRAME_COUNT)) == frame_count
    rendered.release()
