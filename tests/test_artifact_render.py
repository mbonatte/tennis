from pathlib import Path

import cv2
import numpy as np
import pytest

from tennis_analyzer.errors import VideoProcessingError
from tennis_analyzer.pipeline.artifact import read_artifact, write_artifact
from tennis_analyzer.pipeline.service import _draw_ball_trail, _draw_saved_players, render_from_artifact
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


def test_render_removes_partial_video_when_artifact_frame_count_is_wrong(sample_video: Path, tmp_path: Path):
    capture = cv2.VideoCapture(str(sample_video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    artifact = tmp_path / "analysis.json"
    write_artifact(
        artifact,
        {
            "frame_count": frame_count + 1,
            "ball_track": [[None, None] for _ in range(frame_count + 1)],
            "homographies": [None] * (frame_count + 1),
            "court_keypoints": [None] * (frame_count + 1),
            "player_tracks": [[] for _ in range(frame_count + 1)],
            "bounces": [],
            "summary": {},
        },
    )
    destination = tmp_path / "render"

    with pytest.raises(VideoProcessingError, match="frame count"):
        render_from_artifact(sample_video, artifact, destination, VisualizationOptions())

    assert not (destination / ".rendered.mp4").exists()
    assert not (destination / "rendered.mp4").exists()


def test_saved_player_boxes_use_render_specific_names(monkeypatch):
    labels: list[str] = []
    monkeypatch.setattr(cv2, "putText", lambda frame, text, *args: labels.append(text) or frame)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    _draw_saved_players(
        frame,
        [{"role": "bottom_player", "bbox": [10, 10, 40, 60]}],
        boxes=True,
        poses=False,
        top_label="Ana",
        bottom_label="Mauricio",
    )

    assert labels == ["Mauricio"]


def test_render_uses_selected_ball_trail_and_player_box_colors(monkeypatch):
    circles = []
    rectangles = []
    monkeypatch.setattr(cv2, "circle", lambda *args, **kwargs: circles.append(args) or args[0])
    monkeypatch.setattr(cv2, "rectangle", lambda *args, **kwargs: rectangles.append(args) or args[0])
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    visual = VisualizationOptions(ball_trail=True, ball_trail_color="#12ab34", ball_trail_size=5, ball_trail_length=2)

    _draw_ball_trail(frame, [(10, 10), (20, 20)], 1, visual)
    _draw_saved_players(
        frame,
        [{"role": "bottom_player", "bbox": [10, 10, 40, 60]}],
        boxes=True,
        poses=False,
        top_label="Top",
        bottom_label="Bottom",
        box_color=visual.bgr_color("ball_trail_color"),
    )

    assert all(call[3] == (52, 171, 18) for call in circles)
    assert rectangles[0][3] == (52, 171, 18)
