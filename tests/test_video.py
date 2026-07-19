from pathlib import Path

import pytest

from tennis_analyzer.errors import InvalidVideoError, VideoProcessingError
from tennis_analyzer.video import normalize_video, probe_video, validate_video


def test_probe_and_limits(sample_video: Path):
    metadata = probe_video(sample_video)
    assert (metadata.width, metadata.height, metadata.video_codec) == (320, 240, "h264")
    with pytest.raises(InvalidVideoError, match="duration limit"):
        validate_video(metadata, max_duration=0, max_width=1000, max_height=1000)


def test_invalid_video_is_rejected(tmp_path: Path):
    fake = tmp_path / "fake.mp4"
    fake.write_text("not a video")
    with pytest.raises(InvalidVideoError):
        probe_video(fake)


def test_ffmpeg_normalization_builds_argument_array(monkeypatch, tmp_path: Path):
    raw, source, output = tmp_path / "raw.mp4", tmp_path / "source.mp4", tmp_path / "final.mp4"
    raw.touch()
    source.touch()
    captured = {}

    def fake_run(command, timeout):
        captured["command"] = command
        Path(command[-1]).write_bytes(b"video")

    monkeypatch.setattr("tennis_analyzer.video._run", fake_run)
    normalize_video(raw, source, output, timeout=30)
    assert captured["command"][0] == "ffmpeg"
    assert "libx264" in captured["command"]
    assert output.read_bytes() == b"video"


def test_ffmpeg_normalization_removes_partial_temporary_file(monkeypatch, tmp_path: Path):
    raw, source, output = tmp_path / "raw.mp4", tmp_path / "source.mp4", tmp_path / "final.mp4"
    raw.touch()
    source.touch()

    def failing_run(command, timeout):
        Path(command[-1]).write_bytes(b"partial")
        raise VideoProcessingError("conversion failed")

    monkeypatch.setattr("tennis_analyzer.video._run", failing_run)

    with pytest.raises(VideoProcessingError, match="conversion failed"):
        normalize_video(raw, source, output, timeout=30)

    assert not output.with_suffix(".tmp.mp4").exists()
    assert not output.exists()
