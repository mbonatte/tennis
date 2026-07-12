import json
from pathlib import Path

from tennis_analyzer.pipeline import analyze_video
from tennis_analyzer.schemas import PipelineOptions


def test_pipeline_without_models_streams_and_serializes(sample_video: Path, tmp_path: Path):
    progress = []
    result = analyze_video(
        sample_video,
        tmp_path / "output",
        PipelineOptions(chunk_size=16),
        lambda stage, pct, message: progress.append((stage, pct)),
    )
    data = json.loads((tmp_path / "output" / "result.json").read_text())
    assert (tmp_path / "output" / "analyzed.mp4").is_file()
    assert data["schema_version"] == "1.0"
    assert data["summary"]["frames_processed"] == 10
    assert progress[-1] == ("completed", 100)
    assert result.output_video == "analyzed.mp4"
