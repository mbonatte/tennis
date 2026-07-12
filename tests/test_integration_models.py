from __future__ import annotations

import os
from pathlib import Path

import pytest

from tennis_analyzer.pipeline import analyze_video
from tennis_analyzer.schemas import AnalysisOptions, PipelineOptions


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_MODEL_INTEGRATION") != "1", reason="set RUN_MODEL_INTEGRATION=1")
def test_real_models_on_tiny_video(sample_video: Path, tmp_path: Path):
    model_root = Path(os.environ.get("MODEL_ROOT", "models"))
    options = PipelineOptions(
        analysis=AnalysisOptions(ball_tracking=True, court_detection=True, bounce_detection=True, statistics=True),
        chunk_size=16,
        device=os.environ.get("DEVICE", "cpu"),
    )
    result = analyze_video(sample_video, tmp_path / "real-model-output", options, model_root=model_root)
    assert result.pipeline_version and (tmp_path / "real-model-output" / "analyzed.mp4").is_file()
