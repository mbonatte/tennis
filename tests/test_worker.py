from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models import AnalysisJob, JobStatus, RenderOutput
from app.workers.tasks import run_analysis_job, run_render_job
from tennis_analyzer.schemas import AnalysisOptions, AnalysisResult, VideoMetadata, VisualizationOptions


def test_worker_invokes_pipeline_and_records_progress(monkeypatch, sample_video: Path):
    settings = get_settings()
    public_id = "123e4567-e89b-12d3-a456-426614174000"
    input_path = settings.data_root / "jobs" / public_id / "input" / "source.mp4"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(sample_video.read_bytes())
    with SessionLocal() as db:
        job = AnalysisJob(
            public_id=public_id,
            original_filename="match.mp4",
            stored_filename="source.mp4",
            status=JobStatus.queued,
            current_stage="queued",
            input_relative_path=str(input_path.relative_to(settings.data_root)),
            submitted_options={"analysis": vars(AnalysisOptions()), "visualization": vars(VisualizationOptions())},
        )
        db.add(job)
        db.commit()

    def fake_analyze(input_path, output_dir, options, progress, cancelled, model_root):
        output_dir.mkdir(parents=True, exist_ok=True)
        progress("rendering", 55, "halfway")
        progress("completed", 100, "done")
        (output_dir / "analyzed.mp4").write_bytes(b"video")
        (output_dir / "result.json").write_text("{}")
        return AnalysisResult(
            "source.mp4",
            "analyzed.mp4",
            "result.json",
            VideoMetadata(1, 320, 240, 10, 10, "mp4", "h264", None, 5),
            {},
            {},
        )

    monkeypatch.setattr("app.workers.tasks.analyze_video", fake_analyze)
    run_analysis_job(public_id)
    with SessionLocal() as db:
        job = db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
        assert job.status == JobStatus.completed and job.progress == 100 and job.output_size == 5
        assert all(stage["status"] == "completed" for stage in job.workflow["stages"])
        assert "completed" not in {stage["key"] for stage in job.workflow["stages"]}


def test_render_worker_uses_saved_artifact_without_analysis_models(monkeypatch, sample_video: Path):
    settings = get_settings()
    public_id = "223e4567-e89b-12d3-a456-426614174000"
    input_path = settings.data_root / "jobs" / public_id / "input" / "source.mp4"
    artifact_path = settings.data_root / "jobs" / public_id / "output" / "analysis-artifact.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(sample_video.read_bytes())
    artifact_path.write_text("{}")
    with SessionLocal() as db:
        job = AnalysisJob(
            public_id=public_id,
            original_filename="match.mp4",
            stored_filename="source.mp4",
            status=JobStatus.completed,
            current_stage="completed",
            input_relative_path=str(input_path.relative_to(settings.data_root)),
            analysis_artifact_relative_path=str(artifact_path.relative_to(settings.data_root)),
            submitted_options={},
        )
        render = RenderOutput(
            analysis=job,
            status=JobStatus.queued,
            visualization_options=vars(VisualizationOptions(frame_number=True)),
        )
        db.add_all([job, render])
        db.commit()
        render_id = render.public_id

    def fake_render(source, artifact, destination, options, progress):
        assert artifact == artifact_path
        progress("rendering", 50, "half")
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / "rendered.mp4"
        output.write_bytes(b"render")
        return output

    monkeypatch.setattr("app.workers.tasks.render_from_artifact", fake_render)
    run_render_job(render_id)

    with SessionLocal() as db:
        render = db.scalar(select(RenderOutput).where(RenderOutput.public_id == render_id))
        assert render.status == JobStatus.completed
        assert render.progress == 100 and render.output_relative_path.endswith("rendered.mp4")
