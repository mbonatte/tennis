from __future__ import annotations

import logging
import traceback

from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import job_id_context
from app.db.session import SessionLocal
from app.models import AnalysisJob, JobStatus
from app.services.job_state import transition
from app.services.storage import resolve_job_file, safe_job_dir
from tennis_analyzer.errors import AnalysisCancelled, AnalysisError
from tennis_analyzer.pipeline import analyze_video
from tennis_analyzer.schemas import AnalysisOptions, PipelineOptions, VisualizationOptions

logger = logging.getLogger(__name__)


def run_analysis_job(public_id: str) -> None:
    settings = get_settings()
    token = job_id_context.set(public_id)
    try:
        with SessionLocal() as db:
            job = db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
            if not job or job.status != JobStatus.queued:
                logger.warning("Job does not exist or is no longer queued")
                return
            transition(job, JobStatus.running, stage="preparing")
            db.commit()

        def progress(stage: str, percent: int, message: str) -> None:
            with SessionLocal() as progress_db:
                current = progress_db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
                if current and current.status == JobStatus.running:
                    current.current_stage = stage
                    current.progress = max(current.progress, min(99, percent))
                    progress_db.commit()
            logger.info("%s", message, extra={"stage": stage, "progress": percent})

        def cancelled() -> bool:
            with SessionLocal() as check_db:
                current = check_db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
                return not current or current.cancellation_requested

        with SessionLocal() as db:
            job = db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
            assert job is not None
            values = job.submitted_options
            pipeline_options = PipelineOptions(
                analysis=AnalysisOptions(**values["analysis"]),
                visualization=VisualizationOptions(**values["visualization"]),
                chunk_size=settings.analysis_chunk_frames,
                device=settings.device,
            )
            input_path = resolve_job_file(settings.data_root, job.input_relative_path)
            output_dir = safe_job_dir(settings.data_root, public_id) / "output"
        result = analyze_video(input_path, output_dir, pipeline_options, progress, cancelled, settings.model_root)
        with SessionLocal() as db:
            job = db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
            if not job:
                return
            if job.cancellation_requested:
                transition(job, JobStatus.cancelled)
            else:
                transition(job, JobStatus.completed)
                job.output_video_relative_path = str(
                    (output_dir / result.output_video).relative_to(settings.data_root.resolve())
                )
                job.result_relative_path = str(
                    (output_dir / result.result_json).relative_to(settings.data_root.resolve())
                )
                job.output_size = (output_dir / result.output_video).stat().st_size
            db.commit()
    except AnalysisCancelled:
        _mark_terminal(public_id, JobStatus.cancelled, "Analysis cancelled", None)
    except AnalysisError as exc:
        logger.exception("Analysis failed")
        _mark_terminal(public_id, JobStatus.failed, str(exc), traceback.format_exc())
    except Exception:
        logger.exception("Unexpected worker failure")
        _mark_terminal(public_id, JobStatus.failed, "An unexpected processing error occurred", traceback.format_exc())
    finally:
        job_id_context.reset(token)


def _mark_terminal(public_id: str, status: JobStatus, message: str, diagnostic: str | None) -> None:
    with SessionLocal() as db:
        job = db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
        if not job or job.status not in {JobStatus.queued, JobStatus.running}:
            return
        transition(job, status)
        job.error_type = status.value
        job.error_message = message[:2000]
        job.internal_diagnostic = diagnostic
        db.commit()
