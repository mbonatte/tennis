from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from redis import Redis
from redis.exceptions import RedisError
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models import AnalysisJob, JobStatus, RenderOutput
from app.services.job_state import transition
from app.services.queue import enqueue_analysis, enqueue_render
from app.services.storage import delete_job_files, resolve_job_file, safe_job_dir, sanitize_filename, stream_upload
from app.services.workflow import create_workflow, format_duration
from app.services.workflow import workflow_rows as get_workflow_rows
from tennis_analyzer.errors import InvalidVideoError
from tennis_analyzer.schemas import AnalysisOptions, VisualizationOptions
from tennis_analyzer.video import probe_video, validate_video

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
templates.env.globals["workflow_rows"] = lambda workflow: get_workflow_rows(workflow, datetime.now(UTC))
templates.env.globals["format_duration"] = format_duration
logger = logging.getLogger(__name__)


def _job(db: Session, public_id: str) -> AnalysisJob:
    job = db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
    if not job:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return job


def _serialize(job: AnalysisJob) -> dict:
    return {
        "id": job.public_id,
        "filename": job.original_filename,
        "status": job.status.value,
        "stage": job.current_stage,
        "progress": job.progress,
        "workflow": get_workflow_rows(job.workflow, datetime.now(UTC)),
        "options": job.submitted_options,
        "video": {
            "duration_seconds": job.video_duration,
            "width": job.video_width,
            "height": job.video_height,
            "codec": job.video_codec,
        },
        "error": {"type": job.error_type, "message": job.error_message} if job.error_message else None,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "links": {
            "self": f"/jobs/{job.public_id}",
            "status": f"/api/jobs/{job.public_id}",
            "results": f"/api/jobs/{job.public_id}/results",
        },
    }


@router.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readiness(db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> JSONResponse:
    checks = {"database": False, "redis": False}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except SQLAlchemyError as exc:
        logger.warning("Database readiness check failed: %s", str(exc).splitlines()[0])
    try:
        checks["redis"] = bool(Redis.from_url(settings.redis_url).ping())
    except RedisError as exc:
        logger.warning("Redis readiness check failed: %s", str(exc).splitlines()[0])
    ready = all(checks.values())
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "checks": checks}, status_code=200 if ready else 503
    )


@router.get("/", response_class=HTMLResponse)
def home(request: Request, settings: Settings = Depends(get_settings)):
    return templates.TemplateResponse(
        request,
        "home.html",
        {"max_mb": settings.max_upload_bytes // 1024 // 1024, "max_minutes": settings.max_video_duration_seconds // 60},
    )


@router.get("/jobs", response_class=HTMLResponse)
def jobs_page(request: Request, db: Session = Depends(get_db)):
    jobs = db.scalars(select(AnalysisJob).order_by(AnalysisJob.created_at.desc()).limit(100)).all()
    return templates.TemplateResponse(request, "jobs.html", {"jobs": jobs})


@router.post("/jobs")
@router.post("/api/jobs")
async def create_job(
    request: Request,
    video: UploadFile = File(...),
    ball_tracking: bool = Form(False),
    court_detection: bool = Form(False),
    player_tracking: bool = Form(False),
    pose_tracking: bool = Form(False),
    bounce_detection: bool = Form(False),
    scene_cut_detection: bool = Form(True),
    statistics: bool = Form(False),
    point_analysis: bool = Form(False),
    ball_trail: bool = Form(False),
    bounce_markers: bool = Form(False),
    frame_number: bool = Form(False),
    court_overlay: bool = Form(False),
    court_keypoints: bool = Form(False),
    player_boxes: bool = Form(False),
    player_poses: bool = Form(False),
    statistics_overlay: bool = Form(False),
    ball_history_plot: bool = Form(False),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    filename = sanitize_filename(video.filename)
    extension = Path(filename).suffix.lower()
    if extension not in settings.allowed_video_extensions:
        raise HTTPException(415, "Unsupported file extension")
    if video.content_type and not (
        video.content_type.startswith("video/") or video.content_type == "application/octet-stream"
    ):
        raise HTTPException(415, "The upload MIME type is not a video")
    analysis = AnalysisOptions(
        ball_tracking,
        court_detection,
        player_tracking,
        pose_tracking,
        bounce_detection,
        scene_cut_detection,
        statistics,
        point_analysis,
    ).validated()
    visual = VisualizationOptions(
        ball_trail,
        bounce_markers,
        frame_number,
        court_overlay,
        court_keypoints,
        player_boxes,
        player_poses,
        statistics_overlay,
        ball_history_plot,
    )
    try:
        visual.validate_for(analysis)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    public_id = str(uuid.uuid4())
    directory = safe_job_dir(settings.data_root, public_id)
    stored = f"source{extension}"
    input_path = directory / "input" / stored
    try:
        size = await stream_upload(
            video, input_path, limit=settings.max_upload_bytes, chunk_size=settings.upload_chunk_bytes
        )
        metadata = probe_video(input_path, allowed_codecs=set(settings.allowed_video_codecs))
        validate_video(
            metadata,
            max_duration=settings.max_video_duration_seconds,
            max_width=settings.max_video_width,
            max_height=settings.max_video_height,
        )
    except InvalidVideoError as exc:
        delete_job_files(settings.data_root, public_id)
        raise HTTPException(422, str(exc)) from exc
    job = AnalysisJob(
        public_id=public_id,
        original_filename=filename,
        stored_filename=stored,
        input_relative_path=str(input_path.resolve().relative_to(settings.data_root.resolve())),
        input_size=size,
        submitted_options={"analysis": asdict(analysis), "visualization": asdict(visual)},
        workflow=create_workflow(analysis, visual),
        video_duration=metadata.duration_seconds,
        video_width=metadata.width,
        video_height=metadata.height,
        video_codec=metadata.video_codec,
    )
    db.add(job)
    db.flush()
    transition(job, JobStatus.queued)
    try:
        job.queue_job_id = enqueue_analysis(public_id, settings)
        db.commit()
    except Exception as exc:
        logger.exception("Could not enqueue analysis job %s", public_id)
        transition(job, JobStatus.failed)
        job.error_type = "queue_unavailable"
        job.error_message = "The analysis queue is temporarily unavailable. Please try again later."
        db.commit()
        raise HTTPException(503, job.error_message) from exc
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            _serialize(job), status_code=status.HTTP_202_ACCEPTED, headers={"Location": f"/api/jobs/{public_id}"}
        )
    return RedirectResponse(f"/jobs/{public_id}", status_code=303)


@router.get("/jobs/{public_id}", response_class=HTMLResponse)
def job_page(public_id: str, request: Request, db: Session = Depends(get_db)):
    job = _job(db, public_id)
    result = _load_result(job, get_settings()) if job.status == JobStatus.completed else None
    return templates.TemplateResponse(request, "job.html", {"job": job, "result": result})


@router.get("/jobs/{public_id}/status", response_class=HTMLResponse)
def job_status_fragment(public_id: str, request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(request, "_job_status.html", {"job": _job(db, public_id)})


@router.post("/api/jobs/{public_id}/renders")
def create_render(public_id: str, visual: VisualizationOptions, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)):
    job = _job(db, public_id)
    if job.status != JobStatus.completed or not job.analysis_artifact_relative_path:
        raise HTTPException(409, "Analysis must finish before rendering")
    render = RenderOutput(analysis=job, status=JobStatus.queued, visualization_options=asdict(visual))
    db.add(render)
    db.flush()
    enqueue_render(render.public_id, settings)
    db.commit()
    return {"id": render.public_id, "status": render.status.value}


@router.post("/jobs/{public_id}/renders")
def create_render_from_form(
    public_id: str,
    request: Request,
    ball_trail: bool = Form(False),
    bounce_markers: bool = Form(False),
    frame_number: bool = Form(False),
    court_overlay: bool = Form(False),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    job = _job(db, public_id)
    if job.status != JobStatus.completed or not job.analysis_artifact_relative_path:
        raise HTTPException(409, "Analysis must finish before rendering")
    visual = VisualizationOptions(ball_trail=ball_trail, bounce_markers=bounce_markers, frame_number=frame_number, court_overlay=court_overlay)
    render = RenderOutput(analysis=job, status=JobStatus.queued, visualization_options=asdict(visual))
    db.add(render)
    db.flush()
    enqueue_render(render.public_id, settings)
    db.commit()
    return RedirectResponse(f"/jobs/{public_id}", 303)


@router.get("/api/jobs/{public_id}")
def job_status(public_id: str, db: Session = Depends(get_db)) -> dict:
    return _serialize(_job(db, public_id))


@router.get("/api/jobs/{public_id}/results")
def job_results(public_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)) -> dict:
    job = _job(db, public_id)
    if job.status != JobStatus.completed:
        raise HTTPException(409, "Analysis is not completed")
    return _load_result(job, settings)


def _load_result(job: AnalysisJob, settings: Settings) -> dict:
    if not job.result_relative_path:
        raise HTTPException(500, "Completed analysis has no result file")
    path = resolve_job_file(settings.data_root, job.result_relative_path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "Result file is unavailable") from exc


@router.post("/api/jobs/{public_id}/cancel")
@router.post("/jobs/{public_id}/cancel")
def cancel_job(public_id: str, request: Request, db: Session = Depends(get_db)):
    job = _job(db, public_id)
    if job.status not in {JobStatus.queued, JobStatus.running}:
        raise HTTPException(409, "Only queued or running jobs can be cancelled")
    job.cancellation_requested = True
    if job.status == JobStatus.queued:
        transition(job, JobStatus.cancelled)
    db.commit()
    return RedirectResponse(f"/jobs/{public_id}", 303) if not request.url.path.startswith("/api/") else _serialize(job)


@router.delete("/api/jobs/{public_id}", status_code=204)
@router.post("/jobs/{public_id}/delete")
def delete_job(
    public_id: str, request: Request, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)
):
    job = _job(db, public_id)
    if job.status in {JobStatus.queued, JobStatus.running}:
        raise HTTPException(409, "Cancel the running analysis before deleting it")
    db.delete(job)
    db.commit()
    delete_job_files(settings.data_root, public_id)
    if request.method == "POST":
        return RedirectResponse("/jobs", 303)
    return JSONResponse(None, status_code=204)


@router.get("/jobs/{public_id}/files/{kind}")
def job_file(
    public_id: str,
    kind: str,
    request: Request,
    download: bool = False,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    job = _job(db, public_id)
    if job.status != JobStatus.completed:
        raise HTTPException(409, "Analysis is not completed")
    relative = (
        job.output_video_relative_path if kind == "video" else job.result_relative_path if kind == "json" else None
    )
    if not relative:
        raise HTTPException(404, "File not found")
    path = resolve_job_file(settings.data_root, relative)
    if not path.is_file():
        raise HTTPException(404, "File not found")
    media_type = "video/mp4" if kind == "video" else "application/json"
    if download or kind == "json":
        return FileResponse(
            path, media_type=media_type, filename=f"{Path(job.original_filename).stem}-analysis{path.suffix}"
        )
    return _range_response(path, request.headers.get("range"), media_type)


@router.get("/jobs/{public_id}/points/{number}")
def point_video(
    public_id: str,
    number: int,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    job = _job(db, public_id)
    if job.status != JobStatus.completed or not job.result_relative_path:
        raise HTTPException(409, "Analysis is not completed")
    result = _load_result(job, settings)
    point = next((item for item in result.get("points", []) if item.get("number") == number), None)
    if not point:
        raise HTTPException(404, "Point video not found")
    result_path = resolve_job_file(settings.data_root, job.result_relative_path)
    path = (result_path.parent / str(point["video"])).resolve()
    if result_path.parent not in path.parents or not path.is_file():
        raise HTTPException(404, "Point video not found")
    return _range_response(path, request.headers.get("range"), "video/mp4")


def _range_response(path: Path, range_header: str | None, media_type: str):
    size = path.stat().st_size
    if not range_header:
        return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})
    try:
        units, requested = range_header.split("=", 1)
        if units != "bytes" or "," in requested:
            raise ValueError
        start_text, end_text = requested.split("-", 1)
        start = int(start_text) if start_text else max(0, size - int(end_text))
        end = min(size - 1, int(end_text)) if end_text else size - 1
        if start < 0 or end < start or start >= size:
            raise ValueError
    except ValueError:
        return JSONResponse(
            {"detail": "Invalid byte range"}, status_code=416, headers={"Content-Range": f"bytes */{size}"}
        )

    def iterator():
        remaining = end - start + 1
        with path.open("rb") as file:
            file.seek(start)
            while remaining:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        iterator(),
        status_code=206,
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
        },
    )
