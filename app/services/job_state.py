from __future__ import annotations

from datetime import UTC, datetime

from app.models import AnalysisJob, JobStatus

ALLOWED = {
    JobStatus.uploaded: {JobStatus.queued, JobStatus.failed},
    JobStatus.queued: {JobStatus.running, JobStatus.cancelled, JobStatus.failed},
    JobStatus.running: {JobStatus.completed, JobStatus.failed, JobStatus.cancelled},
    JobStatus.completed: set(),
    JobStatus.failed: set(),
    JobStatus.cancelled: set(),
}


def transition(job: AnalysisJob, status: JobStatus, *, stage: str | None = None) -> None:
    current = job.status or JobStatus.uploaded
    if status not in ALLOWED[current]:
        raise ValueError(f"Invalid job transition: {current.value} -> {status.value}")
    now = datetime.now(UTC)
    job.status = status
    job.current_stage = stage or status.value
    if status == JobStatus.running:
        job.started_at = now
    if status in {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}:
        job.completed_at = now
    if status == JobStatus.completed:
        job.progress = 100
