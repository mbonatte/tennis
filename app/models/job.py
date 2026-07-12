from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, BigInteger, DateTime, Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(str, enum.Enum):
    uploaded = "uploaded"
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.uploaded, index=True)
    current_stage: Mapped[str] = mapped_column(String(80), default="uploaded")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    submitted_options: Mapped[dict] = mapped_column(JSON, default=dict)
    input_relative_path: Mapped[str] = mapped_column(String(500))
    output_video_relative_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    result_relative_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    internal_diagnostic: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    input_size: Mapped[int] = mapped_column(BigInteger, default=0)
    output_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    video_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    video_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    video_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    video_codec: Mapped[str | None] = mapped_column(String(40), nullable=True)
    pipeline_version: Mapped[str] = mapped_column(String(40), default="0.1.0")
    queue_job_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    cancellation_requested: Mapped[bool] = mapped_column(default=False)
