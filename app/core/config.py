from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    secret_key: str = "development-only-change-me"
    database_url: str = "sqlite:///./data/tennis.db"
    redis_url: str = "redis://localhost:6379/0"
    data_root: Path = Path("data")
    model_root: Path = Path("models")
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024
    max_video_duration_seconds: int = 3600
    max_video_width: int = 3840
    max_video_height: int = 2160
    allowed_video_extensions: list[str] = Field(
        default_factory=lambda: [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"]
    )
    allowed_video_codecs: list[str] = Field(
        default_factory=lambda: ["h264", "hevc", "mpeg4", "vp8", "vp9", "av1", "mjpeg"]
    )
    job_timeout_seconds: int = 7200
    worker_concurrency: int = 1
    retention_days: int = 30
    log_level: str = "INFO"
    device: str = "cpu"
    allowed_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "testserver"])
    public_base_url: str = "http://localhost:8000"
    upload_chunk_bytes: int = 1024 * 1024
    analysis_chunk_frames: int = 256

    @field_validator("allowed_hosts", "allowed_video_extensions", "allowed_video_codecs", mode="before")
    @classmethod
    def split_hosts(cls, value):
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("max_upload_bytes", "job_timeout_seconds", "worker_concurrency")
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
