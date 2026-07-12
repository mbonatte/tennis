from __future__ import annotations

from app.core.config import Settings


def enqueue_analysis(public_id: str, settings: Settings) -> str:
    from redis import Redis
    from rq import Queue

    connection = Redis.from_url(settings.redis_url)
    queued = Queue("analysis", connection=connection, default_timeout=settings.job_timeout_seconds).enqueue(
        "app.workers.tasks.run_analysis_job",
        public_id,
        job_timeout=settings.job_timeout_seconds,
        result_ttl=86400,
        failure_ttl=604800,
    )
    return queued.id
