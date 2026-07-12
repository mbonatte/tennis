import pytest

from app.models import AnalysisJob, JobStatus
from app.services.job_state import transition


def make_job():
    return AnalysisJob(
        original_filename="x.mp4", stored_filename="source.mp4", input_relative_path="jobs/x/input/source.mp4"
    )


def test_job_state_transitions_and_timestamps():
    job = make_job()
    transition(job, JobStatus.queued)
    transition(job, JobStatus.running)
    transition(job, JobStatus.completed)
    assert job.progress == 100 and job.started_at and job.completed_at


def test_invalid_job_transition_fails():
    job = make_job()
    with pytest.raises(ValueError):
        transition(job, JobStatus.completed)
