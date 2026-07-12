from pathlib import Path

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import AnalysisJob, JobStatus
from app.services.job_state import transition


def upload(client, path: Path):
    with path.open("rb") as video:
        return client.post(
            "/api/jobs",
            files={"video": ("match.mp4", video, "video/mp4")},
            data={"scene_cut_detection": "true", "frame_number": "true"},
        )


def test_job_creation_and_status(client, sample_video: Path):
    response = upload(client, sample_video)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued" and body["progress"] == 0
    status = client.get(body["links"]["status"])
    assert status.status_code == 200 and status.json()["filename"] == "match.mp4"


def test_invalid_upload_and_size_enforcement(client, tmp_path: Path, monkeypatch):
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not-video")
    with fake.open("rb") as file:
        response = client.post("/api/jobs", files={"video": ("fake.mp4", file, "video/mp4")})
    assert response.status_code == 422
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "max_upload_bytes", 1)
    valid = tmp_path / "valid.mp4"
    valid.write_bytes(b"12")
    with valid.open("rb") as file:
        response = client.post("/api/jobs", files={"video": ("valid.mp4", file, "video/mp4")})
    assert response.status_code == 422


def test_completed_result_failed_behavior_and_deletion(client, sample_video: Path):
    created = upload(client, sample_video).json()
    public_id = created["id"]
    with SessionLocal() as db:
        job = db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
        transition(job, JobStatus.running)
        transition(job, JobStatus.completed)
        output = Path(job.input_relative_path).parents[1] / "output"
        from app.core.config import get_settings

        root = get_settings().data_root
        directory = root / output
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "result.json").write_text('{"schema_version":"1.0","summary":{},"shots":[],"bounces":[]}')
        (directory / "analyzed.mp4").write_bytes(b"0123456789")
        job.result_relative_path = str(output / "result.json")
        job.output_video_relative_path = str(output / "analyzed.mp4")
        db.commit()
    assert client.get(f"/api/jobs/{public_id}/results").json()["schema_version"] == "1.0"
    ranged = client.get(f"/jobs/{public_id}/files/video", headers={"Range": "bytes=2-5"})
    assert ranged.status_code == 206 and ranged.content == b"2345"
    assert client.delete(f"/api/jobs/{public_id}").status_code == 204
    assert client.get(f"/api/jobs/{public_id}").status_code == 404


def test_failed_job_exposes_safe_message_only(client, sample_video: Path):
    public_id = upload(client, sample_video).json()["id"]
    with SessionLocal() as db:
        job = db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
        transition(job, JobStatus.failed)
        job.error_type = "model_error"
        job.error_message = "A required model is unavailable"
        job.internal_diagnostic = "secret traceback"
        db.commit()
    body = client.get(f"/api/jobs/{public_id}").json()
    assert body["error"]["message"] == "A required model is unavailable"
    assert "traceback" not in str(body)
