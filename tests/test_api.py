from pathlib import Path

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import AnalysisJob, JobStatus, RenderOutput
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
    assert [stage["key"] for stage in body["workflow"]] == [
        "scanning",
        "ball_tracking",
        "court_detection",
        "pose_tracking",
        "events",
    ]
    assert body["workflow"][0]["status"] == "pending"
    page = client.get(f"/jobs/{body['id']}")
    assert page.status_code == 200 and "Workflow" in page.text and "Stage progress" in page.text
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


def test_completed_analysis_can_queue_independent_render(client, sample_video: Path, monkeypatch):
    public_id = upload(client, sample_video).json()["id"]
    monkeypatch.setattr("app.api.routes.enqueue_render", lambda public_id, settings: f"render-{public_id}")
    with SessionLocal() as db:
        job = db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
        transition(job, JobStatus.running)
        transition(job, JobStatus.completed)
        job.analysis_artifact_relative_path = f"jobs/{public_id}/output/analysis-artifact.json"
        db.commit()

    response = client.post(
        f"/jobs/{public_id}/renders",
        data={
            "ball_trail": "true",
            "player_boxes": "true",
            "top_player_label": "Ana",
            "bottom_player_label": "Mauricio",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    with SessionLocal() as db:
        render = db.scalar(select(RenderOutput))
        assert render.status == JobStatus.queued
        assert render.visualization_options["ball_trail"] is True
        assert render.visualization_options["player_boxes"] is True
        assert render.visualization_options["top_player_label"] == "Ana"
        assert render.visualization_options["bottom_player_label"] == "Mauricio"


def test_analysis_with_active_render_cannot_be_deleted(client, sample_video: Path):
    public_id = upload(client, sample_video).json()["id"]
    with SessionLocal() as db:
        job = db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id))
        transition(job, JobStatus.running)
        transition(job, JobStatus.completed)
        job.renders.append(
            RenderOutput(
                status=JobStatus.running,
                visualization_options={"frame_number": True},
            )
        )
        db.commit()

    response = client.delete(f"/api/jobs/{public_id}")

    assert response.status_code == 409
    with SessionLocal() as db:
        assert db.scalar(select(AnalysisJob).where(AnalysisJob.public_id == public_id)) is not None
