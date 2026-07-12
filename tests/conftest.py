from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEST_ROOT = Path(tempfile.mkdtemp(prefix="tennis-tests-"))
os.environ.update(
    DATABASE_URL=f"sqlite:///{(TEST_ROOT / 'test.db').as_posix()}",
    DATA_ROOT=str(TEST_ROOT / "data"),
    MODEL_ROOT=str(TEST_ROOT / "models"),
    ALLOWED_HOSTS='["testserver"]',
    REDIS_URL="redis://localhost:6399/15",
)

from app.db.session import Base, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def clean_database():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr("app.api.routes.enqueue_analysis", lambda public_id, settings: f"rq-{public_id}")
    return TestClient(app)


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=320x240:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(TEST_ROOT, ignore_errors=True)
