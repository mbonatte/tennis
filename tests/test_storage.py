from pathlib import Path

import pytest

from app.services.storage import resolve_job_file, safe_job_dir, sanitize_filename


def test_filename_sanitization_removes_path_and_unsafe_characters():
    assert sanitize_filename("../../my match<script>.mp4") == "my_match_script_.mp4"


def test_safe_job_path_is_confined(tmp_path: Path):
    identifier = "123e4567-e89b-12d3-a456-426614174000"
    assert safe_job_dir(tmp_path, identifier).parent == (tmp_path / "jobs").resolve()
    with pytest.raises(ValueError):
        safe_job_dir(tmp_path, "../../etc")
    with pytest.raises(ValueError):
        resolve_job_file(tmp_path, "../secret")
