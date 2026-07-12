from __future__ import annotations

import re
import shutil
from pathlib import Path

from fastapi import UploadFile

from tennis_analyzer.errors import InvalidVideoError

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str | None) -> str:
    name = Path(filename or "video").name
    safe = SAFE_NAME.sub("_", name).strip("._")
    return safe[:200] or "video"


def safe_job_dir(data_root: Path, public_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", public_id):
        raise ValueError("Invalid job identifier")
    root = (data_root / "jobs").resolve()
    target = (root / public_id).resolve()
    if target.parent != root:
        raise ValueError("Unsafe job path")
    return target


def resolve_job_file(data_root: Path, relative_path: str) -> Path:
    root = data_root.resolve()
    target = (root / relative_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Unsafe stored path")
    return target


async def stream_upload(upload: UploadFile, target: Path, *, limit: int, chunk_size: int) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    total = 0
    try:
        with temporary.open("xb") as output:
            while chunk := await upload.read(chunk_size):
                total += len(chunk)
                if total > limit:
                    raise InvalidVideoError("Upload exceeds the configured size limit")
                output.write(chunk)
        temporary.replace(target)
        target.chmod(0o600)
        return total
    except BaseException:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def delete_job_files(data_root: Path, public_id: str) -> None:
    directory = safe_job_dir(data_root, public_id)
    if directory.exists():
        shutil.rmtree(directory)
