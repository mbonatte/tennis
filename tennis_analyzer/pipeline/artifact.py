"""Versioned, JSON-safe analysis artifacts used by repeatable rendering."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def write_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    document = {"schema_version": SCHEMA_VERSION, **payload}
    temporary.write_text(json.dumps(document, default=_encode, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def read_artifact(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported analysis artifact version")
    return document


def _encode(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} into an analysis artifact")
