from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Extract common safe checkpoint wrappers and normalize DataParallel keys."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a state dictionary")
    candidate = checkpoint
    for wrapper in ("state_dict", "model_state_dict"):
        wrapped = candidate.get(wrapper)
        if isinstance(wrapped, Mapping):
            candidate = wrapped
            break
    if not candidate or not all(isinstance(key, str) for key in candidate):
        raise TypeError("checkpoint state dictionary has invalid keys")
    state_dict = dict(candidate)
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict
