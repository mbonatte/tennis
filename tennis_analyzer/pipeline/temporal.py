from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class TemporalContextBuffer(Generic[T]):
    """Retain the minimum source-frame tail required by a temporal model."""

    context_size: int
    _tail: list[T] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.context_size < 0:
            raise ValueError("temporal context size cannot be negative")

    def prepend(self, items: list[T]) -> tuple[list[T], int]:
        """Return prior context plus new items and the number to trim from output."""
        overlap = len(self._tail)
        combined = [*self._tail, *items]
        if self.context_size:
            self._tail = combined[-self.context_size :]
        else:
            self._tail.clear()
        return combined, overlap

    def clear(self) -> None:
        self._tail.clear()
