from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

ProgressCallback = Callable[[str, int, str], None]


@dataclass(frozen=True)
class WorkStage:
    name: str
    weight: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("progress stage name cannot be empty")
        if self.weight <= 0:
            raise ValueError("progress stage weight must be positive")


class WeightedProgress:
    """Map enabled multi-pass work to monotonic, throttled integer progress."""

    def __init__(self, stages: Iterable[WorkStage], callback: ProgressCallback | None):
        self._stages = list(stages)
        if not self._stages:
            raise ValueError("at least one progress stage is required")
        names = [stage.name for stage in self._stages]
        if len(names) != len(set(names)):
            raise ValueError("progress stage names must be unique")
        self._callback = callback
        self._total_weight = sum(stage.weight for stage in self._stages)
        self._last_percent = 0
        self._last_stage: str | None = None

    def update(self, stage_name: str, completed: int, total: int, message: str) -> int:
        if total <= 0:
            raise ValueError("progress total must be positive")
        if completed < 0:
            raise ValueError("progress completed work cannot be negative")
        stage_index = next((i for i, stage in enumerate(self._stages) if stage.name == stage_name), None)
        if stage_index is None:
            raise ValueError(f"unknown progress stage: {stage_name}")
        stage = self._stages[stage_index]
        prior = sum(item.weight for item in self._stages[:stage_index])
        fraction = min(1.0, completed / total)
        percent = min(99, 1 + int(98 * (prior + stage.weight * fraction) / self._total_weight))
        percent = max(self._last_percent, percent)
        if self._callback and (percent > self._last_percent or stage_name != self._last_stage):
            self._callback(stage_name, percent, message)
        self._last_percent = percent
        self._last_stage = stage_name
        return percent

    def complete(self, message: str = "Analysis completed") -> None:
        if self._callback:
            self._callback("completed", 100, message)
        self._last_percent = 100
        self._last_stage = "completed"
