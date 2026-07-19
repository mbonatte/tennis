"""Durable, user-facing execution workflow for analysis jobs."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from tennis_analyzer.pipeline.service import progress_stages
from tennis_analyzer.schemas import AnalysisOptions, VisualizationOptions

WORKFLOW_VERSION = 1

_STAGE_LABELS = {
    "scanning": "Inspect video",
    "ball_tracking": "Track ball",
    "court_detection": "Detect court",
    "player_tracking": "Track players",
    "pose_tracking": "Estimate player poses",
    "events": "Detect events and statistics",
    "rendering": "Render annotations",
    "point_analysis": "Create point clips",
    "normalizing": "Finalize video",
}

_STAGE_OPTIONS = {
    "scanning": ["scene_cut_detection"],
    "ball_tracking": ["ball_tracking"],
    "court_detection": ["court_detection"],
    "player_tracking": ["player_tracking"],
    "pose_tracking": ["player_tracking", "pose_tracking"],
    "events": ["bounce_detection", "statistics"],
    "point_analysis": ["point_analysis"],
}

_RENDER_OPTIONS = [
    "ball_trail",
    "bounce_markers",
    "frame_number",
    "court_overlay",
    "court_keypoints",
    "player_boxes",
    "player_poses",
    "statistics_overlay",
    "ball_history_plot",
]


def create_workflow(analysis: AnalysisOptions, visualization: VisualizationOptions | None = None) -> dict[str, Any]:
    """Create the exact stage plan used by the pipeline for one job."""
    return {
        "version": WORKFLOW_VERSION,
        "stages": [
            {
                "key": stage.name,
                "label": _STAGE_LABELS.get(stage.name, _label(stage.name)),
                "weight": stage.weight,
                "options": _selected_options(stage.name, analysis, visualization),
                "status": "pending",
                "progress": 0,
                "started_at": None,
                "completed_at": None,
            }
            for stage in progress_stages(analysis)
        ],
    }


def record_progress(
    workflow: dict[str, Any] | None, stage_name: str, global_progress: int, now: datetime
) -> dict[str, Any]:
    """Record a throttled pipeline update without relying on transient worker state."""
    current = deepcopy(workflow) if workflow else {"version": WORKFLOW_VERSION, "stages": []}
    stages: list[dict[str, Any]] = current["stages"]
    index = next((i for i, stage in enumerate(stages) if stage["key"] == stage_name), None)
    if index is None:
        stages.append(
            {
                "key": stage_name,
                "label": _STAGE_LABELS.get(stage_name, _label(stage_name)),
                "weight": 0,
                "options": [],
                "status": "running",
                "progress": min(99, max(0, global_progress)),
                "started_at": _timestamp(now),
                "completed_at": None,
            }
        )
        return current

    for earlier in stages[:index]:
        if earlier["status"] in {"pending", "running"}:
            earlier.update(status="completed", progress=100, completed_at=_timestamp(now))
    stage = stages[index]
    if stage["started_at"] is None:
        stage["started_at"] = _timestamp(now)
    stage["status"] = "running"
    stage["progress"] = max(stage["progress"], _local_progress(stages, index, global_progress))
    return current


def finalize_workflow(workflow: dict[str, Any] | None, status: str, now: datetime) -> dict[str, Any]:
    """Freeze completed, failed, or cancelled stage timing for the job detail view."""
    current = deepcopy(workflow) if workflow else {"version": WORKFLOW_VERSION, "stages": []}
    terminal_stage_status = "completed" if status == "completed" else status
    for stage in current["stages"]:
        if stage["status"] == "running":
            stage["status"] = terminal_stage_status
            if status == "completed":
                stage["progress"] = 100
            stage["completed_at"] = _timestamp(now)
        elif status == "completed" and stage["status"] == "pending":
            stage.update(status="completed", progress=100, started_at=_timestamp(now), completed_at=_timestamp(now))
    return current


def workflow_rows(workflow: dict[str, Any] | None, now: datetime) -> list[dict[str, Any]]:
    """Return presentation-safe rows including elapsed time for active stages."""
    rows = deepcopy((workflow or {}).get("stages", []))
    for row in rows:
        started = _parse_timestamp(row.get("started_at"))
        finished = _parse_timestamp(row.get("completed_at"))
        row["duration_seconds"] = max(0, int(((finished or now) - started).total_seconds())) if started else None
        row["option_labels"] = [_label(option) for option in row.get("options", [])]
    return rows


def format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "—"
    minutes, remainder = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {remainder}s"
    return f"{remainder}s"


def _local_progress(stages: list[dict[str, Any]], index: int, global_progress: int) -> int:
    total_weight = sum(float(stage.get("weight", 0)) for stage in stages)
    weight = float(stages[index].get("weight", 0))
    if weight <= 0 or total_weight <= 0:
        return min(99, max(0, global_progress))
    prior_weight = sum(float(stage.get("weight", 0)) for stage in stages[:index])
    fraction = ((max(1, global_progress) - 1) / 98 * total_weight - prior_weight) / weight
    return min(100, max(0, int(round(fraction * 100))))


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_timestamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _label(value: str) -> str:
    return value.replace("_", " ").capitalize()


def _selected_options(
    stage_name: str, analysis: AnalysisOptions, visualization: VisualizationOptions | None
) -> list[str]:
    if stage_name == "rendering" and visualization is not None:
        return [name for name in _RENDER_OPTIONS if getattr(visualization, name)]
    return [name for name in _STAGE_OPTIONS.get(stage_name, []) if getattr(analysis, name)]
