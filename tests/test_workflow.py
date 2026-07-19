from datetime import UTC, datetime, timedelta

from app.services.workflow import create_workflow, finalize_workflow, record_progress, workflow_rows
from tennis_analyzer.schemas import AnalysisOptions, VisualizationOptions


def test_workflow_matches_enabled_pipeline_stages_and_groups_render_options():
    analysis = AnalysisOptions(ball_tracking=True, court_detection=True, statistics=True, scene_cut_detection=True)
    visual = VisualizationOptions(ball_trail=True, court_overlay=True, frame_number=True)

    workflow = create_workflow(analysis, visual)

    assert [stage["key"] for stage in workflow["stages"]] == [
        "scanning",
        "ball_tracking",
        "court_detection",
        "events",
        "rendering",
        "normalizing",
    ]
    assert workflow["stages"][1]["options"] == ["ball_tracking"]
    assert workflow["stages"][4]["options"] == ["ball_trail", "frame_number", "court_overlay"]


def test_workflow_records_local_progress_and_preserves_stage_elapsed_time():
    started = datetime(2026, 7, 19, 12, tzinfo=UTC)
    workflow = create_workflow(AnalysisOptions(ball_tracking=True), VisualizationOptions())

    workflow = record_progress(workflow, "scanning", 5, started)
    workflow = record_progress(workflow, "ball_tracking", 40, started + timedelta(seconds=8))
    rows = workflow_rows(workflow, started + timedelta(seconds=15))

    assert rows[0]["status"] == "completed" and rows[0]["progress"] == 100
    assert rows[0]["duration_seconds"] == 8
    assert rows[1]["status"] == "running"
    assert 0 < rows[1]["progress"] < 100
    assert rows[1]["duration_seconds"] == 7


def test_workflow_freezes_the_active_stage_when_a_job_fails():
    started = datetime(2026, 7, 19, 12, tzinfo=UTC)
    workflow = record_progress(create_workflow(AnalysisOptions()), "rendering", 55, started)

    finalized = finalize_workflow(workflow, "failed", started + timedelta(seconds=12))
    rows = workflow_rows(finalized, started + timedelta(days=1))

    rendering = next(row for row in rows if row["key"] == "rendering")
    assert rendering["status"] == "failed"
    assert rendering["duration_seconds"] == 12
