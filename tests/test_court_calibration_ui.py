from pathlib import Path


def test_court_calibration_clicks_select_nearest_marker_and_markers_are_translucent():
    template = Path("app/templates/job.html").read_text(encoding="utf-8")

    assert "existing.reduce((closest, candidate)" in template
    assert "context.globalAlpha = 0.55" in template
    assert "points.findIndex((candidate) => !candidate)" in template
