import pytest

from tennis_analyzer.pipeline.progress import WeightedProgress, WorkStage


def test_progress_is_monotonic_and_reaches_100_only_when_completed():
    updates = []
    progress = WeightedProgress(
        [WorkStage("ball", 4), WorkStage("rendering", 1)],
        lambda stage, percent, message: updates.append((stage, percent)),
    )

    progress.update("ball", 0, 10, "starting")
    progress.update("ball", 5, 10, "half")
    progress.update("ball", 4, 10, "stale update")
    progress.update("ball", 10, 10, "done")
    progress.update("rendering", 10, 10, "rendered")

    assert [percent for _, percent in updates] == sorted(percent for _, percent in updates)
    assert updates[-1][1] == 99
    progress.complete()
    assert updates[-1] == ("completed", 100)


def test_disabled_stages_do_not_reserve_progress_ranges():
    with_court = WeightedProgress([WorkStage("ball", 1), WorkStage("court", 1), WorkStage("rendering", 1)], None)
    without_court = WeightedProgress([WorkStage("ball", 1), WorkStage("rendering", 1)], None)

    assert without_court.update("ball", 1, 1, "done") > with_court.update("ball", 1, 1, "done")


def test_progress_updates_are_throttled_to_integer_changes():
    updates = []
    progress = WeightedProgress(
        [WorkStage("detecting", 1)], lambda stage, percent, message: updates.append((stage, percent))
    )

    for completed in range(1001):
        progress.update("detecting", completed, 1000, "working")

    assert len(updates) <= 100
    assert updates[-1][1] == 99


@pytest.mark.parametrize(
    "stages",
    [[], [WorkStage("one", 1), WorkStage("one", 2)]],
)
def test_invalid_progress_plans_are_rejected(stages):
    with pytest.raises(ValueError):
        WeightedProgress(stages, None)


def test_invalid_stage_weight_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        WorkStage("bad", 0)
