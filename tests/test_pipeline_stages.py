from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tennis_analyzer.pipeline import analyze_video
from tennis_analyzer.pipeline.stages import StageFactories
from tennis_analyzer.schemas import AnalysisOptions, PipelineOptions


@dataclass
class Lifecycle:
    created: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    calls: dict[str, int] = field(default_factory=dict)
    active: int = 0
    peak_active: int = 0


class FakeStage:
    def __init__(self, name, lifecycle, result_factory, *, fail_on_call=None):
        self.name = name
        self.lifecycle = lifecycle
        self.result_factory = result_factory
        self.fail_on_call = fail_on_call
        lifecycle.created.append(name)
        lifecycle.active += 1
        lifecycle.peak_active = max(lifecycle.peak_active, lifecycle.active)

    def process_chunk(self, frames):
        call = self.lifecycle.calls.get(self.name, 0) + 1
        self.lifecycle.calls[self.name] = call
        if call == self.fail_on_call:
            raise RuntimeError(f"{self.name} inference failed")
        return self.result_factory(len(frames))

    def draw(self, frame, tracks):
        return frame

    def close(self):
        self.lifecycle.closed.append(self.name)
        self.lifecycle.active -= 1


def _model_root(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    root.mkdir()
    for name in ("tracknet_model.pt", "tennis_court.pt", "yolo26n.pt"):
        (root / name).touch()
    return root


def _factories(lifecycle, *, failing_ball_call=None):
    return StageFactories(
        ball=lambda path, device, batch: FakeStage(
            "ball", lifecycle, lambda count: [(None, None)] * count, fail_on_call=failing_ball_call
        ),
        court=lambda path, device: FakeStage("court", lifecycle, lambda count: ([None] * count, [None] * count)),
        player=lambda box, pose, device, pose_enabled: FakeStage(
            "player", lifecycle, lambda count: [[] for _ in range(count)]
        ),
        bounce=lambda path: pytest.fail("bounce detector should be disabled"),
    )


def test_low_memory_pipeline_loads_each_enabled_stage_once_and_releases_between_stages(sample_video, tmp_path):
    lifecycle = Lifecycle()
    options = PipelineOptions(
        analysis=AnalysisOptions(ball_tracking=True, court_detection=True, player_tracking=True),
        chunk_size=4,
        ball_batch_size=2,
    )

    result = analyze_video(
        sample_video,
        tmp_path / "output",
        options,
        model_root=_model_root(tmp_path),
        stage_factories=_factories(lifecycle),
    )

    assert lifecycle.created == ["ball", "court", "player"]
    assert lifecycle.closed == lifecycle.created
    assert lifecycle.calls == {"ball": 3, "court": 3, "player": 3}
    assert lifecycle.peak_active == 1
    assert result.summary["frames_processed"] == 10


def test_disabled_model_stages_are_never_constructed(sample_video, tmp_path):
    def unexpected(*args, **kwargs):
        pytest.fail("disabled model stage was constructed")

    factories = StageFactories(ball=unexpected, court=unexpected, player=unexpected, bounce=unexpected)

    result = analyze_video(
        sample_video,
        tmp_path / "output",
        PipelineOptions(chunk_size=4),
        model_root=tmp_path / "models",
        stage_factories=factories,
    )

    assert result.summary["frames_processed"] == 10


def test_model_stage_is_released_when_inference_fails(sample_video, tmp_path):
    lifecycle = Lifecycle()
    options = PipelineOptions(
        analysis=AnalysisOptions(ball_tracking=True),
        chunk_size=4,
        ball_batch_size=2,
    )
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="ball inference failed"):
        analyze_video(
            sample_video,
            output,
            options,
            model_root=_model_root(tmp_path),
            stage_factories=_factories(lifecycle, failing_ball_call=2),
        )

    assert lifecycle.closed == ["ball"]
    assert lifecycle.active == 0
    assert not (output / ".annotated.mp4").exists()
    assert not (output / "analyzed.mp4").exists()


def test_video_writer_is_released_and_partial_output_removed_on_render_failure(monkeypatch, sample_video, tmp_path):
    output = tmp_path / "output"

    class FailingWriter:
        released = False

        def isOpened(self):
            return True

        def write(self, frame):
            (output / ".annotated.mp4").write_bytes(b"partial")
            raise RuntimeError("render failed")

        def release(self):
            self.released = True

    writer = FailingWriter()
    monkeypatch.setattr("tennis_analyzer.pipeline.service.cv2.VideoWriter", lambda *args, **kwargs: writer)

    with pytest.raises(RuntimeError, match="render failed"):
        analyze_video(sample_video, output, PipelineOptions(chunk_size=4))

    assert writer.released
    assert not (output / ".annotated.mp4").exists()
