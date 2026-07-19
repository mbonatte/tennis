from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RESULT_SCHEMA_VERSION = "1.0"
PIPELINE_VERSION = "0.1.0"


class OptionValidationError(ValueError):
    """Raised when an impossible analysis configuration is requested."""


@dataclass(frozen=True)
class AnalysisOptions:
    ball_tracking: bool = False
    court_detection: bool = False
    player_tracking: bool = False
    pose_tracking: bool = False
    bounce_detection: bool = False
    scene_cut_detection: bool = True
    statistics: bool = False
    point_analysis: bool = False

    def validated(self) -> AnalysisOptions:
        values = asdict(self)
        if self.pose_tracking:
            values["player_tracking"] = True
        if self.bounce_detection:
            values["ball_tracking"] = True
        if self.statistics:
            values.update(ball_tracking=True, court_detection=True)
        if self.point_analysis:
            values["scene_cut_detection"] = True
        return AnalysisOptions(**values)


@dataclass(frozen=True)
class VisualizationOptions:
    ball_trail: bool = False
    bounce_markers: bool = False
    frame_number: bool = True
    court_overlay: bool = False
    court_keypoints: bool = False
    player_boxes: bool = False
    player_poses: bool = False
    statistics_overlay: bool = False
    ball_history_plot: bool = False

    def validate_for(self, analysis: AnalysisOptions) -> None:
        requirements = {
            "ball_trail": analysis.ball_tracking,
            "bounce_markers": analysis.bounce_detection,
            "court_overlay": analysis.court_detection,
            "court_keypoints": analysis.court_detection,
            "player_boxes": analysis.player_tracking,
            "player_poses": analysis.pose_tracking,
            "statistics_overlay": analysis.statistics,
            "ball_history_plot": analysis.ball_tracking,
        }
        invalid = [name for name, available in requirements.items() if getattr(self, name) and not available]
        if invalid:
            raise OptionValidationError("Visualization requires disabled analysis stage(s): " + ", ".join(invalid))


@dataclass(frozen=True)
class PipelineOptions:
    analysis: AnalysisOptions = field(default_factory=AnalysisOptions)
    visualization: VisualizationOptions = field(default_factory=VisualizationOptions)
    chunk_size: int = 128
    ball_batch_size: int = 4
    device: str = "cpu"
    execution_mode: str = "low_memory"

    def validated(self) -> PipelineOptions:
        if not 1 <= self.chunk_size <= 2048:
            raise OptionValidationError("chunk_size must be between 1 and 2048")
        if not 1 <= self.ball_batch_size <= self.chunk_size:
            raise OptionValidationError("ball_batch_size must be between 1 and chunk_size")
        analysis = self.analysis.validated()
        self.visualization.validate_for(analysis)
        if self.device != "cpu" and self.device != "cuda" and not self.device.startswith("cuda:"):
            raise OptionValidationError("device must be cpu, cuda, or cuda:<index>")
        if self.execution_mode != "low_memory":
            raise OptionValidationError("execution_mode must be low_memory")
        return PipelineOptions(
            analysis,
            self.visualization,
            self.chunk_size,
            self.ball_batch_size,
            self.device,
            self.execution_mode,
        )


@dataclass(frozen=True)
class VideoMetadata:
    duration_seconds: float
    width: int
    height: int
    fps: float
    frame_count: int | None
    container: str
    video_codec: str
    audio_codec: str | None
    file_size: int


@dataclass(frozen=True)
class AnalysisResult:
    input_filename: str
    output_video: str
    result_json: str
    metadata: VideoMetadata
    analysis_options: dict[str, bool]
    visualization_options: dict[str, bool]
    shots: list[dict[str, Any]] = field(default_factory=list)
    bounces: list[dict[str, Any]] = field(default_factory=list)
    player_statistics: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    scene_cuts: list[int] = field(default_factory=list)
    points: list[dict[str, Any]] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = RESULT_SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
    estimates: Literal["experimental"] = "experimental"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
