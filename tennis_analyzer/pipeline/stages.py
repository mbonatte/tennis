from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


def _loaded(stage_name: str, started: float, stage):
    logger.info(
        "Loaded job-scoped model stage",
        extra={"model_stage": stage_name, "model_load_seconds": round(time.perf_counter() - started, 3)},
    )
    return stage


class ChunkStage(Protocol):
    def process_chunk(self, frames: list[np.ndarray]) -> Any: ...

    def close(self) -> None: ...


class PlayerStage:
    """Adapt a persistent Ultralytics tracker to the common chunk-stage API."""

    def __init__(self, tracker):
        self.tracker = tracker

    def process_chunk(self, frames: list[np.ndarray]):
        tracks = []
        for frame in frames:
            players, _raw_result = self.tracker.track_frame(frame)
            tracks.append(players)
        return tracks

    def draw(self, frame, tracks):
        return self.tracker.draw(frame, tracks, copy_frame=False)

    def close(self) -> None:
        self.tracker.close()


def create_ball_stage(model_path: Path, device: str, batch_size: int):
    from ball import BallTracker

    started = time.perf_counter()
    return _loaded("ball", started, BallTracker.from_checkpoint(model_path, device, batch_size=batch_size))


def create_court_stage(model_path: Path, device: str):
    from court import CourtTracker

    started = time.perf_counter()
    return _loaded("court", started, CourtTracker.from_checkpoint(model_path, device))


def create_player_stage(box_path: Path, pose_path: Path, device: str, pose_enabled: bool):
    from player import BoxPlayerTracker, HybridPlayerTracker

    started = time.perf_counter()
    if pose_enabled:
        tracker = HybridPlayerTracker(
            box_model_path=str(box_path),
            pose_model_path=str(pose_path),
            conf=0.5,
            pose_conf=0.35,
            device=device,
        )
    else:
        tracker = BoxPlayerTracker(model_path=str(box_path), conf=0.5, device=device)
    return _loaded("player_pose" if pose_enabled else "player", started, PlayerStage(tracker))


def create_bounce_detector(model_path: Path):
    from bounce_detector import BounceDetector

    started = time.perf_counter()
    return _loaded("bounce", started, BounceDetector(str(model_path)))


@dataclass(frozen=True)
class StageFactories:
    """Injectable constructors make job-scoped model ownership testable."""

    ball: Callable[[Path, str, int], ChunkStage] = create_ball_stage
    court: Callable[[Path, str], ChunkStage] = create_court_stage
    player: Callable[[Path, Path, str, bool], ChunkStage] = create_player_stage
    bounce: Callable[[Path], Any] = create_bounce_detector
