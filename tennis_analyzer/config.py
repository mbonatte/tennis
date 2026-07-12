from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPaths:
    root: Path
    ball: Path
    bounce: Path
    court: Path
    court_keypoints: Path
    player: Path
    pose: Path

    @classmethod
    def from_root(cls, root: Path) -> ModelPaths:
        return cls(
            root=root,
            ball=root / "tracknet_model.pt",
            bounce=root / "bounce_model.cbm",
            court=root / "tennis_court.pt",
            court_keypoints=root / "keypoints_model.pth",
            player=root / "yolo26n.pt",
            pose=root / "yolo26n-pose.pt",
        )
