"""Deterministic, override-friendly tennis scoring for singles matches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

PlayerRole = Literal["top_player", "bottom_player"]


@dataclass(frozen=True)
class PointRecord:
    start_frame: int
    end_frame: int
    server: PlayerRole | None = None
    winner: PlayerRole | None = None
    confidence: float | None = None
    reason: str | None = None
    estimated: bool = True
    user_overrides: dict = field(default_factory=dict)
    game_boundary: bool = False


def score_match(points: list[PointRecord], initial: dict | None = None) -> dict:
    """Score known point winners; unknown points leave state unchanged and explicit."""
    state = {
        "points": {"top_player": 0, "bottom_player": 0},
        "games": {"top_player": 0, "bottom_player": 0},
        "sets": {"top_player": 0, "bottom_player": 0},
        "completed_sets": [],
    }
    if initial:
        for key in ("points", "games", "sets"):
            state[key].update(initial.get(key, {}))
    rows = []
    for index, point in enumerate(points, start=1):
        before = _display_points(state["points"])
        if point.winner is not None:
            state["points"][point.winner] += 1
            game_winner = _game_winner(state["points"])
            if game_winner:
                state["games"][game_winner] += 1
                state["points"] = {"top_player": 0, "bottom_player": 0}
                set_winner = _set_winner(state["games"])
                if set_winner:
                    state["completed_sets"].append(dict(state["games"]))
                    state["sets"][set_winner] += 1
                    state["games"] = {"top_player": 0, "bottom_player": 0}
        rows.append(
            {
                "point_index": index,
                **asdict(point),
                "score_before": before,
                "score_after": _display_points(state["points"]),
                "games": dict(state["games"]),
                "sets": dict(state["sets"]),
            }
        )
    return {
        "points": rows,
        "games": state["games"],
        "sets": state["sets"],
        "completed_sets": state["completed_sets"],
        "limitations": [
            "Singles scoring. Automatic winner inference is experimental; tiebreak inference requires user review."
        ],
    }


def _game_winner(points: dict[str, int]) -> str | None:
    top, bottom = points["top_player"], points["bottom_player"]
    if max(top, bottom) >= 4 and abs(top - bottom) >= 2:
        return "top_player" if top > bottom else "bottom_player"
    return None


def _set_winner(games: dict[str, int]) -> str | None:
    top, bottom = games["top_player"], games["bottom_player"]
    if max(top, bottom) >= 6 and abs(top - bottom) >= 2:
        return "top_player" if top > bottom else "bottom_player"
    return None


def _display_points(points: dict[str, int]) -> dict[str, str]:
    top, bottom = points["top_player"], points["bottom_player"]
    if top >= 3 and bottom >= 3:
        if top == bottom:
            return {"top_player": "40", "bottom_player": "40", "status": "deuce"}
        leader = "top_player" if top > bottom else "bottom_player"
        return {
            "top_player": "AD" if leader == "top_player" else "40",
            "bottom_player": "AD" if leader == "bottom_player" else "40",
            "status": "advantage",
        }
    values = ["0", "15", "30", "40"]
    return {"top_player": values[top], "bottom_player": values[bottom], "status": "normal"}
