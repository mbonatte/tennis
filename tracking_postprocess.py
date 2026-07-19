from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace


def stabilize_player_roles(player_tracks: Sequence[list], frame_shape, *, max_missing_frames: int = 8) -> list[list]:
    """Apply a second, whole-track continuity gate to player roles.

    Assignment-time tracking protects pose and statistics.  This pass also
    catches a bad box at a chunk boundary and makes artifacts deterministic
    when a tracker temporarily changes its internal ID.
    """
    if not player_tracks:
        return []

    frame_h, frame_w = frame_shape[:2]
    top_limit_y = frame_h * 0.50
    top_min_y = _top_player_min_foot_y(frame_h)
    side_margin_x = frame_w * 0.20
    last_valid = {"top_player": None, "bottom_player": None}
    missing_frames = {"top_player": 0, "bottom_player": 0}
    stabilized_tracks = []

    for players in player_tracks:
        by_role = {player.role: player for player in players}
        top_player = by_role.get("top_player")
        bottom_player = by_role.get("bottom_player")

        if _same_track(top_player, bottom_player):
            top_player = None

        if top_player is not None and not _is_valid_top_player(
            top_player,
            top_limit_y=top_limit_y,
            top_min_y=top_min_y,
            side_margin_x=side_margin_x,
            frame_w=frame_w,
        ):
            top_player = None

        if bottom_player is not None and not _is_valid_bottom_player(
            bottom_player,
            top_limit_y=top_limit_y,
        ):
            bottom_player = None

        top_player = _accept_or_hold("top_player", top_player, last_valid, missing_frames, frame_w, max_missing_frames)
        bottom_player = _accept_or_hold(
            "bottom_player", bottom_player, last_valid, missing_frames, frame_w, max_missing_frames
        )

        frame_players = []
        if top_player is not None:
            frame_players.append(_with_role(top_player, "top_player"))
        if bottom_player is not None and not _same_track(top_player, bottom_player):
            frame_players.append(_with_role(bottom_player, "bottom_player"))

        stabilized_tracks.append(frame_players)

    return stabilized_tracks


def _accept_or_hold(role, candidate, last_valid, missing_frames, frame_w, max_missing_frames):
    previous = last_valid[role]
    if candidate is not None and previous is not None:
        previous_width = max(1, previous.bbox[2] - previous.bbox[0])
        max_center_jump = max(frame_w * 0.18, previous_width * 5.0)
        if _center_distance(candidate.center, previous.center) > max_center_jump:
            candidate = None

    if candidate is not None:
        last_valid[role] = candidate
        missing_frames[role] = 0
        return candidate

    missing_frames[role] += 1
    if previous is not None and missing_frames[role] <= max_missing_frames:
        return replace(previous)
    last_valid[role] = None
    return None


def _is_valid_top_player(player, top_limit_y, top_min_y, side_margin_x, frame_w):
    x1, y1, x2, y2 = player.bbox
    foot_y = y2
    center_x, _ = player.center
    if foot_y < top_min_y:
        return False
    if foot_y > top_limit_y:
        return False
    if center_x < side_margin_x or center_x > frame_w - side_margin_x:
        return False
    return not y2 - y1 < top_limit_y * 0.12


def _top_player_min_foot_y(frame_h):
    high_res_adjustment = max(0.0, min((frame_h - 720) / 360, 1.0)) * 0.04
    return frame_h * (0.18 + high_res_adjustment)


def _is_valid_bottom_player(player, top_limit_y):
    _, _, _, y2 = player.bbox
    return y2 > top_limit_y


def _same_track(player_a, player_b):
    if player_a is None or player_b is None:
        return False
    if player_a.track_id is None or player_b.track_id is None:
        return player_a.bbox == player_b.bbox
    return player_a.track_id == player_b.track_id


def _center_distance(center_a, center_b):
    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]
    return (dx * dx + dy * dy) ** 0.5


def _with_role(player, role: str):
    if player.role == role:
        return player
    return replace(player, role=role)
