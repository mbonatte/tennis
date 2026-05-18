from __future__ import annotations

from dataclasses import replace
from typing import Sequence


def stabilize_player_roles(player_tracks: Sequence[list], frame_shape) -> list[list]:
    """Stabilize top/bottom player roles across temporary bad detections."""
    if not player_tracks:
        return []

    frame_h, frame_w = frame_shape[:2]
    top_limit_y = frame_h * 0.50
    side_margin_x = frame_w * 0.08
    max_top_center_jump = frame_w * 0.16

    last_valid_top = None
    last_valid_bottom = None
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
            side_margin_x=side_margin_x,
            frame_w=frame_w,
        ):
            top_player = None

        if (
            top_player is not None
            and last_valid_top is not None
            and _center_distance(top_player.center, last_valid_top.center) > max_top_center_jump
        ):
            top_player = None

        if bottom_player is not None and not _is_valid_bottom_player(
            bottom_player,
            top_limit_y=top_limit_y,
        ):
            bottom_player = None

        if top_player is None and last_valid_top is not None:
            top_player = replace(last_valid_top)
        elif top_player is not None:
            last_valid_top = top_player

        if bottom_player is None and last_valid_bottom is not None:
            bottom_player = replace(last_valid_bottom)
        elif bottom_player is not None:
            last_valid_bottom = bottom_player

        frame_players = []
        if top_player is not None:
            frame_players.append(_with_role(top_player, "top_player"))
        if bottom_player is not None and not _same_track(top_player, bottom_player):
            frame_players.append(_with_role(bottom_player, "bottom_player"))

        stabilized_tracks.append(frame_players)

    return stabilized_tracks


def _is_valid_top_player(player, top_limit_y, side_margin_x, frame_w):
    x1, y1, x2, y2 = player.bbox
    foot_y = y2
    center_x, _ = player.center
    if foot_y > top_limit_y:
        return False
    if center_x < side_margin_x or center_x > frame_w - side_margin_x:
        return False
    if y2 - y1 < top_limit_y * 0.12:
        return False
    return True


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
