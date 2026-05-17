from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from court_reference import CourtReference


@dataclass(frozen=True)
class ShotEvent:
    frame: int
    player_role: str | None
    ball_speed_kmh: float | None
    reason: str


@dataclass(frozen=True)
class BounceEvent:
    frame: int
    phase: str
    in_out: str
    court_region: str | None
    projected_x: float | None
    projected_y: float | None
    rule: str


@dataclass(frozen=True)
class PlayerStats:
    shots: int = 0
    average_speed_kmh: float | None = None
    max_speed_kmh: float | None = None


@dataclass(frozen=True)
class MatchStats:
    shot_events: list[ShotEvent]
    bounce_events: list[BounceEvent]
    player_stats: dict[str, PlayerStats]
    average_ball_speed_kmh: float | None
    max_ball_speed_kmh: float | None
    scene_cuts: list[int]

    def to_dict(self) -> dict:
        return {
            "shot_events": [asdict(event) for event in self.shot_events],
            "bounce_events": [asdict(event) for event in self.bounce_events],
            "player_stats": {
                role: asdict(stats) for role, stats in self.player_stats.items()
            },
            "average_ball_speed_kmh": self.average_ball_speed_kmh,
            "max_ball_speed_kmh": self.max_ball_speed_kmh,
            "scene_cuts": self.scene_cuts,
        }


def detect_scene_cuts(frames: Sequence, threshold: float = 0.55) -> list[int]:
    """Detect likely hard cuts from frame-to-frame HSV histogram distance."""
    if len(frames) < 2:
        return []

    scene_cuts = []
    previous_hist = _frame_histogram(frames[0])

    for frame_num, frame in enumerate(frames[1:], start=1):
        hist = _frame_histogram(frame)
        distance = cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
        if distance >= threshold:
            scene_cuts.append(frame_num)
        previous_hist = hist

    return scene_cuts


def project_ball_track(ball_track, homography_matrices) -> list[tuple[float, float] | None]:
    projected = []

    for frame_num, ball_point in enumerate(ball_track):
        if frame_num >= len(homography_matrices):
            projected.append(None)
            continue
        projected.append(project_point(ball_point, homography_matrices[frame_num]))

    return projected


def project_player_tracks(player_tracks, homography_matrices) -> list[dict[str, tuple[float, float]]]:
    projected_frames = []

    for frame_num, players in enumerate(player_tracks or []):
        frame_players = {}
        if frame_num < len(homography_matrices) and homography_matrices[frame_num] is not None:
            for player in players:
                frame_players[player.role] = project_point(
                    _player_foot_point(player),
                    homography_matrices[frame_num],
                )
        projected_frames.append(frame_players)

    return projected_frames


def project_point(point, homography_matrix) -> tuple[float, float] | None:
    if homography_matrix is None or point is None or point[0] is None or point[1] is None:
        return None

    point_array = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    projected = cv2.perspectiveTransform(point_array, homography_matrix)
    return float(projected[0, 0, 0]), float(projected[0, 0, 1])


def compute_match_stats(
    ball_track,
    bounces: set[int],
    fps: int,
    homography_matrices=None,
    player_tracks=None,
    scene_cuts: list[int] | None = None,
) -> MatchStats:
    """Compute approximate tennis stats from projected court coordinates."""
    if homography_matrices is not None:
        ball_positions = project_ball_track(ball_track, homography_matrices)
        player_positions = project_player_tracks(player_tracks, homography_matrices)
        speed_scale = _court_meter_scales()
    else:
        ball_positions = [_point_or_none(point) for point in ball_track]
        player_positions = []
        speed_scale = None

    shot_detection_positions = [_point_or_none(point) for point in ball_track]
    ball_speeds = _segment_speeds(ball_positions, fps, speed_scale, max_speed_kmh=280)
    shot_events = detect_shot_events(
        shot_detection_positions,
        bounces=bounces,
        fps=fps,
        ball_speeds=ball_speeds,
        projected_ball_positions=ball_positions,
        player_positions=player_positions,
    )
    player_stats = _compute_player_stats(player_positions, fps, speed_scale, shot_events)
    bounce_events = classify_bounces(bounces, ball_positions, shot_events)

    return MatchStats(
        shot_events=shot_events,
        bounce_events=bounce_events,
        player_stats=player_stats,
        average_ball_speed_kmh=_mean(ball_speeds),
        max_ball_speed_kmh=_max(ball_speeds),
        scene_cuts=scene_cuts or [],
    )


def classify_bounces(
    bounces: set[int],
    projected_ball_positions,
    shot_events: list[ShotEvent],
) -> list[BounceEvent]:
    """Classify bounce locations with serve-box rules before rally rules."""
    court = CourtReference()
    shot_frames = [event.frame for event in shot_events]
    events = []

    for bounce_frame in sorted(bounces):
        point = (
            projected_ball_positions[bounce_frame]
            if bounce_frame < len(projected_ball_positions)
            else None
        )
        previous_shot_index = _previous_shot_index(bounce_frame, shot_frames)

        if previous_shot_index is None:
            events.append(
                BounceEvent(
                    frame=bounce_frame,
                    phase="pre_shot",
                    in_out="unknown",
                    court_region=None,
                    projected_x=None if point is None else point[0],
                    projected_y=None if point is None else point[1],
                    rule="no preceding shot, so serve/game boundary is unknown",
                )
            )
            continue

        if previous_shot_index == 0:
            server_role = shot_events[0].player_role
            in_out, region = _classify_serve_bounce(point, court, server_role)
            events.append(
                BounceEvent(
                    frame=bounce_frame,
                    phase="serve",
                    in_out=in_out,
                    court_region=region,
                    projected_x=None if point is None else point[0],
                    projected_y=None if point is None else point[1],
                    rule="serve uses the opposite service box boundaries",
                )
            )
            continue

        in_out, region = _classify_game_bounce(point, court)
        events.append(
            BounceEvent(
                frame=bounce_frame,
                phase="game",
                in_out=in_out,
                court_region=region,
                projected_x=None if point is None else point[0],
                projected_y=None if point is None else point[1],
                rule="game uses full singles court boundaries",
            )
        )

    return events


def detect_shot_events(
    ball_positions,
    bounces: set[int],
    fps: int,
    ball_speeds: list[float | None],
    projected_ball_positions=None,
    player_positions=None,
    min_gap_frames: int | None = None,
) -> list[ShotEvent]:
    """Detect shots using the sustained vertical trajectory change used by the reference repo."""
    min_gap = min_gap_frames or max(6, int(fps * 0.25))
    shot_frames = detect_sustained_trajectory_changes(ball_positions)
    if not shot_frames:
        return []

    candidates = []
    projected_ball_positions = projected_ball_positions or ball_positions

    for frame_num in shot_frames:
        player_role, player_distance = _nearest_player(
            frame_num,
            projected_ball_positions,
            player_positions,
        )
        ball_speed = _shot_segment_speed(ball_speeds, frame_num, shot_frames)

        candidates.append((frame_num, player_role, ball_speed))

    shot_candidates = _suppress_nearby_shots(candidates, min_gap)
    return [
        ShotEvent(
            frame=frame_num,
            player_role=player_role,
            ball_speed_kmh=ball_speed,
            reason="sustained_ball_trajectory_change",
        )
        for frame_num, player_role, ball_speed in shot_candidates
    ]


def detect_sustained_trajectory_changes(
    ball_positions,
    rolling_window: int = 5,
    minimum_change_frames_for_hit: int = 12,
) -> list[int]:
    """Port of the reference repo's ball-hit frame heuristic for center points."""
    y_values = _interpolate_axis(ball_positions, axis=1)
    if y_values is None:
        return []

    rolling_y = _rolling_mean(y_values, rolling_window)
    delta_y = np.diff(rolling_y, prepend=np.nan)
    lookahead = int(minimum_change_frames_for_hit * 1.2)
    shot_frames = []

    for frame_num in range(1, len(delta_y) - lookahead):
        changing_from_down_to_up = delta_y[frame_num] > 0 and delta_y[frame_num + 1] < 0
        changing_from_up_to_down = delta_y[frame_num] < 0 and delta_y[frame_num + 1] > 0
        if not (changing_from_down_to_up or changing_from_up_to_down):
            continue

        change_count = 0
        for change_frame in range(frame_num + 1, frame_num + lookahead + 1):
            if changing_from_down_to_up and delta_y[change_frame] < 0:
                change_count += 1
            elif changing_from_up_to_down and delta_y[change_frame] > 0:
                change_count += 1

        if change_count > minimum_change_frames_for_hit - 1:
            shot_frames.append(frame_num)

    return shot_frames


def _previous_shot_index(frame_num: int, shot_frames: list[int]) -> int | None:
    previous_indices = [
        index for index, shot_frame in enumerate(shot_frames) if shot_frame < frame_num
    ]
    if not previous_indices:
        return None
    return previous_indices[-1]


def _classify_serve_bounce(point, court: CourtReference, server_role: str | None):
    if point is None:
        return "unknown", None

    x, y = point
    left_x = court.left_inner_line[0][0]
    center_x = court.middle_line[0][0]
    right_x = court.right_inner_line[0][0]

    if server_role == "top_player":
        service_y_min = court.net[0][1]
        service_y_max = court.bottom_inner_line[0][1]
        court_half = "bottom"
    else:
        service_y_min = court.top_inner_line[0][1]
        service_y_max = court.net[0][1]
        court_half = "top"

    if not _inside_rect(x, y, left_x, right_x, service_y_min, service_y_max):
        return "out", None

    side = "left" if x <= center_x else "right"
    return "in", f"{court_half}_{side}_service_box"


def _classify_game_bounce(point, court: CourtReference):
    if point is None:
        return "unknown", None

    x, y = point
    left_x = court.left_inner_line[0][0]
    right_x = court.right_inner_line[0][0]
    top_y = court.baseline_top[0][1]
    bottom_y = court.baseline_bottom[0][1]

    if _inside_rect(x, y, left_x, right_x, top_y, bottom_y):
        return "in", "singles_court"
    return "out", None


def _inside_rect(x, y, left_x, right_x, top_y, bottom_y):
    return left_x <= x <= right_x and top_y <= y <= bottom_y


def draw_stats_overlay(frames: Sequence, stats: MatchStats) -> list:
    output_frames = []
    stats_by_frame = {event.frame: event for event in stats.shot_events}

    for frame_num, frame in enumerate(frames):
        annotated = frame.copy()
        _draw_stats_panel(annotated, stats)

        if frame_num in stats_by_frame:
            event = stats_by_frame[frame_num]
            label = "shot"
            if event.player_role:
                label += f": {event.player_role}"
            if event.ball_speed_kmh is not None:
                label += f" {event.ball_speed_kmh:.1f} km/h"

            cv2.putText(
                annotated,
                label,
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        output_frames.append(annotated)

    return output_frames


def save_stats(stats: MatchStats, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")


def _frame_histogram(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).astype("float32")


def _player_foot_point(player):
    x1, _, x2, y2 = player.bbox
    return ((x1 + x2) / 2, y2)


def _court_meter_scales() -> tuple[float, float]:
    court = CourtReference()
    meters_per_x = 10.97 / court.court_width
    meters_per_y = 23.77 / court.court_height
    return meters_per_x, meters_per_y


def _point_or_none(point):
    if point is None or point[0] is None or point[1] is None:
        return None
    return float(point[0]), float(point[1])


def _segment_speeds(
    positions,
    fps: int,
    scales: tuple[float, float] | None,
    max_speed_kmh: float | None = None,
):
    speeds = [None]

    for previous, current in zip(positions, positions[1:]):
        if previous is None or current is None:
            speeds.append(None)
            continue

        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        if scales is not None:
            dx *= scales[0]
            dy *= scales[1]

        distance = float(np.sqrt(dx * dx + dy * dy))
        speed = distance * fps * 3.6
        if max_speed_kmh is not None and speed > max_speed_kmh:
            speeds.append(None)
        else:
            speeds.append(speed)

    return speeds


def _interpolate_axis(positions, axis: int):
    values = np.array(
        [np.nan if point is None else point[axis] for point in positions],
        dtype=float,
    )
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) < 4:
        return None

    indices = np.arange(len(values))
    return np.interp(indices, valid, values[valid])


def _rolling_mean(values, window: int):
    result = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result[index] = float(np.mean(values[start:index + 1]))
    return result


def _near_any(frame_num: int, frames: set[int], radius: int) -> bool:
    return any(abs(frame_num - other) <= radius for other in frames)


def _suppress_nearby_frames(frames: list[int], min_gap: int) -> list[int]:
    kept = []
    for frame_num in frames:
        if not kept or frame_num - kept[-1] >= min_gap:
            kept.append(frame_num)
    return kept


def _suppress_nearby_shots(candidates, min_gap: int):
    kept = []
    for candidate in candidates:
        frame_num = candidate[0]
        if not kept or frame_num - kept[-1][0] >= min_gap:
            kept.append(candidate)
    return kept


def _window_mean(values, center: int, radius: int):
    window = [
        value
        for value in values[max(0, center - radius):center + radius + 1]
        if value is not None and np.isfinite(value)
    ]
    return _mean(window)


def _shot_segment_speed(ball_speeds, frame_num: int, shot_frames: list[int]):
    try:
        frame_index = shot_frames.index(frame_num)
    except ValueError:
        return _window_mean(ball_speeds, frame_num, radius=5)

    if frame_index + 1 < len(shot_frames):
        end_frame = shot_frames[frame_index + 1]
    else:
        end_frame = min(len(ball_speeds), frame_num + 15)

    segment = [
        speed
        for speed in ball_speeds[frame_num:end_frame]
        if speed is not None and np.isfinite(speed)
    ]
    return _mean(segment)


def _mean(values):
    clean = [value for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return None
    return float(np.mean(clean))


def _max(values):
    clean = [value for value in values if value is not None and np.isfinite(value)]
    if not clean:
        return None
    return float(np.max(clean))


def _nearest_player_role(frame_num, ball_positions, player_positions):
    role, _ = _nearest_player(frame_num, ball_positions, player_positions)
    return role


def _nearest_player(frame_num, ball_positions, player_positions):
    if not player_positions or frame_num >= len(player_positions):
        return None, np.inf

    ball_position = ball_positions[frame_num]
    if ball_position is None:
        return None, np.inf

    best_role = None
    best_distance = np.inf
    for role, player_position in player_positions[frame_num].items():
        if player_position is None:
            continue

        distance = np.linalg.norm(np.array(ball_position) - np.array(player_position))
        if distance < best_distance:
            best_distance = distance
            best_role = role

    return best_role, best_distance


def _compute_player_stats(player_positions, fps, scales, shot_events):
    roles = ["top_player", "bottom_player"]
    shot_counts = {role: 0 for role in roles}
    for event in shot_events:
        if event.player_role in shot_counts:
            shot_counts[event.player_role] += 1

    stats = {}
    for role in roles:
        role_positions = [frame.get(role) for frame in player_positions]
        speeds = _segment_speeds(role_positions, fps, scales, max_speed_kmh=45)
        stats[role] = PlayerStats(
            shots=shot_counts[role],
            average_speed_kmh=_mean(speeds),
            max_speed_kmh=_max(speeds),
        )

    return stats


def _draw_stats_panel(frame, stats: MatchStats) -> None:
    height, width = frame.shape[:2]
    panel_width = min(430, width - 20)
    panel_height = 180
    x1 = 10
    y1 = height - panel_height - 10
    x2 = x1 + panel_width
    y2 = y1 + panel_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    lines = [
        f"Shots: {len(stats.shot_events)}",
        f"Ball avg/max: {_format_speed(stats.average_ball_speed_kmh)} / {_format_speed(stats.max_ball_speed_kmh)}",
        _format_bounce_summary(stats.bounce_events),
    ]

    for role, player_stats in stats.player_stats.items():
        label = role.replace("_", " ")
        lines.append(
            f"{label}: shots {player_stats.shots}, "
            f"avg/max {_format_speed(player_stats.average_speed_kmh)} / "
            f"{_format_speed(player_stats.max_speed_kmh)}"
        )

    if stats.scene_cuts:
        lines.append(f"Scene cuts: {len(stats.scene_cuts)}")

    for line_num, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x1 + 12, y1 + 28 + line_num * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )


def _format_speed(value):
    if value is None:
        return "n/a"
    return f"{value:.1f} km/h"


def _format_bounce_summary(bounce_events: list[BounceEvent]):
    in_count = sum(1 for event in bounce_events if event.in_out == "in")
    out_count = sum(1 for event in bounce_events if event.in_out == "out")
    unknown_count = sum(1 for event in bounce_events if event.in_out == "unknown")
    return f"Bounces: in {in_count}, out {out_count}, unknown {unknown_count}"
