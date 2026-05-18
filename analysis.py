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
    total_distance_m: float | None = None
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

    point_start_frame = _analysis_start_frame(scene_cuts, len(ball_track))
    bounces = {frame for frame in bounces if frame > point_start_frame}
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
    shot_events = _recover_initial_contact_without_bounces(
        shot_events,
        shot_detection_positions,
        ball_positions,
        player_positions,
        fps,
    )
    bounces = augment_bounce_candidates_from_trajectory(
        bounces,
        shot_detection_positions,
        shot_events,
        fps,
        projected_ball_positions=ball_positions,
    )
    shot_events = refine_shot_events_from_bounces(
        shot_events,
        bounces,
        shot_detection_positions,
        ball_positions,
        player_positions,
        fps,
        projected_bounce_positions=ball_positions,
    )
    shot_events = [event for event in shot_events if event.frame > point_start_frame]
    shot_events = enforce_rally_player_alternation(shot_events)
    bounces = refine_bounce_frames(
        bounces,
        shot_detection_positions,
        shot_events,
        fps,
        projected_ball_positions=ball_positions,
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


def _analysis_start_frame(scene_cuts: list[int] | None, num_frames: int) -> int:
    valid_cuts = [frame for frame in (scene_cuts or []) if 0 < frame < num_frames]
    return max(valid_cuts, default=-1)


def enforce_rally_player_alternation(shot_events: list[ShotEvent]) -> list[ShotEvent]:
    """Correct ambiguous player ownership by alternating consecutive rally hits."""
    corrected = []
    last_role = None

    for event in sorted(shot_events, key=lambda shot: shot.frame):
        role = event.player_role
        if role is None and last_role in {"top_player", "bottom_player"}:
            role = _opposite_player(last_role)
        elif role in {"top_player", "bottom_player"} and role == last_role:
            role = _opposite_player(role)

        corrected_event = ShotEvent(
            frame=event.frame,
            player_role=role,
            ball_speed_kmh=event.ball_speed_kmh,
            reason=event.reason if role == event.player_role else f"{event.reason}_alternation_corrected",
        )
        corrected.append(corrected_event)

        if corrected_event.player_role in {"top_player", "bottom_player"}:
            last_role = corrected_event.player_role

    return corrected


def _opposite_player(role: str):
    return "bottom_player" if role == "top_player" else "top_player"


def refine_bounce_frames(
    bounces: set[int],
    ball_positions,
    shot_events: list[ShotEvent],
    fps: int,
    projected_ball_positions=None,
) -> set[int]:
    """Remove pre-shot toss detections and recover bounces hidden by short occlusions."""
    shot_frames = [event.frame for event in shot_events]
    if not shot_frames:
        return set()

    first_shot_frame = shot_frames[0]
    min_shot_to_bounce_gap = max(6, int(fps * 0.22))
    bounces = _filter_projected_bounce_outliers(bounces, projected_ball_positions)
    refined = {
        frame
        for frame in bounces
        if frame - first_shot_frame >= min_shot_to_bounce_gap
    }
    recovered = recover_occluded_bounces(ball_positions, shot_frames, fps)

    for frame in recovered:
        if frame <= first_shot_frame:
            continue
        if _near_any(frame, refined, radius=max(4, int(fps * 0.16))):
            continue
        refined.add(frame)

    refined = _drop_bounces_too_close_to_shots(refined, shot_events, fps)
    return _suppress_duplicate_pre_shot_bounces(refined, shot_events, fps)


def augment_bounce_candidates_from_trajectory(
    bounces: set[int],
    ball_positions,
    shot_events: list[ShotEvent],
    fps: int,
    projected_ball_positions=None,
) -> set[int]:
    """Recover clear trajectory turns that the learned bounce model missed."""
    refined = set(bounces)
    shot_frames = sorted(event.frame for event in shot_events)
    if not shot_frames:
        return refined

    projected_ball_positions = projected_ball_positions or ball_positions
    first_shot = shot_frames[0]
    first_bounce_candidate = _best_bounce_turn(
        ball_positions,
        first_shot + max(5, int(fps * 0.18)),
        min(first_shot + max(24, int(fps * 0.95)), len(ball_positions) - 2),
        fps,
    )
    if (
        first_bounce_candidate is not None
        and not _near_any(first_bounce_candidate, refined, radius=max(4, int(fps * 0.16)))
        and not _projected_bounce_is_far_out(first_bounce_candidate, projected_ball_positions)
    ):
        refined.add(first_bounce_candidate)

    for index, shot_frame in enumerate(shot_frames):
        next_shot = shot_frames[index + 1] if index + 1 < len(shot_frames) else len(ball_positions)
        start = shot_frame + max(5, int(fps * 0.18))
        end = min(next_shot - max(3, int(fps * 0.12)), shot_frame + max(35, int(fps * 1.4)), len(ball_positions) - 2)
        if start >= end:
            continue
        if any(start <= bounce <= end for bounce in refined):
            continue

        candidate = _best_bounce_turn(ball_positions, start, end, fps)
        if candidate is None:
            continue
        if _projected_bounce_is_far_out(candidate, projected_ball_positions):
            continue
        refined.add(candidate)

    return refined


def _best_bounce_turn(ball_positions, start: int, end: int, fps: int) -> int | None:
    candidates = []
    min_turn = max(18, int(fps * 0.7))
    for frame_num in range(start, end + 1):
        prev_v = _local_y_velocity(ball_positions, frame_num, direction=-1)
        next_v = _local_y_velocity(ball_positions, frame_num, direction=1)
        if prev_v is None or next_v is None:
            continue

        direction_flip = (prev_v > 0 >= next_v) or (prev_v <= 0 < next_v)
        turn = abs(next_v - prev_v)
        if not direction_flip or turn < min_turn:
            continue

        point = _point_or_none(ball_positions[frame_num])
        if point is None:
            continue
        support = _local_track_support(ball_positions, frame_num, radius=3, max_distance=90)
        if support < 2:
            continue
        candidates.append((support * 300 + turn, frame_num))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def _local_track_support(ball_positions, frame_num: int, radius: int, max_distance: float) -> int:
    point = _point_or_none(ball_positions[frame_num])
    if point is None:
        return 0

    support = 0
    for index in range(max(0, frame_num - radius), min(len(ball_positions), frame_num + radius + 1)):
        other = _point_or_none(ball_positions[index])
        if other is None:
            continue
        if np.hypot(point[0] - other[0], point[1] - other[1]) <= max_distance:
            support += 1
    return support


def _projected_bounce_is_far_out(frame_num: int, projected_ball_positions) -> bool:
    if projected_ball_positions is None or frame_num >= len(projected_ball_positions):
        return False
    point = _point_or_none(projected_ball_positions[frame_num])
    if point is None:
        return False

    court = CourtReference()
    left_x = court.left_court_line[0][0] - 260
    right_x = court.right_court_line[0][0] + 260
    top_y = court.baseline_top[0][1] - 450
    bottom_y = court.baseline_bottom[0][1] + 450
    x, y = point
    return not (left_x <= x <= right_x and top_y <= y <= bottom_y)


def _suppress_duplicate_pre_shot_bounces(
    bounces: set[int],
    shot_events: list[ShotEvent],
    fps: int,
) -> set[int]:
    refined = set(bounces)
    min_gap = max(5, int(fps * 0.18))
    max_gap = max(18, int(fps * 0.75))
    preferred_gap = max(9, int(fps * 0.40))

    for shot in shot_events[1:]:
        candidates = [
            frame
            for frame in refined
            if min_gap <= shot.frame - frame <= max_gap
        ]
        close_duplicates = [
            frame
            for frame in refined
            if 0 < shot.frame - frame < min_gap
        ]

        if not candidates:
            continue

        best = min(candidates, key=lambda frame: abs((shot.frame - frame) - preferred_gap))
        for frame in candidates:
            if frame != best:
                refined.discard(frame)
        for frame in close_duplicates:
            refined.discard(frame)

    return refined


def _drop_bounces_too_close_to_shots(
    bounces: set[int],
    shot_events: list[ShotEvent],
    fps: int,
) -> set[int]:
    shot_frames = [event.frame for event in shot_events]
    radius = max(3, int(fps * 0.12))
    return {
        bounce
        for bounce in bounces
        if not any(0 <= abs(bounce - shot_frame) <= radius for shot_frame in shot_frames)
    }


def refine_shot_events_from_bounces(
    shot_events: list[ShotEvent],
    bounces: set[int],
    ball_positions,
    projected_ball_positions,
    player_positions,
    fps: int,
    projected_bounce_positions=None,
) -> list[ShotEvent]:
    """Use bounce context to remove bounce-as-shot labels and recover post-bounce hits."""
    min_gap = max(6, int(fps * 0.22))
    original_bounces = set(bounces)
    bounces = _filter_projected_bounce_outliers(bounces, projected_bounce_positions)
    rejected_bounces = original_bounces - bounces
    filtered = [
        event
        for event in shot_events
        if not _near_any(event.frame, bounces, radius=max(2, int(fps * 0.08)))
        and _has_enough_track_context(ball_positions, event.frame, radius=4)
    ]
    filtered = _drop_pre_serve_toss_events(filtered, bounces, ball_positions, fps)
    filtered = _drop_in_flight_pre_bounce_shots(filtered, bounces, fps)
    filtered = _ensure_initial_serve_shot(filtered, bounces, ball_positions, projected_ball_positions, player_positions, fps)
    filtered = _recover_initial_contact_before_first_bounce(
        filtered,
        bounces,
        ball_positions,
        projected_ball_positions,
        player_positions,
        fps,
    )
    filtered.extend(
        _recover_shots_from_rejected_bounces(
            rejected_bounces,
            ball_positions,
            projected_ball_positions,
            player_positions,
            fps,
        )
    )

    for bounce_frame in sorted(bounces):
        if _has_shot_after_bounce(filtered, bounce_frame, fps):
            continue

        hit_frame = _estimate_post_bounce_hit_frame(ball_positions, bounce_frame, fps)
        if hit_frame is None:
            continue
        if not _has_contact_turn(ball_positions, hit_frame, min_turn=max(18, int(fps * 0.7))):
            continue
        if hit_frame + max(8, int(fps * 0.3)) >= len(ball_positions):
            continue
        if _near_any(hit_frame, rejected_bounces, radius=max(2, int(fps * 0.08))):
            continue
        if _near_any(hit_frame, {event.frame for event in filtered}, radius=min_gap):
            continue

        player_role, _ = _nearest_player(
            hit_frame,
            projected_ball_positions,
            player_positions,
        )
        filtered.append(
            ShotEvent(
                frame=hit_frame,
                player_role=player_role,
                ball_speed_kmh=None,
                reason="post_bounce_hit_window",
            )
        )

    filtered = _align_shots_to_bounce_contact_windows(filtered, bounces, ball_positions, fps)
    filtered = _drop_implausible_post_bounce_shots(filtered, bounces, ball_positions, fps)
    return _merge_shot_events(filtered, min_gap=min_gap)


def _drop_pre_serve_toss_events(
    shot_events: list[ShotEvent],
    bounces: set[int],
    ball_positions,
    fps: int,
) -> list[ShotEvent]:
    first_bounce = _first_plausible_bounce(bounces, ball_positions, fps)
    if first_bounce is None:
        return shot_events

    serve_frame = _estimate_pre_bounce_serve_hit(ball_positions, first_bounce, fps)
    if serve_frame is None:
        return shot_events

    keep_from = serve_frame - max(2, int(fps * 0.08))
    return [
        event
        for event in shot_events
        if event.frame >= keep_from or event.frame > first_bounce
    ]


def _drop_in_flight_pre_bounce_shots(
    shot_events: list[ShotEvent],
    bounces: set[int],
    fps: int,
) -> list[ShotEvent]:
    if not bounces:
        return shot_events

    max_pre_bounce_gap = max(8, int(fps * 0.55))
    max_post_bounce_gap = max(16, int(fps * 0.8))
    kept = []
    first_event_frame = min(event.frame for event in shot_events)

    for event in sorted(shot_events, key=lambda shot: shot.frame):
        previous_bounces = [frame for frame in bounces if frame < event.frame]
        next_bounces = [frame for frame in bounces if frame > event.frame]
        previous_gap = event.frame - max(previous_bounces) if previous_bounces else None
        next_gap = min(next_bounces) - event.frame if next_bounces else None

        is_before_next_bounce = next_gap is not None and next_gap <= max_pre_bounce_gap
        is_not_after_recent_bounce = previous_gap is None or previous_gap > max_post_bounce_gap
        if event.frame == first_event_frame and previous_gap is None:
            kept.append(event)
            continue
        if is_before_next_bounce and is_not_after_recent_bounce:
            continue

        kept.append(event)

    return kept


def _recover_initial_contact_before_first_bounce(
    shot_events: list[ShotEvent],
    bounces: set[int],
    ball_positions,
    projected_ball_positions,
    player_positions,
    fps: int,
) -> list[ShotEvent]:
    if not bounces or not shot_events:
        return shot_events

    first_bounce = min(bounces)
    first_shot = min(event.frame for event in shot_events)
    search_end = min(first_shot - max(6, int(fps * 0.22)), first_bounce - max(6, int(fps * 0.22)))
    if search_end <= max(6, int(fps * 0.22)):
        return shot_events

    contact_frame = _first_strong_contact_turn(ball_positions, 1, search_end, fps)
    if contact_frame is None:
        return shot_events
    if first_shot - contact_frame < max(6, int(fps * 0.22)):
        return shot_events

    player_role, _ = _nearest_player(
        contact_frame,
        projected_ball_positions,
        player_positions,
    )
    return [
        ShotEvent(
            frame=contact_frame,
            player_role=player_role,
            ball_speed_kmh=None,
            reason="early_contact_recovery",
        ),
        *shot_events,
    ]


def _recover_initial_contact_without_bounces(
    shot_events: list[ShotEvent],
    ball_positions,
    projected_ball_positions,
    player_positions,
    fps: int,
) -> list[ShotEvent]:
    if not shot_events:
        return shot_events

    first_event = min(shot_events, key=lambda event: event.frame)
    contact_frame = _first_strong_contact_turn(
        ball_positions,
        1,
        first_event.frame,
        fps,
    )
    if contact_frame is None:
        return shot_events
    if abs(first_event.frame - contact_frame) <= max(3, int(fps * 0.12)):
        return [
            ShotEvent(
                frame=contact_frame,
                player_role=first_event.player_role,
                ball_speed_kmh=first_event.ball_speed_kmh,
                reason=first_event.reason,
            )
            if event is first_event
            else event
            for event in shot_events
        ]
    if first_event.frame - contact_frame < max(10, int(fps * 0.4)):
        return shot_events

    player_role, _ = _nearest_player(
        contact_frame,
        projected_ball_positions,
        player_positions,
    )
    return [
        ShotEvent(
            frame=contact_frame,
            player_role=player_role,
            ball_speed_kmh=None,
            reason="early_contact_recovery",
        ),
        *shot_events,
    ]


def _recover_shots_from_rejected_bounces(
    rejected_bounces: set[int],
    ball_positions,
    projected_ball_positions,
    player_positions,
    fps: int,
) -> list[ShotEvent]:
    recovered = []
    for bounce_frame in sorted(rejected_bounces):
        shot_frame = _estimate_shot_near_rejected_bounce(ball_positions, bounce_frame, fps)
        if shot_frame is None:
            continue

        player_role, _ = _nearest_player(
            shot_frame,
            projected_ball_positions,
            player_positions,
        )
        if player_role is None:
            continue

        recovered.append(
            ShotEvent(
                frame=shot_frame,
                player_role=player_role,
                ball_speed_kmh=None,
                reason="rejected_bounce_contact_recovery",
            )
        )

    return recovered


def _estimate_shot_near_rejected_bounce(ball_positions, bounce_frame: int, fps: int):
    search_radius = max(3, int(fps * 0.12))
    candidates = []

    for frame_num in range(bounce_frame, min(len(ball_positions), bounce_frame + search_radius + 1)):
        point = _point_or_none(ball_positions[frame_num])
        if point is None:
            continue

        next_v = _local_y_velocity(ball_positions, frame_num, direction=1)
        if next_v is None:
            continue

        # After contact the ball often moves sharply upward in image space.
        if next_v < -12:
            preferred_offset = 2
            offset_penalty = abs((frame_num - bounce_frame) - preferred_offset)
            candidates.append((-next_v - offset_penalty * 10, frame_num))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def _ensure_initial_serve_shot(
    shot_events: list[ShotEvent],
    bounces: set[int],
    ball_positions,
    projected_ball_positions,
    player_positions,
    fps: int,
) -> list[ShotEvent]:
    if not bounces or not shot_events:
        return shot_events

    first_bounce = _first_plausible_bounce(bounces, ball_positions, fps)
    if first_bounce is None:
        return shot_events

    first_shot = min(event.frame for event in shot_events)
    if first_shot < first_bounce:
        return shot_events

    serve_frame = _estimate_pre_bounce_serve_hit(ball_positions, first_bounce, fps)
    if serve_frame is None:
        return shot_events

    player_role = _infer_server_role_from_first_bounce(first_bounce, projected_ball_positions)
    if player_role is None:
        player_role, _ = _nearest_player(serve_frame, projected_ball_positions, player_positions)
    return [
        ShotEvent(
            frame=serve_frame,
            player_role=player_role,
            ball_speed_kmh=None,
            reason="pre_bounce_serve_hit",
        ),
        *shot_events,
    ]


def _infer_server_role_from_first_bounce(first_bounce: int, projected_ball_positions):
    if projected_ball_positions is None or first_bounce >= len(projected_ball_positions):
        return None

    point = _point_or_none(projected_ball_positions[first_bounce])
    if point is None:
        point = _interpolated_position_at(projected_ball_positions, first_bounce, max_gap=8)
    if point is None:
        return None

    court = CourtReference()
    return "bottom_player" if point[1] < court.net[0][1] else "top_player"


def _first_plausible_bounce(bounces: set[int], ball_positions, fps: int):
    for frame in sorted(bounces):
        if _has_enough_track_context(ball_positions, frame, radius=max(3, int(fps * 0.12))):
            return frame
    return None


def _estimate_pre_bounce_serve_hit(ball_positions, bounce_frame: int, fps: int):
    start = max(1, bounce_frame - max(16, int(fps * 0.6)))
    end = max(start, bounce_frame - max(6, int(fps * 0.2)))
    candidates = []

    for frame_num in range(start, end + 1):
        point = _point_or_none(ball_positions[frame_num])
        if point is None:
            continue

        prev_v = _local_y_velocity(ball_positions, frame_num, direction=-1)
        next_v = _local_y_velocity(ball_positions, frame_num, direction=1)
        if prev_v is None or next_v is None:
            continue

        # Prefer the first frame where the serve starts moving sharply downward.
        if next_v >= 12:
            return frame_num
        if next_v > 0:
            candidates.append((next_v, frame_num))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def _filter_projected_bounce_outliers(bounces: set[int], projected_ball_positions):
    if projected_ball_positions is None:
        return set(bounces)

    court = CourtReference()
    left_x = court.left_court_line[0][0] - 180
    right_x = court.right_court_line[0][0] + 180
    top_y = court.baseline_top[0][1] - 350
    bottom_y = court.baseline_bottom[0][1] + 350

    filtered = set()
    for frame in bounces:
        if frame >= len(projected_ball_positions):
            continue
        point = projected_ball_positions[frame]
        if point is None:
            filtered.add(frame)
            continue

        x, y = point
        if left_x <= x <= right_x and top_y <= y <= bottom_y:
            filtered.add(frame)

    return filtered


def _has_shot_after_bounce(shot_events: list[ShotEvent], bounce_frame: int, fps: int):
    min_after = max(2, int(fps * 0.08))
    max_after = max(24, int(fps * 0.8))
    return any(min_after <= event.frame - bounce_frame <= max_after for event in shot_events)


def _estimate_post_bounce_hit_frame(ball_positions, bounce_frame: int, fps: int):
    start = bounce_frame + max(5, int(fps * 0.18))
    end = min(len(ball_positions) - 1, bounce_frame + max(14, int(fps * 0.6)))
    if start >= end:
        return None

    candidates = []
    preferred_offset = max(7, int(fps * 0.3))
    for frame_num in range(start, end + 1):
        point = _point_or_none(ball_positions[frame_num])
        if point is None:
            continue

        prev_v = _local_y_velocity(ball_positions, frame_num, direction=-1)
        next_v = _local_y_velocity(ball_positions, frame_num, direction=1)
        if prev_v is None or next_v is None:
            continue

        offset_penalty = abs((frame_num - bounce_frame) - preferred_offset)
        turn_score = abs(next_v - prev_v) - offset_penalty * 1.2
        candidates.append((turn_score, frame_num))

    if not candidates:
        return None

    min_score = max(18, int(fps * 0.7))
    candidates = [(score, frame) for score, frame in candidates if score >= min_score]
    if not candidates:
        return None
    candidates.sort(key=lambda candidate: candidate[1])
    return candidates[0][1]


def _align_shots_to_bounce_contact_windows(
    shot_events: list[ShotEvent],
    bounces: set[int],
    ball_positions,
    fps: int,
) -> list[ShotEvent]:
    if not bounces or not shot_events:
        return shot_events

    aligned = list(shot_events)
    for bounce_frame in sorted(bounces):
        hit_frame = _estimate_post_bounce_hit_frame(ball_positions, bounce_frame, fps)
        if hit_frame is None:
            continue

        min_after = max(3, int(fps * 0.12))
        max_after = max(16, int(fps * 0.65))
        candidates = [
            event
            for event in aligned
            if min_after <= event.frame - bounce_frame <= max_after
        ]
        if not candidates:
            continue

        closest = min(candidates, key=lambda event: abs(event.frame - hit_frame))
        if abs(closest.frame - hit_frame) > max(5, int(fps * 0.22)):
            continue

        aligned.remove(closest)
        aligned.append(
            ShotEvent(
                frame=hit_frame,
                player_role=closest.player_role,
                ball_speed_kmh=closest.ball_speed_kmh,
                reason=closest.reason,
            )
        )

    return aligned


def _drop_implausible_post_bounce_shots(
    shot_events: list[ShotEvent],
    bounces: set[int],
    ball_positions,
    fps: int,
) -> list[ShotEvent]:
    if not bounces:
        return shot_events

    min_after = max(5, int(fps * 0.2))
    last_event_frame = max(event.frame for event in shot_events)
    kept = []
    for event in sorted(shot_events, key=lambda shot: shot.frame):
        previous_bounces = [frame for frame in bounces if frame < event.frame]
        if not previous_bounces:
            kept.append(event)
            continue

        previous_bounce = max(previous_bounces)
        gap = event.frame - previous_bounce
        if gap < min_after:
            continue

        if event.reason == "post_bounce_hit_window" and not _has_contact_turn(
            ball_positions,
            event.frame,
            min_turn=max(18, int(fps * 0.7)),
        ):
            continue

        next_bounces = [frame for frame in bounces if frame > event.frame]
        if not next_bounces and gap <= max(18, int(fps * 0.72)):
            if not _has_contact_turn(ball_positions, event.frame, min_turn=max(24, int(fps * 0.9))):
                continue
        if event.frame == last_event_frame and gap <= max(18, int(fps * 0.72)):
            if not _has_contact_turn(ball_positions, event.frame, min_turn=max(18, int(fps * 0.7))):
                continue

        kept.append(event)

    return kept


def _first_strong_contact_turn(ball_positions, start: int, end: int, fps: int) -> int | None:
    min_turn = max(24, int(fps * 0.9))
    for frame_num in range(max(1, start), min(end, len(ball_positions) - 2) + 1):
        if _has_contact_turn(ball_positions, frame_num, min_turn=min_turn):
            return frame_num
    return None


def _has_contact_turn(ball_positions, frame_num: int, min_turn: float) -> bool:
    prev_v = _local_y_velocity(ball_positions, frame_num, direction=-1)
    next_v = _local_y_velocity(ball_positions, frame_num, direction=1)
    if prev_v is None or next_v is None:
        return False

    direction_flip = (prev_v > 0 >= next_v) or (prev_v <= 0 < next_v)
    return direction_flip and abs(next_v - prev_v) >= min_turn


def _merge_shot_events(events: list[ShotEvent], min_gap: int):
    events = sorted(events, key=lambda event: event.frame)
    merged = []

    for event in events:
        if not merged or event.frame - merged[-1].frame >= min_gap:
            merged.append(event)
            continue

        previous = merged[-1]
        if _shot_priority(event) > _shot_priority(previous):
            merged[-1] = event

    return merged


def _shot_priority(event: ShotEvent):
    priority = 0
    if event.player_role is not None:
        priority += 2
    if event.ball_speed_kmh is not None:
        priority += 1
    if event.reason == "post_bounce_hit_window":
        priority += 1
    return priority


def _has_enough_track_context(ball_positions, frame_num: int, radius: int):
    start = max(0, frame_num - radius)
    end = min(len(ball_positions), frame_num + radius + 1)
    valid = sum(1 for point in ball_positions[start:end] if _point_or_none(point) is not None)
    return valid >= max(3, radius)


def recover_occluded_bounces(ball_positions, shot_frames: list[int], fps: int) -> set[int]:
    """Estimate bounces hidden inside short missing-track gaps."""
    recovered = set()
    max_gap_frames = max(3, int(fps * 0.22))
    min_gap_frames = 2

    index = 0
    while index < len(ball_positions):
        if _point_or_none(ball_positions[index]) is not None:
            index += 1
            continue

        gap_start = index
        while index < len(ball_positions) and _point_or_none(ball_positions[index]) is None:
            index += 1
        gap_end = index - 1
        gap_length = gap_end - gap_start + 1

        if gap_length < min_gap_frames or gap_length > max_gap_frames:
            continue

        before = _nearest_valid_point(ball_positions, gap_start - 1, step=-1, max_steps=4)
        after = _nearest_valid_point(ball_positions, gap_end + 1, step=1, max_steps=4)
        if before is None or after is None:
            continue

        before_frame, before_point = before
        after_frame, after_point = after
        if _crosses_shot(before_frame, after_frame, shot_frames):
            continue

        before_velocity = _local_y_velocity(ball_positions, before_frame, direction=-1)
        after_velocity = _local_y_velocity(ball_positions, after_frame, direction=1)
        if before_velocity is None or after_velocity is None:
            continue

        # Image y grows downward: falling is positive, rising is negative.
        falling_then_rising = before_velocity > 0 and after_velocity < 0
        falling_then_low_before_next_shot = (
            before_velocity > 0
            and after_point[1] >= before_point[1]
            and _next_shot_after(gap_end, shot_frames, max_frames=int(fps * 0.4)) is not None
        )
        if not (falling_then_rising or falling_then_low_before_next_shot):
            continue

        recovered.add((gap_start + gap_end) // 2)

    return recovered


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
        if point is None:
            point = _interpolated_position_at(
                projected_ball_positions,
                bounce_frame,
                max_gap=8,
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


def _nearest_valid_point(ball_positions, start: int, step: int, max_steps: int):
    frame = start
    checked = 0
    while 0 <= frame < len(ball_positions) and checked < max_steps:
        point = _point_or_none(ball_positions[frame])
        if point is not None:
            return frame, point
        frame += step
        checked += 1
    return None


def _interpolated_position_at(positions, frame_num: int, max_gap: int):
    before = _nearest_valid_point(positions, frame_num - 1, step=-1, max_steps=max_gap)
    after = _nearest_valid_point(positions, frame_num + 1, step=1, max_steps=max_gap)
    if before is None or after is None:
        return None

    before_frame, before_point = before
    after_frame, after_point = after
    frame_span = after_frame - before_frame
    if frame_span <= 0:
        return None

    t = (frame_num - before_frame) / frame_span
    x = before_point[0] + (after_point[0] - before_point[0]) * t
    y = before_point[1] + (after_point[1] - before_point[1]) * t
    return float(x), float(y)


def _local_y_velocity(ball_positions, frame: int, direction: int):
    current = _point_or_none(ball_positions[frame])
    if current is None:
        return None

    other = _nearest_valid_point(
        ball_positions,
        frame + direction,
        step=direction,
        max_steps=4,
    )
    if other is None:
        return None

    other_frame, other_point = other
    frame_delta = frame - other_frame
    if frame_delta == 0:
        return None
    return (current[1] - other_point[1]) / frame_delta


def _crosses_shot(start_frame: int, end_frame: int, shot_frames: list[int]):
    return any(start_frame <= shot_frame <= end_frame for shot_frame in shot_frames)


def _next_shot_after(frame_num: int, shot_frames: list[int], max_frames: int):
    for shot_frame in shot_frames:
        if 0 < shot_frame - frame_num <= max_frames:
            return shot_frame
    return None


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


def _total_distance_m(
    positions,
    scales: tuple[float, float] | None,
    max_speed_kmh: float | None,
    fps: int,
):
    if scales is None:
        return None

    total_distance = 0.0
    for previous, current in zip(positions, positions[1:]):
        if previous is None or current is None:
            continue

        dx = (current[0] - previous[0]) * scales[0]
        dy = (current[1] - previous[1]) * scales[1]
        distance = float(np.sqrt(dx * dx + dy * dy))

        speed_kmh = distance * fps * 3.6
        if max_speed_kmh is not None and speed_kmh > max_speed_kmh:
            continue

        total_distance += distance

    return total_distance


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
    if not ball_speeds:
        return None

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
        distance_m = _total_distance_m(role_positions, scales, max_speed_kmh=45, fps=fps)
        stats[role] = PlayerStats(
            shots=shot_counts[role],
            total_distance_m=distance_m,
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
            f"dist {_format_distance(player_stats.total_distance_m)}, "
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


def _format_distance(value):
    if value is None:
        return "n/a"
    return f"{value:.1f} m"


def _format_bounce_summary(bounce_events: list[BounceEvent]):
    in_count = sum(1 for event in bounce_events if event.in_out == "in")
    out_count = sum(1 for event in bounce_events if event.in_out == "out")
    unknown_count = sum(1 for event in bounce_events if event.in_out == "unknown")
    return f"Bounces: in {in_count}, out {out_count}, unknown {unknown_count}"
