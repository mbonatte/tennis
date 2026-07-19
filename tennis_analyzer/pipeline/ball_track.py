from __future__ import annotations

import math
from itertools import groupby

import numpy as np

BallPoint = tuple[float | None, float | None]


def euclidean(first: BallPoint, second: BallPoint) -> float:
    if first[0] is None or second[0] is None:
        return -1.0
    return math.hypot(first[0] - second[0], first[1] - second[1])


def ball_distances(track: list[BallPoint]) -> list[float]:
    if not track:
        return []
    return [-1.0, *[euclidean(current, previous) for previous, current in zip(track, track[1:], strict=False)]]


def remove_abrupt_jumps(
    track: list[BallPoint],
    *,
    minimum_jump: float = 70.0,
    velocity_multiplier: float = 3.0,
) -> list[BallPoint]:
    """Remove a one-frame excursion that cannot belong to a continuous track.

    TrackNet may occasionally lock onto a similarly sized object for one frame.
    A point is rejected only when it is far from both neighbours while those
    neighbours remain mutually consistent.  The threshold adapts to the local
    observed ball speed, so a sustained fast rally is not treated as a jump.
    Missing points are intentionally left for the global interpolation pass.
    """
    if len(track) < 3:
        return track

    distances = [distance for distance in ball_distances(track) if distance >= 0.0 and math.isfinite(distance)]
    typical_speed = float(np.median(distances)) if distances else 0.0
    max_excursion = max(minimum_jump, typical_speed * velocity_multiplier)

    for index in range(1, len(track) - 1):
        previous, current, following = track[index - 1 : index + 2]
        if any(point[0] is None or point[1] is None for point in (previous, current, following)):
            continue
        previous_distance = euclidean(previous, current)
        following_distance = euclidean(current, following)
        neighbour_distance = euclidean(previous, following)
        if (
            previous_distance > max_excursion
            and following_distance > max_excursion
            and neighbour_distance <= max_excursion
        ):
            track[index] = (None, None)
    return track


def remove_outliers(track: list[BallPoint], distances: list[float], max_distance: float = 100) -> list[BallPoint]:
    outliers = list(np.where(np.asarray(distances) > max_distance)[0])
    for index in outliers.copy():
        next_distance = distances[index + 1] if index + 1 < len(distances) else None
        if next_distance is not None and (next_distance > max_distance or next_distance == -1):
            track[index] = (None, None)
            if index in outliers:
                outliers.remove(index)
        elif index > 0 and distances[index - 1] == -1:
            track[index - 1] = (None, None)
    return track


def split_track(track: list[BallPoint], max_gap: int = 4, max_distance_gap: float = 80, min_track: int = 5):
    groups = [
        (missing, sum(1 for _ in values)) for missing, values in groupby(0 if p[0] is not None else 1 for p in track)
    ]
    cursor = 0
    start = 0
    result = []
    for group_index, (missing, length) in enumerate(groups):
        if missing == 1 and 0 < group_index < len(groups) - 1:
            distance = euclidean(track[cursor - 1], track[cursor + length])
            if length >= max_gap or distance / max(length, 1) > max_distance_gap:
                if cursor - start > min_track:
                    result.append([start, cursor])
                start = cursor + length - 1
        cursor += length
    if len(track) - start > min_track:
        result.append([start, len(track)])
    return result


def interpolation(coordinates: list[BallPoint]) -> list[BallPoint]:
    x_values = np.asarray([point[0] if point[0] is not None else np.nan for point in coordinates], dtype=float)
    y_values = np.asarray([point[1] if point[1] is not None else np.nan for point in coordinates], dtype=float)
    for values in (x_values, y_values):
        valid = np.where(~np.isnan(values))[0]
        missing = np.where(np.isnan(values))[0]
        if len(valid) >= 2:
            values[missing] = np.interp(missing, valid, values[valid])
    return [
        (float(x), float(y)) if np.isfinite(x) and np.isfinite(y) else (None, None)
        for x, y in zip(x_values, y_values, strict=True)
    ]


def postprocess_ball_track(raw_track: list[BallPoint], *, extrapolation: bool = True) -> list[BallPoint]:
    """Apply continuity-based filtering once, after all raw chunks are joined."""
    ball_track = [
        point
        if point[0] is not None and point[1] is not None and np.isfinite(point[0]) and np.isfinite(point[1])
        else (None, None)
        for point in raw_track
    ]
    ball_track = remove_abrupt_jumps(ball_track)
    ball_track = remove_outliers(ball_track, ball_distances(ball_track))
    if extrapolation:
        for start, end in split_track(ball_track):
            ball_track[start:end] = interpolation(ball_track[start:end])
    return [
        (float(point[0]), float(point[1]))
        if point[0] is not None and point[1] is not None and np.isfinite(point[0]) and np.isfinite(point[1])
        else (None, None)
        for point in ball_track
    ]
