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
    ball_track = list(raw_track)
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
