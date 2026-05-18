from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

from BallTrack.ball_tracker import load_model, infer_model_batched, remove_outliers, split_track, interpolation


def track_ball(frames, extrapolation=True):
    project_root = Path(__file__).resolve().parent

    model_path = project_root / "weights" / "tracknet_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_threads = max(1, (os.cpu_count() or 4) - 1)
    
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)

    print(f"Using device: {device}")

    use_compile = False
    model = load_model(model_path, device, use_compile=use_compile)

    # ball_track, dists = infer_model(frames, model, device)
    ball_track, dists = infer_model_batched(
        frames,
        model,
        device=device,
        batch_size=8,
        use_amp=False,
    )
    ball_track = remove_outliers(ball_track, dists)

    if extrapolation:
        subtracks = split_track(ball_track)
        for start, end in subtracks:
            ball_subtrack = ball_track[start:end]
            ball_track[start:end] = interpolation(ball_subtrack)

    return ball_track

def draw_track(frames, ball_track, trace=7):
    """
    Draw ball tracking trail on frames.

    Returns:
        processed_frames: list of frames with drawings
    """
    processed_frames = []

    for num in range(len(frames)):
        frame = frames[num].copy()

        for i in range(trace):
            idx = num - i
            if idx < 0:
                break

            if _has_valid_ball_point(ball_track[idx]):
                x = int(ball_track[idx][0])
                y = int(ball_track[idx][1])

                cv2.circle(
                    frame,
                    (x, y),
                    radius=0,
                    color=(0, 0, 255),
                    thickness=max(1, 10 - i),
                )
            else:
                break

        processed_frames.append(frame)

    return processed_frames


def _has_valid_ball_point(point):
    if point is None or point[0] is None or point[1] is None:
        return False
    return np.isfinite(point[0]) and np.isfinite(point[1])
