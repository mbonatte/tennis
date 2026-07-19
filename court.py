from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch

from court_detection_net import CourtDetectorNet
from court_reference import CourtReference
from tennis_analyzer.errors import VideoProcessingError


class CourtTracker:
    """Own one court detector for an entire analysis stage."""

    def __init__(self, detector, device, batch_size=None):
        self.detector = detector
        self.device = torch.device(device)
        self.batch_size = batch_size or (8 if self.device.type == "cuda" else 1)

    @classmethod
    def from_checkpoint(cls, model_path="weights/tennis_court.pt", device_name=None, batch_size=None):
        device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        return cls(CourtDetectorNet(model_path, device), device, batch_size)

    def process_chunk(self, frames):
        if not frames:
            return [], []
        homographies, keypoints = self.detector.infer_model(frames, batch_size=self.batch_size)
        if len(homographies) != len(frames) or len(keypoints) != len(frames):
            raise VideoProcessingError("Court inference returned an unexpected number of frame results")
        return homographies, keypoints

    def close(self):
        self.detector = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def track_court(frames, model_path="weights/tennis_court.pt", device_name=None):
    """Compatibility helper; the production pipeline owns ``CourtTracker``."""
    tracker = CourtTracker.from_checkpoint(model_path, device_name)
    try:
        return tracker.process_chunk(frames)
    finally:
        tracker.close()


def draw_court(
    frames,
    homography_matrices,
    kps_courts,
    ball_track=None,
    bounces=None,
    player_tracks=None,
):
    """Draw court keypoints plus a minimap with projected tracked objects."""
    court_img = get_court_img()
    output_frames = []
    bounces = bounces or set()

    frame_height, frame_width = frames[0].shape[:2]
    width_minimap = min(166, max(60, frame_width // 4))
    height_minimap = min(350, max(100, frame_height - 60))

    for frame_num, (frame, homography_matrix, kps_court) in enumerate(
        zip(frames, homography_matrices, kps_courts, strict=False)
    ):
        img_res = frame.copy()

        # draw court keypoints
        if kps_court is not None:
            # print(kps_court)
            for key_point in kps_court:
                # print(key_point)
                x = int(key_point[0][0])
                y = int(key_point[0][1])
                cv2.circle(img_res, (x, y), radius=0, color=(0, 0, 255), thickness=10)

        height, width, _ = img_res.shape

        minimap = court_img.copy()
        if homography_matrix is not None:
            if ball_track is not None and frame_num < len(ball_track):
                ball_point = ball_track[frame_num]
                if _has_point(ball_point):
                    _draw_projected_point(
                        minimap,
                        homography_matrix,
                        ball_point,
                        color=(0, 255, 255),
                        radius=30,
                    )

            if frame_num in bounces and ball_track is not None and frame_num < len(ball_track):
                ball_point = ball_track[frame_num]
                if not _has_point(ball_point):
                    ball_point = _interpolated_point(ball_track, frame_num, max_gap=8)
                if _has_point(ball_point):
                    _draw_projected_point(
                        minimap,
                        homography_matrix,
                        ball_point,
                        color=(0, 165, 255),
                        radius=60,
                    )

            if player_tracks is not None and frame_num < len(player_tracks):
                for player in player_tracks[frame_num]:
                    foot_point = _player_foot_point(player)
                    is_top_player = player.role == "top_player"
                    color = (255, 0, 0) if is_top_player else (0, 255, 0)
                    label = "T" if is_top_player else "B"
                    _draw_projected_point(
                        minimap,
                        homography_matrix,
                        foot_point,
                        color=color,
                        radius=36,
                        label=label,
                    )

        minimap = cv2.resize(minimap, (width_minimap, height_minimap))
        top = max(0, min(30, height - height_minimap))
        right = max(width_minimap, width - 30)
        left = right - width_minimap
        img_res[top : (top + height_minimap), left:right, :] = minimap
        output_frames.append(img_res)
    return output_frames


def draw_court_overlay_in_place(frame, homography_matrix, ball_point=None, bounce=False, players=None):
    """Draw only the court minimap onto an already-owned render frame."""
    minimap = get_court_img().copy()
    if homography_matrix is not None:
        if _has_point(ball_point):
            _draw_projected_point(minimap, homography_matrix, ball_point, color=(0, 255, 255), radius=30)
            if bounce:
                _draw_projected_point(minimap, homography_matrix, ball_point, color=(0, 165, 255), radius=60)
        for player in players or []:
            is_top_player = player.role == "top_player"
            _draw_projected_point(
                minimap,
                homography_matrix,
                _player_foot_point(player),
                color=(255, 0, 0) if is_top_player else (0, 255, 0),
                radius=36,
                label="T" if is_top_player else "B",
            )

    frame_height, frame_width = frame.shape[:2]
    width = min(166, max(60, frame_width // 4))
    height = min(350, max(100, frame_height - 60))
    minimap = cv2.resize(minimap, (width, height))
    top = max(0, min(30, frame_height - height))
    right = max(width, frame_width - 30)
    left = right - width
    frame[top : top + height, left:right, :] = minimap
    return frame


def _has_point(point):
    return point is not None and point[0] is not None and point[1] is not None


def _interpolated_point(points, frame_num, max_gap):
    before = _nearest_valid_point(points, frame_num - 1, step=-1, max_steps=max_gap)
    after = _nearest_valid_point(points, frame_num + 1, step=1, max_steps=max_gap)
    if before is None or after is None:
        return (None, None)

    before_frame, before_point = before
    after_frame, after_point = after
    frame_span = after_frame - before_frame
    if frame_span <= 0:
        return (None, None)

    t = (frame_num - before_frame) / frame_span
    x = before_point[0] + (after_point[0] - before_point[0]) * t
    y = before_point[1] + (after_point[1] - before_point[1]) * t
    return (x, y)


def _nearest_valid_point(points, start, step, max_steps):
    frame = start
    checked = 0
    while 0 <= frame < len(points) and checked < max_steps:
        point = points[frame]
        if _has_point(point):
            return frame, point
        frame += step
        checked += 1
    return None


def _draw_projected_point(image, homography_matrix, point, color, radius, label=None):
    point_array = np.array(point, dtype=np.float32).reshape(1, 1, 2)
    projected = cv2.perspectiveTransform(point_array, homography_matrix)
    x = int(projected[0, 0, 0])
    y = int(projected[0, 0, 1])

    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        cv2.circle(image, (x, y), radius=radius, color=color, thickness=12)
        if label:
            cv2.putText(
                image,
                label,
                (x - 16, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                color,
                5,
            )


def _player_foot_point(player):
    x1, _, x2, y2 = player.bbox
    return ((x1 + x2) / 2, y2)


def draw_keypoints(frames, kps_court):
    output_frames = []

    for i, frame in enumerate(frames):
        img_res = frame.copy()

        if kps_court[i] is not None:
            for j, kp in enumerate(kps_court[i]):
                x = int(kp[0, 0])
                y = int(kp[0, 1])

                cv2.circle(img_res, (x, y), radius=8, color=(0, 0, 255), thickness=-1)
                cv2.putText(img_res, str(j), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        output_frames.append(img_res)

    return output_frames


@lru_cache(maxsize=1)
def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2) * 255).astype(np.uint8)
    return court_img
