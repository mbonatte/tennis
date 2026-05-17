from __future__ import annotations


import cv2
import torch
import numpy as np

from court_detection_net import CourtDetectorNet
from court_reference import CourtReference


def track_court(frames, model_path="weights/tennis_court.pt"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    court_detector = CourtDetectorNet(model_path, device)
    homography_matrices, kps_court = court_detector.infer_model(frames)

    return homography_matrices, kps_court 

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

    width_minimap = 166
    height_minimap = 350
    
    for frame_num, (frame, homography_matrix, kps_court) in enumerate(
        zip(frames, homography_matrices, kps_courts)
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
        img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
        output_frames.append(img_res)
    return output_frames


def _has_point(point):
    return point is not None and point[0] is not None and point[1] is not None


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
                cv2.putText(
                    img_res,
                    str(j),
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        output_frames.append(img_res)

    return output_frames

def get_court_img():
    court_reference = CourtReference()
    court = court_reference.build_court_reference()
    court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
    court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
    return court_img
