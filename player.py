import copy
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class TrackedPlayer:
    role: str                         # "top_player" or "bottom_player"
    track_id: Optional[int]
    bbox: Tuple[int, int, int, int]    # x1, y1, x2, y2
    conf: float
    center: Tuple[int, int]


@dataclass
class TrackedPosePlayer(TrackedPlayer):
    keypoints_xy: np.ndarray           # shape: (17, 2)
    keypoints_conf: np.ndarray         # shape: (17,)


@dataclass
class HybridTrackedPlayer:
    role: str
    track_id: Optional[int]

    # Always available from box tracker
    bbox: Tuple[int, int, int, int]
    conf: float
    center: Tuple[int, int]

    # Optional pose data
    has_pose: bool = False
    keypoints_xy: Optional[np.ndarray] = None
    keypoints_conf: Optional[np.ndarray] = None

class BasePlayerTracker:
    """
    Parent class for tennis player tracking.

    Child classes should implement:
        - _extract_detections()
        - draw()
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.5,
        tracker: str = "bytetrack.yaml",
        min_box_area: int = 500,
        device: str = "cpu",
    ):
        self.device = device
        self.model = YOLO(model_path)
        self.model.overrides["verbose"] = False
        self.model.to(device)
        self.model.model.eval()

        self.conf = conf
        self.tracker = tracker
        self.min_box_area = min_box_area

        self.last_top_player = None
        self.last_bottom_player = None
        self._top_missing_frames = 0
        self._bottom_missing_frames = 0
        self.max_missing_frames = 8

    @torch.inference_mode()
    def track_frame(self, frame):
        """
        Run YOLO tracking on one frame.
        """
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],          # person class
            conf=self.conf,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        detections = self._extract_detections(result)
        players = self._assign_top_bottom_players(detections, frame.shape)

        return players, result

    def track_frames(self, frames):
        """
        Run tracking on a list of frames.
        """
        all_players = []
        raw_results = []

        for frame in frames:
            players, result = self.track_frame(frame)
            all_players.append(players)
            raw_results.append(result)

        return all_players, raw_results

    def close(self):
        self.model = None
        self.last_top_player = None
        self.last_bottom_player = None
        self._top_missing_frames = 0
        self._bottom_missing_frames = 0
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def annotate_frames(self, frames, player_tracks):
        """
        Draw tracking result on a list of frames.
        """
        annotated_frames = []

        for frame, players in zip(frames, player_tracks):
            annotated_frames.append(self.draw(frame, players))

        return annotated_frames

    def _extract_base_box_data(self, result):
        """
        Shared box extraction used by both box and pose trackers.
        """
        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        ids = result.boxes.id
        if ids is not None:
            ids = ids.cpu().numpy().astype(int)
        else:
            ids = [None] * len(boxes)

        for box, conf, track_id in zip(boxes, confs, ids):
            x1, y1, x2, y2 = box.astype(int)

            w = x2 - x1
            h = y2 - y1
            area = w * h

            if area < self.min_box_area:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append({
                "track_id": None if track_id is None else int(track_id),
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "center": (cx, cy),
                "area": area,
            })

        return detections

    def _assign_top_bottom_players(self, detections, frame_shape):
        """
        Assign detections to top_player and bottom_player.

        This works well for a fixed behind-the-court tennis camera:
        - far-side player is usually in the top half
        - near-side player is usually in the bottom half
        """
        frame_h, _ = frame_shape[:2]
        mid_y = frame_h / 2

        top_candidates = []
        bottom_candidates = []

        for det in detections:
            _, cy = det["center"]

            # Once ByteTrack has established an identity, its player may move
            # across the image midpoint.  Preserve that identity instead of
            # reassigning it solely because of a fixed screen-half heuristic.
            if self._matches_previous_role(det, self.last_top_player):
                top_candidates.append(det)
            elif self._matches_previous_role(det, self.last_bottom_player):
                bottom_candidates.append(det)
            elif cy < mid_y and self._is_top_player_candidate(det, frame_shape):
                top_candidates.append(det)
            elif cy >= mid_y and self._is_bottom_player_candidate(det, frame_shape):
                bottom_candidates.append(det)

        top_player = self._select_continuous_candidate(top_candidates, self.last_top_player, frame_shape)
        bottom_player = self._select_continuous_candidate(bottom_candidates, self.last_bottom_player, frame_shape)

        top_player = self._accept_or_hold("top", top_player)
        bottom_player = self._accept_or_hold("bottom", bottom_player)

        players = []

        if top_player is not None:
            players.append(self._make_player("top_player", top_player))

        if bottom_player is not None:
            players.append(self._make_player("bottom_player", bottom_player))

        return players

    def _select_continuous_candidate(self, candidates, previous, frame_shape):
        """Prefer a plausible continuation over a high-confidence bystander."""
        if not candidates:
            return None
        if previous is None:
            return max(candidates, key=self._score_detection)

        same_identity = [
            candidate
            for candidate in candidates
            if previous["track_id"] is not None and candidate["track_id"] == previous["track_id"]
        ]
        if same_identity:
            return max(same_identity, key=self._score_detection)

        frame_w = frame_shape[1]
        previous_width = max(1, previous["bbox"][2] - previous["bbox"][0])
        # A changed tracker ID is a weaker continuity signal than an existing
        # ID, so use a tighter reacquisition gate.  This prevents the far-side
        # role from taking the near player when its own detection is lost.
        max_center_jump = max(frame_w * 0.10, previous_width * 3.0)
        continuous = []
        for candidate in candidates:
            distance = self._center_distance(candidate["center"], previous["center"])
            if distance > max_center_jump:
                continue
            identity_bonus = (
                0.75
                if candidate["track_id"] is not None and candidate["track_id"] == previous["track_id"]
                else 0.0
            )
            continuity_bonus = 0.75 * (1.0 - distance / max_center_jump)
            continuous.append((self._score_detection(candidate) + identity_bonus + continuity_bonus, candidate))
        return max(continuous, key=lambda item: item[0])[1] if continuous else None

    def _accept_or_hold(self, role, candidate):
        previous_attribute = f"last_{role}_player"
        missing_attribute = f"_{role}_missing_frames"
        if candidate is not None:
            setattr(self, previous_attribute, candidate)
            setattr(self, missing_attribute, 0)
            return candidate
        missing_frames = getattr(self, missing_attribute) + 1
        setattr(self, missing_attribute, missing_frames)
        previous = getattr(self, previous_attribute)
        if previous is not None and missing_frames <= self.max_missing_frames:
            return previous
        setattr(self, previous_attribute, None)
        return None

    @staticmethod
    def _center_distance(first, second):
        return float(np.hypot(first[0] - second[0], first[1] - second[1]))

    @staticmethod
    def _matches_previous_role(det, previous):
        return (
            previous is not None
            and previous["track_id"] is not None
            and det["track_id"] is not None
            and det["track_id"] == previous["track_id"]
        )

    def _score_detection(self, det):
        """
        Default scoring function.

        Child classes may override this.
        """
        area_score = min(det["area"] / 50000, 1.0)
        return det["conf"] * 0.7 + area_score * 0.3

    def _is_top_player_candidate(self, det, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        x1, _, x2, y2 = det["bbox"]
        center_x, _ = det["center"]
        min_foot_y = self._top_player_min_foot_y(frame_h)
        side_margin_x = self._top_player_side_margin_x(frame_w)
        if y2 < min_foot_y:
            return False
        if y2 > frame_h * 0.50:
            return False
        if center_x < side_margin_x or center_x > frame_w - side_margin_x:
            return False
        if x2 <= x1:
            return False
        return True

    def _top_player_min_foot_y(self, frame_h):
        high_res_adjustment = max(0.0, min((frame_h - 720) / 360, 1.0)) * 0.04
        return frame_h * (0.18 + high_res_adjustment)

    def _top_player_side_margin_x(self, frame_w):
        return frame_w * 0.20

    def _is_bottom_player_candidate(self, det, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        _, _, _, y2 = det["bbox"]
        center_x, _ = det["center"]
        if y2 < frame_h * 0.50:
            return False
        if center_x < frame_w * 0.20 or center_x > frame_w * 0.98:
            return False
        return True

    def _extract_detections(self, result):
        """
        Must be implemented by child class.
        """
        raise NotImplementedError

    def _make_player(self, role, det):
        """
        Must be implemented by child class.
        """
        raise NotImplementedError

    def draw(self, frame, players):
        """
        Must be implemented by child class.
        """
        raise NotImplementedError
    
class BoxPlayerTracker(BasePlayerTracker):
    """
    Child class for bounding-box player tracking.
    """

    def __init__(
        self,
        model_path: str = "weights/yolo11n.pt",
        conf: float = 0.5,
        tracker: str = "bytetrack.yaml",
        min_box_area: int = 500,
        device: str = "cpu",
    ):
        super().__init__(
            model_path=model_path,
            conf=conf,
            tracker=tracker,
            min_box_area=min_box_area,
            device=device,
        )

    def _extract_detections(self, result):
        return self._extract_base_box_data(result)

    def _make_player(self, role, det):
        return TrackedPlayer(
            role=role,
            track_id=det["track_id"],
            bbox=det["bbox"],
            conf=det["conf"],
            center=det["center"],
        )

    def draw(self, frame, players, *, copy_frame=True):
        annotated = frame.copy() if copy_frame else frame

        for player in players:
            x1, y1, x2, y2 = player.bbox

            label = player.role
            if player.track_id is not None:
                label += f" ID:{player.track_id}"

            label += f" {player.conf:.2f}"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.circle(
                annotated,
                player.center,
                5,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return annotated
    
class PosePlayerTracker(BasePlayerTracker):
    """
    Child class for pose/keypoint player tracking.
    """

    COCO_SKELETON = [
        (5, 7),    # left shoulder to left elbow
        (7, 9),    # left elbow to left wrist
        (6, 8),    # right shoulder to right elbow
        (8, 10),   # right elbow to right wrist
        (5, 6),    # shoulders
        (5, 11),   # left shoulder to left hip
        (6, 12),   # right shoulder to right hip
        (11, 12),  # hips
        (11, 13),  # left hip to left knee
        (13, 15),  # left knee to left ankle
        (12, 14),  # right hip to right knee
        (14, 16),  # right knee to right ankle
    ]

    KEYPOINT_NAMES = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
    }

    def __init__(
        self,
        model_path: str = "weights/yolo11n-pose.pt",
        conf: float = 0.5,
        tracker: str = "bytetrack.yaml",
        min_box_area: int = 500,
        keypoint_conf_threshold: float = 0.3,
        device: str = "cpu",
    ):
        super().__init__(
            model_path=model_path,
            conf=conf,
            tracker=tracker,
            min_box_area=min_box_area,
            device=device,
        )

        self.keypoint_conf_threshold = keypoint_conf_threshold

    def _extract_detections(self, result):
        detections = self._extract_base_box_data(result)

        if not detections:
            return []

        if result.keypoints is None:
            return []

        keypoints_xy = result.keypoints.xy.cpu().numpy()

        if result.keypoints.conf is not None:
            keypoints_conf = result.keypoints.conf.cpu().numpy()
        else:
            keypoints_conf = np.ones((len(keypoints_xy), keypoints_xy.shape[1]))

        # Important:
        # _extract_base_box_data filters small boxes.
        # So we need to rebuild pose detections carefully using the original result data.
        pose_detections = []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        ids = result.boxes.id
        if ids is not None:
            ids = ids.cpu().numpy().astype(int)
        else:
            ids = [None] * len(boxes)

        for box, conf, track_id, kpts_xy, kpts_conf in zip(
            boxes,
            confs,
            ids,
            keypoints_xy,
            keypoints_conf,
        ):
            x1, y1, x2, y2 = box.astype(int)

            w = x2 - x1
            h = y2 - y1
            area = w * h

            if area < self.min_box_area:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            pose_detections.append({
                "track_id": None if track_id is None else int(track_id),
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "center": (cx, cy),
                "area": area,
                "keypoints_xy": kpts_xy,
                "keypoints_conf": kpts_conf,
            })

        return pose_detections

    def _score_detection(self, det):
        pose_visibility = float(
            np.mean(det["keypoints_conf"] > self.keypoint_conf_threshold)
        )

        area_score = min(det["area"] / 50000, 1.0)

        return (
            det["conf"] * 0.5
            + pose_visibility * 0.3
            + area_score * 0.2
        )

    def _make_player(self, role, det):
        return TrackedPosePlayer(
            role=role,
            track_id=det["track_id"],
            bbox=det["bbox"],
            conf=det["conf"],
            center=det["center"],
            keypoints_xy=det["keypoints_xy"],
            keypoints_conf=det["keypoints_conf"],
        )

    def draw(self, frame, players, *, copy_frame=True):
        annotated = frame.copy() if copy_frame else frame

        for player in players:
            self._draw_skeleton(annotated, player)
            self._draw_keypoints(annotated, player)
            self._draw_label(annotated, player)

        return annotated

    def _draw_skeleton(self, frame, player):
        keypoints_xy = player.keypoints_xy
        keypoints_conf = player.keypoints_conf

        for start_idx, end_idx in self.COCO_SKELETON:
            if keypoints_conf[start_idx] < self.keypoint_conf_threshold:
                continue

            if keypoints_conf[end_idx] < self.keypoint_conf_threshold:
                continue

            x1, y1 = keypoints_xy[start_idx].astype(int)
            x2, y2 = keypoints_xy[end_idx].astype(int)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

    def _draw_keypoints(self, frame, player):
        keypoints_xy = player.keypoints_xy
        keypoints_conf = player.keypoints_conf

        for idx, (x, y) in enumerate(keypoints_xy):
            if keypoints_conf[idx] < self.keypoint_conf_threshold:
                continue

            cv2.circle(
                frame,
                (int(x), int(y)),
                4,
                (0, 0, 255),
                -1,
            )

    def _draw_label(self, frame, player):
        label = player.role

        if player.track_id is not None:
            label += f" ID:{player.track_id}"

        label += f" {player.conf:.2f}"

        cx, cy = player.center

        cv2.putText(
            frame,
            label,
            (cx - 50, max(cy - 20, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

class HybridPlayerTracker(BoxPlayerTracker):
    """
    Uses box tracking as the primary player tracker.
    Uses pose detection as optional extra metadata.

    The player is never dropped just because pose is missing.
    """

    COCO_SKELETON = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    KEYPOINT_NAMES = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
    }

    def __init__(
        self,
        box_model_path: str = "weights/yolo11n.pt",
        pose_model_path: str = "weights/yolo11n-pose.pt",
        conf: float = 0.5,
        pose_conf: float = 0.35,
        tracker: str = "bytetrack.yaml",
        min_box_area: int = 500,
        keypoint_conf_threshold: float = 0.3,
        max_pose_match_distance: float = 120.0,
        top_recovery_conf: float = 0.15,
        top_recovery_zoom: float = 2.0,
        device: str = "cpu",
    ):
        super().__init__(
            model_path=box_model_path,
            conf=conf,
            tracker=tracker,
            min_box_area=min_box_area,
            device=device,
        )

        self.pose_model = YOLO(pose_model_path)
        self.pose_model.overrides["verbose"] = False
        self.pose_model.to(device)
        self.pose_model.model.eval()
        # Recovery needs an independent predictor so ROI inference cannot reset
        # ByteTrack state, but it does not need to read the checkpoint twice.
        self.recovery_model = copy.deepcopy(self.model)
        self.recovery_model.overrides["verbose"] = False
        self.recovery_model.model.eval()

        self.pose_conf = pose_conf
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.max_pose_match_distance = max_pose_match_distance
        self.top_recovery_conf = top_recovery_conf
        self.top_recovery_zoom = top_recovery_zoom
        self.last_valid_top_player = None
        self.last_valid_bottom_player = None

    @torch.inference_mode()
    def track_frame(self, frame):
        """
        1. Track players with box model.
        2. Detect poses with pose model.
        3. Attach nearest pose to each tracked box player.
        """
        box_players, box_result = super().track_frame(frame)
        box_players = self._recover_top_player_if_needed(frame, box_players)
        box_players = self._recover_bottom_player_if_needed(frame, box_players)

        pose_result = self._run_pose_model(frame)
        pose_detections = self._extract_pose_detections(pose_result)

        hybrid_players = []

        for player in box_players:
            matched_pose = self._match_pose_to_player(player, pose_detections)

            if matched_pose is None:
                hybrid_players.append(
                    HybridTrackedPlayer(
                        role=player.role,
                        track_id=player.track_id,
                        bbox=player.bbox,
                        conf=player.conf,
                        center=player.center,
                        has_pose=False,
                    )
                )
            else:
                hybrid_players.append(
                    HybridTrackedPlayer(
                        role=player.role,
                        track_id=player.track_id,
                        bbox=player.bbox,
                        conf=player.conf,
                        center=player.center,
                        has_pose=True,
                        keypoints_xy=matched_pose["keypoints_xy"],
                        keypoints_conf=matched_pose["keypoints_conf"],
                    )
                )

        return hybrid_players, {
            "box_result": box_result,
            "pose_result": pose_result,
        }

    def _recover_top_player_if_needed(self, frame, players):
        top_player = self._get_role(players, "top_player")

        if top_player is not None and self._is_valid_top_player(top_player, frame.shape):
            if top_player.track_id is not None:
                self.last_valid_top_player = top_player
            return players

        recovered_top = self._detect_top_player_in_zoomed_roi(frame)
        players = [player for player in players if player.role != "top_player"]

        if recovered_top is not None:
            self.last_valid_top_player = recovered_top
            players.append(recovered_top)

        return sorted(players, key=lambda player: 0 if player.role == "top_player" else 1)

    def _detect_top_player_in_zoomed_roi(self, frame):
        if self.last_valid_top_player is None:
            return None

        roi, offset = self._top_recovery_roi(frame, self.last_valid_top_player.bbox)
        if roi.size == 0:
            return None

        zoomed_roi = cv2.resize(
            roi,
            None,
            fx=self.top_recovery_zoom,
            fy=self.top_recovery_zoom,
            interpolation=cv2.INTER_LINEAR,
        )
        results = self.recovery_model.predict(
            zoomed_roi,
            classes=[0],
            conf=self.top_recovery_conf,
            device=self.device,
            verbose=False,
        )
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        best_player = None
        best_score = -np.inf

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box / self.top_recovery_zoom
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            bbox = tuple(int(value) for value in (x1, y1, x2, y2))
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            candidate = TrackedPlayer(
                role="top_player",
                track_id=None,
                bbox=bbox,
                conf=float(conf),
                center=center,
            )

            if not self._is_valid_top_player(candidate, frame.shape):
                continue

            distance = self._center_distance(center, self.last_valid_top_player.center)
            if distance > 180:
                continue

            score = float(conf) - distance / 500.0
            if score > best_score:
                best_score = score
                best_player = candidate

        return best_player

    def _top_recovery_roi(self, frame, bbox):
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        pad_x = max(140, int(width * 4.0))
        pad_y = max(80, int(height * 2.0))

        roi_x1 = max(0, x1 - pad_x)
        roi_y1 = max(0, y1 - pad_y)
        roi_x2 = min(frame_w, x2 + pad_x)
        roi_y2 = min(int(frame_h * 0.58), y2 + pad_y)

        return frame[roi_y1:roi_y2, roi_x1:roi_x2], (roi_x1, roi_y1)

    def _recover_bottom_player_if_needed(self, frame, players):
        bottom_player = self._get_role(players, "bottom_player")
        if bottom_player is not None and self._is_valid_bottom_recovery_player(bottom_player, frame.shape):
            if bottom_player.track_id is not None:
                self.last_valid_bottom_player = bottom_player
            return players

        recovered_bottom = self._detect_bottom_player_in_zoomed_roi(frame)
        players = [player for player in players if player.role != "bottom_player"]
        if recovered_bottom is not None:
            self.last_valid_bottom_player = recovered_bottom
            players.append(recovered_bottom)
        return sorted(players, key=lambda player: 0 if player.role == "top_player" else 1)

    def _detect_bottom_player_in_zoomed_roi(self, frame):
        if self.last_valid_bottom_player is None:
            return None

        roi, offset = self._bottom_recovery_roi(frame, self.last_valid_bottom_player.bbox)
        if roi.size == 0:
            return None
        zoomed_roi = cv2.resize(
            roi,
            None,
            fx=self.top_recovery_zoom,
            fy=self.top_recovery_zoom,
            interpolation=cv2.INTER_LINEAR,
        )
        result = self.recovery_model.predict(
            zoomed_roi,
            classes=[0],
            conf=self.top_recovery_conf,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        best_player = None
        best_score = -np.inf
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box / self.top_recovery_zoom
            bbox = tuple(int(value) for value in (x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1]))
            center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            candidate = TrackedPlayer("bottom_player", None, bbox, float(conf), center)
            if not self._is_valid_bottom_recovery_player(candidate, frame.shape):
                continue
            distance = self._center_distance(center, self.last_valid_bottom_player.center)
            max_distance = max(180.0, (self.last_valid_bottom_player.bbox[2] - self.last_valid_bottom_player.bbox[0]) * 5.0)
            if distance > max_distance:
                continue
            score = float(conf) - distance / 500.0
            if score > best_score:
                best_score = score
                best_player = candidate
        return best_player

    def _bottom_recovery_roi(self, frame, bbox):
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        pad_x = max(140, int(width * 4.0))
        pad_y = max(100, int(height * 2.5))
        roi_x1 = max(0, x1 - pad_x)
        roi_y1 = max(0, y1 - pad_y)
        roi_x2 = min(frame_w, x2 + pad_x)
        roi_y2 = min(frame_h, y2 + pad_y)
        return frame[roi_y1:roi_y2, roi_x1:roi_x2], (roi_x1, roi_y1)

    def _is_valid_bottom_recovery_player(self, player, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = player.bbox
        center_x, _ = player.center
        return (
            x2 > x1
            and y2 > y1
            and y2 - y1 >= frame_h * 0.06
            and y2 >= frame_h * 0.18
            and frame_w * 0.05 <= center_x <= frame_w * 0.98
        )

    def _is_valid_top_player(self, player, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        x1, _, x2, y2 = player.bbox
        box_height = y2 - player.bbox[1]
        center_x, _ = player.center
        if y2 < self._top_player_min_foot_y(frame_h):
            return False
        if y2 > frame_h * 0.58:
            return False
        if center_x < self._top_player_side_margin_x(frame_w) or center_x > frame_w - self._top_player_side_margin_x(frame_w):
            return False
        if x2 <= x1:
            return False
        if box_height < frame_h * 0.06:
            return False
        return True

    def _get_role(self, players, role):
        return next((player for player in players if player.role == role), None)

    def _run_pose_model(self, frame):
        results = self.pose_model.predict(
            frame,
            classes=[0],
            conf=self.pose_conf,
            device=self.device,
            verbose=False,
        )

        return results[0]

    def close(self):
        self.pose_model = None
        self.recovery_model = None
        self.last_valid_top_player = None
        self.last_valid_bottom_player = None
        super().close()

    def _extract_pose_detections(self, pose_result):
        detections = []

        if pose_result.boxes is None or pose_result.keypoints is None:
            return detections

        boxes = pose_result.boxes.xyxy.cpu().numpy()
        confs = pose_result.boxes.conf.cpu().numpy()
        keypoints_xy = pose_result.keypoints.xy.cpu().numpy()

        if pose_result.keypoints.conf is not None:
            keypoints_conf = pose_result.keypoints.conf.cpu().numpy()
        else:
            keypoints_conf = np.ones((len(keypoints_xy), keypoints_xy.shape[1]))

        for box, conf, kpts_xy, kpts_conf in zip(
            boxes,
            confs,
            keypoints_xy,
            keypoints_conf,
        ):
            x1, y1, x2, y2 = box.astype(int)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": float(conf),
                "center": (cx, cy),
                "keypoints_xy": kpts_xy,
                "keypoints_conf": kpts_conf,
            })

        return detections

    def _match_pose_to_player(self, player, pose_detections):
        """
        Match pose detection to tracked box player.

        Uses:
        - center distance
        - IoU overlap

        Returns the best matching pose, or None.
        """
        if not pose_detections:
            return None

        best_pose = None
        best_score = -1

        for pose in pose_detections:
            distance = self._center_distance(player.center, pose["center"])
            iou = self._bbox_iou(player.bbox, pose["bbox"])

            if distance > self.max_pose_match_distance and iou < 0.1:
                continue

            distance_score = max(
                0.0,
                1.0 - distance / self.max_pose_match_distance
            )

            score = iou * 0.7 + distance_score * 0.3

            if score > best_score:
                best_score = score
                best_pose = pose

        return best_pose

    def _center_distance(self, center_a, center_b):
        ax, ay = center_a
        bx, by = center_b

        return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))

    def _bbox_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter_area

        if union == 0:
            return 0.0

        return inter_area / union

    def draw(self, frame, players, *, copy_frame=True):
        annotated = frame.copy() if copy_frame else frame

        for player in players:
            self._draw_box(annotated, player)

            if player.has_pose:
                self._draw_skeleton(annotated, player)
                self._draw_keypoints(annotated, player)

        return annotated

    def _draw_box(self, frame, player):
        x1, y1, x2, y2 = player.bbox

        if player.has_pose:
            label = f"{player.role} pose"
        else:
            label = f"{player.role} box-only"

        if player.track_id is not None:
            label += f" ID:{player.track_id}"

        label += f" {player.conf:.2f}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.circle(
            frame,
            player.center,
            5,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def _draw_skeleton(self, frame, player):
        keypoints_xy = player.keypoints_xy
        keypoints_conf = player.keypoints_conf

        for start_idx, end_idx in self.COCO_SKELETON:
            if keypoints_conf[start_idx] < self.keypoint_conf_threshold:
                continue

            if keypoints_conf[end_idx] < self.keypoint_conf_threshold:
                continue

            x1, y1 = keypoints_xy[start_idx].astype(int)
            x2, y2 = keypoints_xy[end_idx].astype(int)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )

    def _draw_keypoints(self, frame, player):
        keypoints_xy = player.keypoints_xy
        keypoints_conf = player.keypoints_conf

        for idx, (x, y) in enumerate(keypoints_xy):
            if keypoints_conf[idx] < self.keypoint_conf_threshold:
                continue

            cv2.circle(
                frame,
                (int(x), int(y)),
                4,
                (0, 0, 255),
                -1,
            )
