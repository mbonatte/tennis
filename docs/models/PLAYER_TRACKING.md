# Player Tracking Model Card

## Model Identity

- **Feature Name**: Player Tracking & Pose Estimation
- **Checkpoint Filenames**:
  - Primary (Bounding Boxes): `yolo26n.pt`
  - Optional (Pose Estimation): `yolo26n-pose.pt`
- **Framework and Model Class**: PyTorch / Ultralytics YOLOv8/11.
- **Model Architecture**: Ultralytics YOLO CNN architectures with ByteTrack tracking head.
- **Upstream Repository/Publication**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
- **Upstream Version/Commit**: Unknown.
- **Training Dataset**: COCO Dataset (filtering for class `0` - Person).
- **Model License**: GPL-3.0 (Ultralytics).
- **Weight License**: Unknown (Likely Ultralytics proprietary or AGPL; manual verification required).
- **Checksum of Supported Checkpoints**:
  - `yolo26n.pt` SHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
  - `yolo26n-pose.pt` SHA-256: `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9`

---

## Purpose

- **Predictions**: Detects and tracks player bounding boxes `(x1, y1, x2, y2)` and skeletal joints (17 COCO keypoints).
- **Importance**: Identifies the players on the court, assigns roles (`top_player` vs `bottom_player`), measures running distance, and aligns shot frames based on racket contact proximity.
- **Dependencies**: Downstream shot detection and player statistics depend directly on these tracks.
- **Model-Free Operation**: No. Player statistics, role-based annotations, and proximity shot heuristics cannot run without this stage.

---

## Inputs

- **Tensor Format**: PyTorch FloatTensor.
- **Dimensions**: Handled dynamically by the Ultralytics YOLO backend (typically scales input to `640x640` pixels).
- **Temporal Context**: None (single-frame detector coupled with a temporal tracker).
- **Normalization**: Handled internally by the YOLO backend.

---

## Outputs

- **Bounding Boxes**: Returns coordinates `(x1, y1, x2, y2)`, track IDs, class probability, and bounding box center `(cx, cy)`.
- **Pose Keypoints**: Returns 17 joint coordinates and confidence scores matching the standard COCO layout:
  - `0`: nose, `1`: left_eye, `2`: right_eye, `3`: left_ear, `4`: right_ear, `5`: left_shoulder, `6`: right_shoulder, `7`: left_elbow, `8`: right_elbow, `9`: left_wrist, `10`: right_wrist, `11`: left_hip, `12`: right_hip, `13`: left_knee, `14`: right_knee, `15`: left_ankle, `16`: right_ankle.
- **Conversion Scheme**: Player outputs are mapped into custom schemas [TrackedPlayer](file:///C:/Users/mbonatte/Documents/Coding/tennis/player.py#L12) and [HybridTrackedPlayer](file:///C:/Users/mbonatte/Documents/Coding/tennis/player.py#L27).

---

## Local Enhancements

1. **Zoomed ROI Recovery Logic**:
   - *Implementation*: [_recover_top_player_if_needed](file:///C:/Users/mbonatte/Documents/Coding/tennis/player.py#L755) and [_recover_bottom_player_if_needed](file:///C:/Users/mbonatte/Documents/Coding/tennis/player.py#L849) in [player.py](file:///C:/Users/mbonatte/Documents/Coding/tennis/player.py).
   - *Purpose*: Prevents player dropouts when a player moves fast or gets occluded. If the primary tracker loses a player, the pipeline crops a zoomed region (padded up to `4x` the player's last known width and `2x` height) around their last position. It resizes the crop (`2x` bilinear scaling) and runs a duplicate YOLO model predicting at a lower confidence threshold (`0.15`). If found, it recovers their coordinates.
2. **Hybrid Box and Pose Tracking Matching**:
   - *Implementation*: [_match_pose_to_player](file:///C:/Users/mbonatte/Documents/Coding/tennis/player.py#L1006) in [player.py](file:///C:/Users/mbonatte/Documents/Coding/tennis/player.py).
   - *Purpose*: YOLO-Pose is less robust than bounding box detectors. The pipeline tracks bounding boxes as the primary anchor, runs pose detection independently, and matches poses to tracked player boxes using a combined IoU (0.7 weight) and center distance (0.3 weight) metric. Players are never dropped if pose estimation fails; they simply fall back to box-only tracking.
3. **Tracking Role Stabilization**:
   - *Implementation*: [stabilize_player_roles](file:///C:/Users/mbonatte/Documents/Coding/tennis/tracking_postprocess.py#L7) in [tracking_postprocess.py](file:///C:/Users/mbonatte/Documents/Coding/tennis/tracking_postprocess.py).
   - *Purpose*: Corrects camera midpoint tracking errors and chunk boundary switches. It applies a second, whole-track continuity filter, matching IDs, enforcing court-side constraints, and holding coordinates for up to 8 missing frames to ensure smooth and deterministic results.

---

## Configuration

- **Primary Box Confidence**: `0.5`
- **Secondary Pose Confidence**: `0.35`
- **Keypoint Confidence Threshold**: `0.3`
- **Minimum Box Area**: `500` pixels.
- **Max Missing Frames (Hold)**: `8` frames.
- **Recovery Zoom Factor**: `2.0`
- **Top Player Recovery Confidence**: `0.15`
- **Max Pose Match Distance**: `120.0` pixels.

---

## Assumptions

- **Opposing Court Sides**: The model assumes one player occupies the top screen half (`top_player`) and the other occupies the bottom half (`bottom_player`).
- **Standard Camera Angle**: Assumes a fixed behind-the-court camera perspective.
- **Singles Play**: The role assignment logic is tailored to singles matches.

---

## Limitations and Failure Modes

- **Spectator and Official Confusion**: Spectators sitting behind the court or line officials standing near the baselines can occasionally be detected as players, causing ID switches.
- **Far-Side Pose Estimation**: Because the `top_player` is far from the camera, they appear small and blurry, which often causes YOLO-Pose keypoints to drop below the `0.3` confidence threshold.
- **Overlapping/Crossing Players**: If players run to the same side of the net (e.g. during a hand-shake or warm-up), the tracking roles can get swapped.

---

## Validation

- **Unit Tests**:
  - `tests/test_player_lifecycle.py`: Validates role assignments, missing-frame holds, and zoomed ROI recovery.
  - `tests/test_tracking_postprocess.py`: Tests role stabilization, screen-side constraints, and ID continuity.
- **How to Run Tests**:
  ```bash
  python -m pytest tests/test_player_lifecycle.py tests/test_tracking_postprocess.py
  ```
