# Court Detection Model Card

## Model Identity

- **Feature Name**: Court Detection
- **Checkpoint Filename**: `tennis_court.pt`
- **Framework and Model Class**: PyTorch, [CourtDetectorNet](file:///C:/Users/mbonatte/Documents/Coding/tennis/court_detection_net.py#L12)
- **Model Architecture**: The model class instantiates the [Tracker](file:///C:/Users/mbonatte/Documents/Coding/tennis/tracknet.py#L16) network with `out_channels=15` (fully convolutional encoder-decoder with ConvBlocks and max pooling/upsampling).
- **Upstream Repository/Publication**: Unknown.
- **Upstream Version/Commit**: Unknown.
- **Training Dataset**: Unknown.
- **Model License**: Unknown.
- **Weight License**: Unknown (Manual verification required; weights are not distributed).
- **Checksum of Supported Checkpoint**: `09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04`

---

## Purpose

- **Predictions**: Predicts heatmaps for 14 essential line intersections (keypoints) on the tennis court.
- **Importance**: Provides the landmark correspondences required to compute the homography matrix. This matrix maps 2D screen pixels to physical coordinates in the canonical court coordinate system.
- **Dependencies**: The court-space projections of the ball and players depend on this stage's computed homography matrices.
- **Model-Free Operation**: Yes. If the user provides a manual four-corner calibration on a representative frame, the pipeline completely bypasses model inference and applies a static homography across all frames.

---

## Inputs

- **Tensor Format**: PyTorch FloatTensor.
- **Dimensions**: `(batch_size, 3, 360, 640)` (where `3` represents BGR channels, and `360x640` is the normalized model input resolution).
- **Temporal Context**: None (single-frame inference).
- **Normalization**: Pixel values are scaled to the range `[0.0, 1.0]` by dividing by `255.0`.

---

## Outputs

- **Tensor Format**: FloatTensor of shape `(batch_size, 15, 360, 640)`. The 15th channel is unused, and the first 14 channels correspond to the heatmaps for the 14 court keypoints.
- **Keypoint Mapping (Index 0 to 13)**:
  - `0`: Top-left doubles corner (intersection of top baseline and left doubles sideline)
  - `1`: Top-right doubles corner (intersection of top baseline and right doubles sideline)
  - `2`: Bottom-left doubles corner (intersection of bottom baseline and left doubles sideline)
  - `3`: Bottom-right doubles corner (intersection of bottom baseline and right doubles sideline)
  - `4`: Top-left singles corner (intersection of top baseline and left singles sideline)
  - `5`: Bottom-left singles corner (intersection of bottom baseline and left singles sideline)
  - `6`: Top-right singles corner (intersection of top baseline and right singles sideline)
  - `7`: Bottom-right singles corner (intersection of bottom baseline and right singles sideline)
  - `8`: Top-left service corner (intersection of top service line and left singles sideline)
  - `9`: Top-right service corner (intersection of top service line and right singles sideline)
  - `10`: Bottom-left service corner (intersection of bottom service line and left singles sideline)
  - `11`: Bottom-right service corner (intersection of bottom service line and right singles sideline)
  - `12`: Top center service T (intersection of top service line and center service line)
  - `13`: Bottom center service T (intersection of bottom service line and center service line)
- **Extraction Scheme**:
  - Heatmap channels are thresholded at `170`.
  - `cv2.HoughCircles` detects circles on each keypoint heatmap. The first circle's center is selected as the keypoint $(x, y)$ coordinate in model resolution and scaled back to the source frame width/height.

---

## Local Enhancements

1. **SymPy Line-Intersection Refinement**:
   - *Implementation*: [refine_kps](file:///C:/Users/mbonatte/Documents/Coding/tennis/postprocess.py#L22) in [postprocess.py](file:///C:/Users/mbonatte/Documents/Coding/tennis/postprocess.py).
   - *Purpose*: Corrects deep learning coordinate drift using traditional image processing. For all keypoints except `8`, `9`, and `12` (which are skipped due to net occlusion and distance-blur), the system crops an `80x80` window. It runs Probabilistic Hough Line Transform, merges lines within 20 pixels, and calculates the mathematical intersection of the remaining two lines using SymPy. If valid, the intersection replaces the model coordinate.
2. **Homography Configuration Search**:
   - *Implementation*: [get_trans_matrix](file:///C:/Users/mbonatte/Documents/Coding/tennis/homography.py#L32) in [homography.py](file:///C:/Users/mbonatte/Documents/Coding/tennis/homography.py).
   - *Purpose*: Evaluates 12 standard court configurations (sets of 4 keypoints). For each config, if all 4 points are detected, it computes the homography, projects the reference court keypoints, and calculates the mean projection distance error against the other detected keypoints. It selects the homography that minimizes this error.
3. **Adaptive Stability Calibration**:
   - *Implementation*: [_sampled_court_detection](file:///C:/Users/mbonatte/Documents/Coding/tennis/tennis_analyzer/pipeline/service.py#L105) in [service.py](file:///C:/Users/mbonatte/Documents/Coding/tennis/tennis_analyzer/pipeline/service.py).
   - *Purpose*: Runs court detection on 9 representative frames and scene cuts first. If the computed homographies vary by `<= 2%` of the image diagonal, it assumes a static camera and reuses the median homography across all frames. If the variance is $> 2\%$, it automatically falls back to full per-frame court model inference.

---

## Configuration

- **Heatmap Threshold**: Fixed at `170`.
- **Hough Circles Parameters**: `dp=1`, `minDist=20`, `param1=50`, `param2=2`, `minRadius=10`, `maxRadius=25`.
- **Refinement Crop Size**: `40` pixels radius (total size `80x80`).
- **Line Merging Tolerance**: `20` pixels.
- **Line Detection Threshold**: Grayscale threshold `155`, Hough threshold `30`, min line length `10`, max line gap `30`.

---

## Assumptions

- **Stable Perspectives**: Assumes a classic television or coaching behind-the-court camera perspective.
- **Court Line Contrast**: Assumes court lines are painted white and stand out clearly against the court surface.
- **Singles Boundaries**: Standard singles court markings are expected.

---

## Limitations and Failure Modes

- **Net Occlusion**: Net tape and net posts can overlap with top-half keypoints (e.g. `8`, `9`, `12`), causing model misdetections.
- **Dynamic Shadows**: Strong shadows from stadium roofs or outdoor lighting can obscure line intersections.
- **Extreme Camera Movement**: Rapid panning or zooming during a rally can result in keypoint tracking loss, causing homography failures.
- **Motion Blur**: Rapid camera panning can blur lines, leading to unsuccessful line refinement.

---

## Validation

- **Unit Tests**:
  - `tests/test_court_tracking.py`: Validates model inference, chunk handling, and keypoint alignment.
  - `tests/test_court_calibration.py`: Validates homography calculations, median selection, and coordinate propagation.
- **How to Run Tests**:
  ```bash
  python -m pytest tests/test_court_tracking.py tests/test_court_calibration.py
  ```
