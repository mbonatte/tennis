# Ball Tracking Model Card

## Model Identity

- **Feature Name**: Ball Tracking
- **Checkpoint Filename**: `tracknet_model.pt`
- **Framework and Model Class**: PyTorch, [BallTrackerNet](file:///C:/Users/mbonatte/Documents/Coding/tennis/BallTrack/model.py#L16)
- **Model Architecture**: TrackNet-style Fully Convolutional Network (VGG-based encoder-decoder architecture with 18 ConvBlocks, downsampling via MaxPool, and upsampling via Upsample).
- **Upstream Repository/Publication**: Derived from the original TrackNet implementation (TrackNet: A Deep Learning Network for Tracking High-speed Small Objects, Huang et al.).
- **Upstream Version/Commit**: Unknown.
- **Training Dataset**: Unknown.
- **Model License**: Unknown.
- **Weight License**: Unknown (Manual verification required; weights are not distributed).
- **Checksum of Supported Checkpoint**: `c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076`

---

## Purpose

- **Predictions**: Predicts the 2D coordinate $(x, y)$ of the tennis ball in screen coordinates.
- **Importance**: The ball trajectory is the foundational signal for detecting bounce events, shot contacts, in/out calls, ball speed estimates, shot ownership, and scorecard logic.
- **Dependencies**: The downstream [CatBoost Bounce Model](BOUNCE_DETECTION.md) and event heuristics directly depend on this model's outputs.
- **Model-Free Operation**: The system cannot run ball-based event analysis without this model, although scene-cut detection and basic UI frame-number rendering can be executed.

---

## Inputs

- **Tensor Format**: PyTorch FloatTensor.
- **Dimensions**: `(batch_size, 9, 360, 640)` (representing batch, color channels, height, and width).
- **Frame Count / Context**: 3 consecutive BGR frames (current, previous, and pre-previous frames concatenated along the channel dimension, resulting in $3 \text{ frames} \times 3 \text{ BGR channels} = 9$ channels).
- **Resolution**: `640x360` pixels.
- **Color Format**: BGR.
- **Normalization**: Pixel values are divided by `255.0` to map them into the range `[0.0, 1.0]`.

---

## Outputs

- **Tensor Format**: FloatTensor of shape `(batch_size, 256, 230400)` (where `230400 = 360 * 640`).
- **Coordinate System**: Screen space coordinates.
- **Confidence Values**: Implicitly represented by the softmax logit scores across the 256 channels for each pixel.
- **Class Definitions**:
  - The model treats the output heatmap as a classification problem with 256 intensity classes (0 to 255).
  - The `argmax(dim=1)` along the 256 classes yields the intensity value `[0, 255]` for each of the $360 \times 640$ pixels.
- **Conversion Scheme**:
  - The intensity map is multiplied by `255` and cast to `np.uint8`.
  - **The Inversion Ring Behavior**: Because of uint8 arithmetic, multiplying by `255` inverts the intensities: a high probability center value of `255` maps to `255 * 255 = 65025 % 256 = 1`, and a margin value of `1` maps to `1 * 255 % 256 = 255`. This transforms the solid circle heatmap into a high-contrast binary ring.
  - Thresholding is applied at `127`, and `cv2.HoughCircles` detects circles on this donut contour.
  - If a single circle is found, its center coordinates are scaled back to the source video dimensions. If zero or multiple circles are found, it is represented as a missing coordinates tuple: `(None, None)`.

---

## Local Enhancements

1. **Temporal Context Buffer**:
   - *Implementation*: [BallTracker](file:///C:/Users/mbonatte/Documents/Coding/tennis/ball.py#L183) and [TemporalContextBuffer](file:///C:/Users/mbonatte/Documents/Coding/tennis/tennis_analyzer/pipeline/temporal.py).
   - *Purpose*: Prepends the last 2 frames of the previous chunk to the current chunk before inference. This prevents missing ball predictions at chunk boundaries and guarantees that chunked video processing produces results identical to unchunked processing.
2. **Abrupt-Jump Removal**:
   - *Implementation*: [remove_abrupt_jumps](file:///C:/Users/mbonatte/Documents/Coding/tennis/tennis_analyzer/pipeline/ball_track.py#L23).
   - *Purpose*: Eliminates single-frame outliers where the ball tracking leaps to a different white object on the screen. The displacement threshold adapts dynamically to the local median speed so fast ball paths are not incorrectly dropped.
3. **Outlier Filtering**:
   - *Implementation*: [remove_outliers](file:///C:/Users/mbonatte/Documents/Coding/tennis/tennis_analyzer/pipeline/ball_track.py#L60).
   - *Purpose*: Drops coordinate tracks that exceed realistic movement thresholds.
4. **Trajectory Splitting and Interpolation**:
   - *Implementation*: [split_track](file:///C:/Users/mbonatte/Documents/Coding/tennis/tennis_analyzer/pipeline/ball_track.py#L73) and [interpolation](file:///C:/Users/mbonatte/Documents/Coding/tennis/tennis_analyzer/pipeline/ball_track.py#L93).
   - *Purpose*: Splits tracks when there are large gaps ($>= 4$ frames) or distances ($> 80$ pixels). Within valid tracks, missing values are filled in using linear interpolation.

---

## Configuration

- **Confidence Threshold**: Fixed threshold of `127` in the inverted heatmap contour extraction.
- **Batch Size**: Configurable via `ANALYSIS_BALL_BATCH_SIZE` (defaults to `8`).
- **Temporal Window**: Fixed at 3 frames.
- **Device Support**: Run on CPU or GPU/CUDA (configured via the `DEVICE` environment variable). GPU utilizes Automatic Mixed Precision (AMP).

---

## Assumptions

- **Fixed or Stable Camera**: The ball tracker assumes a relative behind-the-court or high-angle perspective. Panning or zooming does not prevent tracking but can cause displacement spikes.
- **Visibility**: The ball must be visible in at least a few consecutive frames to form a track.
- **Single Ball**: The model assumes there is only one ball in active play.
- **Resolution**: Minimum resolution of `1280x720` is recommended for high accuracy, though it is downsampled to `640x360` internally.

---

## Limitations and Failure Modes

- **Motion Blur**: When the ball travels extremely fast (e.g. during a hard serve), it can stretch into a blur, causing the model to miss detections.
- **White Objects**: TrackNet is prone to latching onto other fast-moving white items like sneakers, racket margins, court lines, and spectators' white clothes.
- **Net Occlusion**: When the ball passes behind the net tape, it may be lost for 1-3 frames.
- **Multi-Ball Scenarios**: During warm-ups or if multiple balls are on court, the model will arbitrary jump between them or fail to predict a valid coordinate.

---

## Validation

- **Unit Tests**:
  - `test_ball_tracker_preserves_temporal_context_and_result_length`: Verifies context buffering over chunk boundaries.
  - `test_global_postprocessing_interpolates_across_former_chunk_boundary`: Verifies interpolation works across chunk seams.
  - `test_global_postprocessing_removes_an_isolated_ball_jump_and_interpolates_it`: Verifies single-frame outlier removal.
  - `test_global_postprocessing_keeps_consistently_fast_ball_motion`: Ensures high-speed trajectories are preserved.
- **How to Run Tests**:
  ```bash
  python -m pytest tests/test_ball_tracking.py
  ```
