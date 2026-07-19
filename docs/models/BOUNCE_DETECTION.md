# Bounce Detection Model Card

## Model Identity

- **Feature Name**: Bounce Detection
- **Checkpoint Filename**: `bounce_model.cbm`
- **Framework and Model Class**: CatBoost, `CatBoostRegressor`
- **Model Architecture**: Gradient Boosted Decision Trees (GBDT).
- **Upstream Repository/Publication**: Unknown.
- **Upstream Version/Commit**: Unknown.
- **Training Dataset**: Unknown.
- **Model License**: Unknown.
- **Weight License**: Unknown (Manual verification required; weights are not distributed).
- **Checksum of Supported Checkpoint**: `f525c96b843e47e261a4ea3fbe80f3498980c19821ac41a34b2299a0950ec531`

---

## Purpose

- **Predictions**: Evaluates the ball's 2D coordinate trajectory patterns to predict whether a frame contains a court bounce (racket contact is resolved separately in later stages).
- **Importance**: Identifies where the ball hit the ground, which is essential to determine in/out calls, point boundaries, and rally statistics.
- **Dependencies**: Downstream scoring, in/out classification, and shot-contact heuristics depend directly on these bounce indices.
- **Model-Free Operation**: No. Bounce-related event parsing cannot be run without this model.

---

## Inputs

- **Format**: 12 tabular trajectory features computed from a rolling 5-frame window of ball tracking coordinates ($t-2, t-1, t, t+1, t+2$).
- **Tabular Features**:
  - For each temporal step $i \in \{1, 2\}$ relative to current frame $t$:
    - **X-axis Displacements**: Absolute differences $|x_{t-i} - x_t|$ and $|x_{t+i} - x_t|$.
    - **X-axis Ratios**: Ratios of displacements: $\frac{|x_{t-i} - x_t|}{|x_{t+i} - x_t| + \epsilon}$.
    - **Y-axis Displacements**: Signed differences $(y_{t-i} - y_t)$ and $(y_{t+i} - y_t)$.
    - **Y-axis Ratios**: Ratios of displacements: $\frac{y_{t-i} - y_t}{(y_{t+i} - y_t) + \epsilon}$.
- **Temporal Context**: A window of 2 frames in the past and 2 frames in the future is required for each prediction frame.

---

## Local Enhancements

1. **Trajectory Spline Extrapolation**:
   - *Implementation*: [smooth_predictions](file:///C:/Users/mbonatte/Documents/Coding/tennis/bounce_detector.py#L67) in [bounce_detector.py](file:///C:/Users/mbonatte/Documents/Coding/tennis/bounce_detector.py).
   - *Purpose*: Fills in tracking gaps before computing features. If a ball coordinate is missing but the past 5 frames are valid, it extrapolates coordinates using a local `CubicSpline` for up to 3 consecutive missing frames. Additionally, if the next tracked frame has a distance $> 80$ pixels from the extrapolated point, it is treated as a tracking anomaly and cleared.
2. **Non-Maximum Peak Suppression**:
   - *Implementation*: [postprocess](file:///C:/Users/mbonatte/Documents/Coding/tennis/bounce_detector.py#L111) in [bounce_detector.py](file:///C:/Users/mbonatte/Documents/Coding/tennis/bounce_detector.py).
   - *Purpose*: Prevents multiple bounce detections for a single event. When multiple consecutive frames exceed the bounce threshold, the post-process merges them and selects the frame with the maximum regression score as the exact bounce moment.

---

## Configuration

- **Decision Threshold**: `0.45`
- **Extrapolation Support Window**: `5` frames.
- **Maximum Consecutive Extrapolations**: `3` frames.
- **Coordinate Outlier Distance Limit**: `80` pixels.

---

## Assumptions

- **High Tracking Quality**: Assumes a reasonably complete ball trajectory. If the tracking contains gaps larger than 4 frames, feature engineering fails for those frames.
- **Consistent Frame Rate**: Feature values (specifically speed ratios) assume a stable frame rate.

---

## Limitations and Failure Modes

- **Occlusion Gaps**: If the ball is occluded (e.g., passing behind the net or player), the CatBoost model will miss the bounce due to missing features.
- **High-Velocity Racket Contacts**: Sudden direction changes during racket contact (especially volleys or close player hits) can exhibit features identical to court bounces, resulting in false bounce detections. (These are resolved in downstream heuristics).
- **Camera Vibration**: Quick camera vibrations can alter coordinates, creating false bounce predictions.

---

## Validation

- **Unit Tests**:
  - `tests/test_pipeline_stages.py`: Validates model loading, feature calculation, and prediction formatting.
- **How to Run Tests**:
  ```bash
  python -m pytest tests/test_pipeline_stages.py
  ```
