# Tennis Analyzer

This project analyzes a tennis video with three computer-vision stages:

1. Ball tracking with `tracknet_model.pt`.
2. Court keypoint and homography detection with `tennis_court.pt`.
3. Optional player box and pose tracking with YOLO models.

`main.py` is the best place to restart. It now has an `AnalyzerConfig` section at the top where you can turn stages and drawings on or off.

## Current Default Run

The default run reads `input.mp4`, tracks the ball, tracks the court and players for stats, predicts bounces, draws the red ball trail, draws bounce markers/frame numbers/stats, writes `output.mp4`, and writes `analysis_stats.json`.

```powershell
.\.venv\Scripts\python.exe main.py
```

Useful command-line options:

```powershell
.\.venv\Scripts\python.exe main.py --draw-court
.\.venv\Scripts\python.exe main.py --draw-court --draw-players
.\.venv\Scripts\python.exe main.py --input input.mp4 --output output_debug.mp4
.\.venv\Scripts\python.exe main.py --input input.mp4 --plot-ball-history
.\.venv\Scripts\python.exe main.py --input long_compilation.mp4 --split-points-dir points
.\.venv\Scripts\python.exe main.py --input long_compilation.mp4 --analyze-points-dir point_analysis --draw-court
.\.venv\Scripts\python.exe main.py --input long_compilation.mp4 --analyze-points-dir point_analysis --keep-debug-scenes
.\.venv\Scripts\python.exe main.py --no-players
```

## Important Files

- `main.py`: pipeline entry point and runtime switches.
- `ball.py`: loads the TrackNet model, predicts ball positions, removes outliers, interpolates gaps, and draws the ball trail.
- `bounce_detector.py`: uses the tracked ball x/y positions to predict likely bounce frames.
- `court.py`: wraps court detection and court drawing helpers.
- `court_detection_net.py`: neural-network inference for court keypoints and homography matrices.
- `homography.py`: chooses the best court homography from detected keypoints.
- `postprocess.py`: refines detected court keypoints using local line intersections.
- `player.py`: player bounding-box tracking, pose tracking, and hybrid player annotation.
- `tracking_postprocess.py`: stabilizes top/bottom player roles when detections briefly swap, duplicate, or drift off court.
- `analysis.py`: scene cuts, reference-style shot events, bounce in/out calls, projected speeds, player stats, JSON export, and stats overlay.
- `analysis_stats.json`: latest generated match statistics.
- `.cache/`: cached model predictions for faster repeated runs.
- `REFERENCE_COMPARISON.md`: notes about what the reference projects include and what is still missing here.

## Where To Continue

Start by editing `AnalyzerConfig` in `main.py`.

Useful switches:

- Set `draw_players=True` to render player boxes and poses.
- Set `draw_court_keypoints=True` to render detected court keypoints.
- Set `draw_court_overlay=True` or run `--draw-court` to render the minimap court overlay. The minimap shows the ball in yellow, bounces in orange, top player as blue `T`, and bottom player as green `B`.
- Set `input_path=PROJECT_ROOT / "your_video.mp4"` to analyze another file.
- Run with `--plot-ball-history` to save a debug PNG with ball Y pixel history on the left axis, ball X pixel history on the right axis, red shot markers, and green dashed bounce markers. Single-video runs write `<stats-name>_ball_history.png`; point-analysis runs write per-point plots to `point_analysis/ball_history_plots`.
- Run with `--split-points-dir points` to stream-split a long compilation into one raw MP4 per detected scene/point without running the full analyzer. Files are named with the point number and source frame range.
- Run with `--analyze-points-dir point_analysis` to split a compilation, quickly discard very short scenes, analyze the remaining candidates, and keep only scenes that look like played points. Played point videos are written to `point_analysis/point_videos`, and their JSON files are written to `point_analysis/point_stats`.
- Raw scene clips and rejected clips are temporary by default. Add `--keep-debug-scenes` when you want to inspect `raw_scenes` and `rejected_scenes`.
- Tune the cheap pre-analysis filter with `--min-analyze-frames`; clips shorter than this are rejected before the expensive ball/court/player analysis runs.

Shot frames use the reference repo's sustained vertical trajectory-change heuristic adapted to TrackNet center points. Speed and player statistics are still heuristic estimates from monocular video, so treat them as debug analytics until they are validated against labeled match events.

Bounce calls in `analysis_stats.json` use different rules for serve and rally play: the first bounce after the first shot is checked against the opposite service boxes, while later game bounces are checked against the full singles court.

Short missing ball tracks can happen when the ball is hidden behind a player. The analyzer now rejects pre-shot toss detections and recovers likely hidden bounces by looking for a short occlusion where the ball is falling before the gap, reappears low, and the next shot follows soon after.
