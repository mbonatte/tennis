# Reference Comparison

These notes compare this project with the repositories and article you listed.

## Sources Checked

- https://github.com/abdullahtarek/tennis_analysis
- https://github.com/yastrebksv/TennisProject
- https://github.com/yastrebksv/TennisCourtDetector
- https://github.com/yastrebksv/TrackNet
- https://medium.com/@kosolapov.aetp/tennis-analysis-using-deep-learning-and-machine-learning-a5a74db7e2ee

## Already Present Here

- TrackNet-style ball tracking through `BallTrack/` and `ball.py`.
- Court keypoint detection through `court_detection_net.py`.
- Court keypoint refinement in `postprocess.py`.
- Homography reconstruction in `homography.py`.
- CatBoost bounce prediction in `bounce_detector.py`.
- Player tracking through YOLO box tracking and YOLO pose tracking in `player.py`.

## Added In This Pass

- CLI options in `main.py`.
- Bounce markers on the output video.
- Frame numbers on the output video.
- Minimap plotting for ball position, bounce position, and player foot positions when `--draw-court` is enabled.
- Scene cut detection.
- Cached model predictions in `.cache/`.
- Shot detection from sustained vertical ball trajectory changes, adapted from the reference repo.
- Ball speed, player movement speed, shot counts, stats overlay, and JSON export.

Example:

```powershell
.\.venv\Scripts\python.exe main.py --draw-court --draw-players
```

## Missing Or Not Yet Ported

- Training notebooks and dataset preparation scripts.
- Labeled event validation for shot/bounce/stat accuracy.
- Advanced scene handling beyond hard-cut detection.

## Different Approaches To Review Later

- `abdullahtarek/tennis_analysis` uses YOLO for the tennis ball, while this project uses TrackNet-style heatmap tracking. TrackNet is usually better suited to tiny, blurry ball motion, but YOLO can be simpler to train and debug.
- `abdullahtarek/tennis_analysis` builds player and ball stats from mini-court coordinates. This project now computes similar stats from projected court coordinates, but the event detection and speed filtering are simpler heuristics.
- `yastrebksv/TennisProject` uses Faster R-CNN for person detection and filters players by projecting court masks. This project uses YOLO tracking and a top/bottom split, which is faster and gives track IDs, but the court-mask approach may reject spectators more reliably.
- `yastrebksv/TennisProject` uses scene detection before rendering. This project detects hard cuts, but it does not yet split analysis into independent rallies/scenes.
- The court detector reference project reports the best court keypoint quality when combining keypoint refinement with homography reconstruction. This project already follows that direction.
