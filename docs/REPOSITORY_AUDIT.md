# Repository audit

Audit date: 2026-07-12. Git history through `44f8dbd` and the complete working tree were inspected before implementation.

## Original Python inventory

| File | Purpose | Original pipeline use | Finding |
|---|---|---|---|
| `main.py` | CLI orchestration, video I/O, caching, scene/point splitting, drawing | Primary entry point | Active but monolithic; `read_video()` and `analyze_video()` retained every decoded frame. Replaced as the public entry point by `tennis_analyzer`; retained temporarily for heuristics/plot adapter. |
| `analysis.py` | scene cuts, projection, shot/bounce refinement, speed/player statistics, overlays, JSON | Imported by `main.py` | Active, useful heuristic code; statistics are not validated ground truth. |
| `ball.py` | TrackNet loading, batched inference, interpolation, trail rendering | Active | Used a hardcoded `weights/tracknet_model.pt`, chose device globally, printed to stdout. Now accepts explicit model/device and is called in bounded chunks. |
| `bounce_detector.py` | CatBoost bounce features/inference | Active | Requires `bounce_model.cbm`; operates on coordinate arrays, not images. |
| `bounce_detection.py` | older CatBoost training/experimentation script | Not imported | Obsolete/duplicate; removed after import audit. Git history retains it. |
| `court.py` | court inference, keypoints, minimap | Active | Had a relative default model path; now accepts explicit device/path. Some drawing helpers assume valid projected points. |
| `court_detection_net.py` | court network preprocessing/inference | Active via `court.py` | Depends on root `tracknet.py`, `postprocess.py`, `homography.py`; model load occurs per adapter invocation. |
| `court_reference.py` | canonical court geometry/image | Active via analysis/court | Contains a manual `__main__` image generator; generated PNG is not source. |
| `homography.py` | fit/select court transform | Active | Uses SciPy and OpenCV. |
| `postprocess.py` | line detection and keypoint refinement | Active | Uses SymPy and SciPy. |
| `player.py` | YOLO box, pose, hybrid tracking/drawing | Active when selected | Hardcoded default weight names differ from the actual local `yolo26n*` names; production constructs it with explicit paths. |
| `tracking_postprocess.py` | stabilize top/bottom player roles | Active in legacy pipeline | Useful, but the new chunk adapter currently leaves cross-chunk role stabilization to the tracker. |
| `tracknet.py` | court-keypoint neural architecture | Active via `court_detection_net.py` | Name is confusing because it is not the ball TrackNet. |
| `BallTrack/model.py` | ball TrackNet architecture | Active via `ball.py` | Third-party-derived code in submodule. |
| `ball.py` adapter functions | ball preprocessing/inference/postprocessing | Active | Application-owned adapter batches frames and uses the upstream submodule's neural architecture. |

## Data flow and memory

The old default flow was `main.py -> read_video -> player/ball/court -> analysis -> drawing -> save_video`. It held the source frames and multiple full copied annotation lists concurrently. A long 4K upload could exhaust memory. Point splitting had a separate streaming implementation, but normal analysis did not use it.

The new service iterates `ANALYSIS_CHUNK_FRAMES` images (default 256) during detection, keeps only compact tracks/homographies/events, releases each image chunk, then decodes again for a streaming render. TrackNet needs three adjacent frames and batches of eight inside each chunk. Chunk boundaries can lose up to two frames of temporal context and are documented as a current accuracy limitation. Per-frame metadata still grows linearly, at far lower cost than decoded images.

## Models and licensing

Six local weights were present under ignored `weights/`; none were tracked. The active full pipeline needs ball, bounce, court, player, and pose weights. `keypoints_model.pth` is not imported anywhere. Exact observed sizes and hashes are in `models/README.md`.

No source URL, license, training provenance, or redistribution grant exists in the repository for any weight. They must not be committed or redistributed until reviewed. The 42–95 MB files would need an artifact store or Git LFS if redistribution becomes lawful. Startup is feature-aware: model-free scene cuts/frame numbering work; selecting a model feature reports exactly which file is absent.

## BallTrack submodule

`.gitmodules` points to `https://github.com/yastrebksv/TrackNet.git`. The original gitlink referenced a local-only commit (`e4a6ff6`) that was not fetchable from the configured remote, so a clean GitHub Actions checkout could never build. The gitlink is now pinned to the fetchable upstream commit `730ea17`; application-specific inference/batching lives in `ball.py`, while the upstream neural architecture remains in the submodule to preserve provenance. No license file exists at that upstream commit, so redistribution still requires manual review. Docker includes the cleanly checked-out submodule; local source builds must initialize it, while the VPS pulls the already-built image and needs no submodule or repository.

## Dependencies/imports

Production web: FastAPI, Starlette/Jinja2, python-multipart, SQLAlchemy 2, Alembic, psycopg, Redis, RQ, Pydantic Settings, Uvicorn. Video/ML: FFmpeg/FFprobe, OpenCV, NumPy, PyTorch, torchvision, Ultralytics, CatBoost, pandas, SciPy, SymPy, Matplotlib, tqdm. Testing: Pytest, HTTPX, Ruff. `pickle` was used only by the legacy cache; loading a tampered cache could execute code, so the production job pipeline does not use it.

## Generated/untracked material

The working tree contained `input.mp4` (11.2 MB), `input_many_scenes.mp4` (394 MB), several output MP4s, `analysis_stats.json`, a plot PNG, `.cache/`, `__pycache__/`, `court_configurations/`, `point_analysis/` clips/stats/plots, six weights, and `.venv/`. None was tracked. `.gitignore` and `.dockerignore` now cover uploads, outputs, point clips, plots, caches, logs, DB/Redis/Postgres data, secrets, environments, editor/OS files, and model formats. The Git object database was small (~266 KiB loose objects); no large tracked artifact needed index removal.

## Hardcoded/OS assumptions and defects

- Legacy defaults assumed repository-root `input.mp4`, `output.mp4`, `.cache`, `weights/`, and named statistics files.
- Legacy README commands were PowerShell-specific. Application subprocesses now use argument arrays and `Path`; no shell interpolation or Windows-only command remains in production.
- OpenCV wrote `mp4v`, which is not consistently browser-compatible and dropped audio. The new finalizer uses H.264/yuv420p, AAC when present, `faststart`, and atomic rename.
- Original filenames were directly supplied to CLI paths. Web storage now uses a UUID directory plus `source.<allowed extension>`.
- The old pipeline used `print`, global device/thread configuration, unbounded caches, and non-atomic output.
- Model construction repeats at each detection chunk for ball/court, trading speed for bounded memory and minimal changes to proven model code. A persistent model-session adapter is a priority optimization.
- Scene splitting equated hard cuts with points; this is experimental and will misclassify replays/camera changes.
- In/out, shot, bounce recovery, speeds, and distances are monocular heuristics without calibration validation.
- No original automated tests, dependency lock, application server, database, worker, migrations, upload safety, access control, or deployment definition existed.

`REFERENCE_COMPARISON.md` was removed because the repository-specific comparison notes were superseded by this audit, `ANALYSIS_OPTIONS.md`, and the limitations in the README. No active code or unique operational instruction referenced it.
