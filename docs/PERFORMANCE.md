# Pipeline performance and constrained-host operation

## Execution model

`ANALYSIS_EXECUTION_MODE=low_memory` is the only production mode. The worker first scans the source to establish the exact readable frame count and optional scene cuts, then performs one bounded pass for each enabled model family, followed by statistics/events and a streaming render pass. Only one model stage is live at a time. Compact coordinates, homographies, keypoints, and player records remain in memory; decoded source and rendered frames do not.

Ball, court, and player/pose owners are created once per job stage, reused for every chunk, and closed in `finally` blocks. Sequential RQ jobs therefore do not share mutable tracker state. Player/ByteTrack state persists across every chunk in one job and resets between jobs. Court inference is stateless; its model persists, but each frame's homography is independent. No additional court smoothing is applied, so ordinary frame-to-frame model jitter remains a known limitation rather than a chunk-boundary reset.

TrackNet consumes three source frames. Its adapter retains the final two source frames, prepends them to the next chunk, and trims exactly the two overlap outputs. Source chunks themselves remain non-overlapping. The final coordinate list has one zero-based entry per decoded source frame; the first two video frames use `(None, None)` because no complete temporal window exists. Outlier filtering, splitting, interpolation, and non-finite normalization run once after all raw chunks are concatenated.

PyTorch ball and court inference uses `torch.inference_mode()` and `.eval()`. Player and pose entry points are also inference-only; Ultralytics receives the explicit configured device. CUDA remains supported through `DEVICE=cuda`, but CPU is the deployment default. Mixed precision is not enabled automatically.

## Recommended settings

For 2 vCPU and 8 GB RAM:

```env
DEVICE=cpu
WORKER_CONCURRENCY=1
ANALYSIS_EXECUTION_MODE=low_memory
ANALYSIS_CHUNK_FRAMES=128
ANALYSIS_BALL_BATCH_SIZE=4
MAX_VIDEO_WIDTH=1920
MAX_VIDEO_HEIGHT=1080
JOB_TIMEOUT_SECONDS=86400
```

If memory pressure occurs, reduce `ANALYSIS_CHUNK_FRAMES` to 64 and `ANALYSIS_BALL_BATCH_SIZE` to 1 or 2. Disable pose and experimental statistics before reducing below 64; repeated video decoding becomes a larger fraction of runtime at very small chunks. Never increase `WORKER_CONCURRENCY` on an 8 GB host without measuring simultaneous peak RSS.

For larger CPU systems, test 128 and 256 frames and ball batches 4 and 8. Larger values are not automatically faster because model inputs are resized and decoding/post-processing can dominate. `ball_batch_size` must not exceed the chunk size. GPU deployments should start at chunk 128/batch 4, observe VRAM, and increase only after a representative run.

## Weight handling

Only selected stages are validated and loaded. Required files are listed in [`models/README.md`](../models/README.md). Custom PyTorch checkpoints use `weights_only=True`, explicit `map_location`, `.eval()`, common `state_dict`/`model_state_dict` wrappers, and consistent DataParallel `module.` prefix removal. Incompatible weights fail with a stage-specific safe error; arbitrary model downloads are not performed.

Hybrid player/pose analysis loads the box and pose checkpoints once. Its independent recovery predictor is cloned from the already-loaded box model so ROI recovery cannot disturb ByteTrack state without rereading the checkpoint.

## Benchmarks

The benchmark below was collected on Windows 11/Python 3.12 using a deterministic 300-frame 640x360 synthetic video and three 16 MiB fake models. Each scenario ran in a fresh process. It measures orchestration, decoding, allocation, and rendering overhead—not neural-network inference performance.

| Scenario | Chunk | Mode | Model loads | Runtime (s) | Peak RSS (MiB) | FPS |
|---|---:|---|---:|---:|---:|---:|
| Reload per chunk baseline | 64 | comparison | 15 | 0.268 | 157.1 | 1118.1 |
| Persistent models | 64 | comparison | 3 | 0.228 | 188.0 | 1318.0 |
| Low-memory multi-pass | 64 | low_memory | 3 | 0.430 | 163.6 | 698.2 |
| Reload per chunk baseline | 128 | comparison | 9 | 0.242 | 233.0 | 1237.3 |
| Persistent models | 128 | comparison | 3 | 0.224 | 266.2 | 1337.3 |
| Low-memory multi-pass | 128 | low_memory | 3 | 0.413 | 251.2 | 727.1 |
| Reload per chunk baseline | 256 | comparison | 6 | 0.238 | 264.1 | 1260.0 |
| Persistent models | 256 | comparison | 3 | 0.227 | 301.4 | 1322.8 |
| Low-memory multi-pass | 256 | low_memory | 3 | 0.418 | 283.6 | 717.9 |

The comparison demonstrates load-count and memory-lifecycle behavior. Low-memory mode deliberately trades extra decode passes for lower model residency. At 128 frames it used about 15 MiB less peak RSS than the all-model persistent comparator; real models have much larger and less uniform allocations, so this fake result must not be extrapolated as a production memory claim.

A real-weight smoke run used the verified local weights, a 10-frame 320x240 H.264 input, `--full`, chunk 4, ball batch 2, `DEVICE=cpu`, and Docker limits of 2 CPUs/8 GB:

| Scenario | Chunk | Mode | Checkpoint reads | Runtime (s) | Memory | FPS |
|---|---:|---|---:|---:|---:|---:|
| Real models, all stages | 4 | low_memory | 6 | 116.1 | 653.7 MiB highest observed sample | 0.086 |

The output was H.264, 320x240, 10 FPS, 10 frames, and 1.000 seconds. Memory was sampled rather than continuously profiled, so 653.7 MiB is an observed value, not a guaranteed peak. This run preceded the final recovery-predictor clone and therefore read the player box checkpoint twice (six reads total). The current implementation reads five checkpoints, validated separately with real Ultralytics models, but its full runtime was not remeasured. This tiny uniform video is a correctness smoke test, not a throughput estimate for match footage.

Run the reproducible fake-model comparison:

```bash
python -m scripts.benchmark_pipeline --seconds 10 --fps 30 --width 640 --height 360 --chunk-sizes 64 128 256 --fake-model-mb 16 --output benchmark-results.json
```

Run a one-minute synthetic CPU overhead benchmark:

```bash
python -m scripts.benchmark_pipeline --seconds 60 --fps 30 --width 1280 --height 720 --chunk-sizes 64 128 256 --fake-model-mb 16 --output benchmark-one-minute.json
```

Run a real-weight smoke test after placing weights in `models/`:

```bash
RUN_MODEL_INTEGRATION=1 MODEL_ROOT=models DEVICE=cpu python -m pytest -q -m integration tests/test_integration_models.py
```

## Monitoring and verification

```bash
docker compose stats worker postgres redis
docker compose logs -f worker
docker compose logs worker | grep "Loaded job-scoped model stage"
```

For one full job, the last command should show one load event for each enabled stage. Each event includes `model_stage` and `model_load_seconds` in structured logs. The regression test is:

```bash
python -m pytest -q tests/test_pipeline_stages.py::test_low_memory_pipeline_loads_each_enabled_stage_once_and_releases_between_stages
```

## Troubleshooting

- **Slow processing:** CPU full utilization is expected. Disable pose, player recovery-intensive analysis, statistics, or unneeded overlays. Check that logs show one model load per enabled stage rather than one per chunk. Do not judge completion from an old single-pass percentage range; progress is now weighted across enabled passes.
- **Out of memory:** keep one worker, set chunk 64 and ball batch 1 or 2, enforce the 1080p upload ceiling, and inspect `docker stats`. Increasing swap may prevent a kill but can make inference dramatically slower.
- **Worker timeout:** the default is 24 hours. Raise `JOB_TIMEOUT_SECONDS` only after measuring representative input; it remains a hard bound for hangs.
- **Missing or incompatible model:** compare `MODEL_ROOT` contents with `models/README.md`. The worker validates selected files before decoding and fails the owning stage if checkpoint contents are incompatible.
- **Partial/corrupt output:** the pipeline refuses changed decoded dimensions, declared/read frame-count mismatch, empty encoder output, and final dimension/frame-count/duration mismatch. Temporary videos and JSON are removed on handled failures.

## Known limitations

- Multi-pass low-memory execution decodes the source once per enabled model family plus scan and render passes.
- Variable-frame-rate inputs are rendered at FFprobe's average FPS; exact source presentation timestamps are not retained by OpenCV.
- Court homographies are not globally smoothed and may jitter for difficult frames.
- Cancellation is checked between chunks, not inside a single model call or FFmpeg encode.
- Compact intermediate results remain in RAM. They are substantially smaller than frames; no pickle or untrusted intermediate deserialization is used.
- Model provenance and redistribution rights remain unresolved as documented in `models/README.md`.
