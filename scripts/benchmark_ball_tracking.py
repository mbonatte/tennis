"""Profile real TrackNet ball tracking across safe chunk and batch sizes."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ball import BallTracker
from scripts.benchmark_pipeline import sample_peak_rss
from tennis_analyzer.pipeline.ball_track import postprocess_ball_track
from tennis_analyzer.pipeline.chunks import iter_frame_chunks
from tennis_analyzer.video import probe_video


@dataclass
class BallBenchmarkResult:
    chunk_size: int
    batch_size: int
    frame_count: int
    runtime_seconds: float
    peak_rss_mb: float
    decode_seconds: float
    continuity_seconds: float
    resize_seconds: float
    input_seconds: float
    transfer_seconds: float
    inference_seconds: float
    prediction_postprocess_seconds: float
    frames_per_second: float


def benchmark_ball_tracking(
    video: Path, model: Path, device: str, chunk_size: int, batch_size: int, max_frames: int | None = None
) -> BallBenchmarkResult:
    """Run the production ball stage once and return independent timing totals."""
    tracker = BallTracker.from_checkpoint(model, device, batch_size=batch_size, use_amp=device.startswith("cuda"))
    raw_track = []
    decode_seconds = 0.0

    def record_decode(_frames: int, elapsed: float) -> None:
        nonlocal decode_seconds
        decode_seconds += elapsed

    started = time.perf_counter()
    try:
        with sample_peak_rss() as peak:
            for chunk in iter_frame_chunks(video, chunk_size, on_decode=record_decode):
                remaining = None if max_frames is None else max_frames - len(raw_track)
                if remaining is not None and remaining <= 0:
                    break
                raw_track.extend(tracker.process_chunk(chunk.frames if remaining is None else chunk.frames[:remaining]))
            continuity_started = time.perf_counter()
            final_track = postprocess_ball_track(raw_track)
            continuity_seconds = time.perf_counter() - continuity_started
    finally:
        timing = tracker.timing_fields()
        tracker.close()
    if len(final_track) != len(raw_track):
        raise RuntimeError("ball tracker did not return one result per decoded frame")
    runtime = time.perf_counter() - started
    return BallBenchmarkResult(
        chunk_size=chunk_size,
        batch_size=batch_size,
        frame_count=len(final_track),
        runtime_seconds=runtime,
        peak_rss_mb=peak[0],
        decode_seconds=decode_seconds,
        continuity_seconds=continuity_seconds,
        resize_seconds=float(timing["ball_resize_seconds"]),
        input_seconds=float(timing["ball_input_seconds"]),
        transfer_seconds=float(timing["ball_transfer_seconds"]),
        inference_seconds=float(timing["ball_inference_seconds"]),
        prediction_postprocess_seconds=float(timing["ball_postprocess_seconds"]),
        frames_per_second=len(final_track) / runtime if runtime else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the production TrackNet ball stage")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/tracknet_model.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--max-frames", type=int, help="Profile a deterministic prefix for a quick smoke benchmark")
    parser.add_argument("--output", type=Path, default=Path("ball-benchmark-results.json"))
    args = parser.parse_args()
    if not args.input.is_file() or not args.model.is_file():
        parser.error("--input and --model must name existing files")
    if any(size <= 0 for size in [*args.chunk_sizes, *args.batch_sizes]) or args.max_frames == 0:
        parser.error("chunk and batch sizes must be positive")
    results = [
        benchmark_ball_tracking(args.input, args.model, args.device, chunk_size, batch_size, args.max_frames)
        for chunk_size in args.chunk_sizes
        for batch_size in args.batch_sizes
        if batch_size <= chunk_size
    ]
    metadata = probe_video(args.input)
    payload = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
        },
        "video": {"path": str(args.input), "width": metadata.width, "height": metadata.height, "fps": metadata.fps},
        "results": [asdict(result) for result in results],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
