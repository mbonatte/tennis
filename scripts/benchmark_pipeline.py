from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing
import os
import platform
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from tennis_analyzer.pipeline.chunks import iter_frame_chunks
from tennis_analyzer.video import probe_video

STAGE_COUNT = 3


def _rss_mb() -> float:
    if os.name == "nt":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
        get_memory.restype = ctypes.c_int
        handle = get_process()
        if not get_memory(handle, ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return counters.WorkingSetSize / 1024 / 1024
    statm = Path("/proc/self/statm")
    if statm.is_file():
        resident_pages = int(statm.read_text(encoding="ascii").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024
    return 0.0


@contextmanager
def sample_peak_rss(interval_seconds: float = 0.01):
    peak = [_rss_mb()]
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(interval_seconds):
            peak[0] = max(peak[0], _rss_mb())

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield peak
    finally:
        stop.set()
        thread.join()
        peak[0] = max(peak[0], _rss_mb())


class FakeModel:
    """Deterministic CPU work and allocation; not a neural-model benchmark."""

    def __init__(self, allocation_mb: int):
        self.allocation = np.ones((allocation_mb * 1024 * 1024 // 4,), dtype=np.float32)

    def infer(self, frames: list[np.ndarray]) -> float:
        return sum(float(frame[::16, ::16].mean()) for frame in frames)


@dataclass
class BenchmarkResult:
    scenario: str
    chunk_size: int
    execution_mode: str
    model_load_count: int
    runtime_seconds: float
    peak_rss_mb: float
    frames_per_second: float
    model_load_seconds: float
    inference_seconds: float
    video_decode_seconds: float
    rendering_seconds: float
    frame_count: int
    width: int
    height: int
    device: str = "cpu"
    benchmark_kind: str = "deterministic fake-model pipeline overhead"


def _load_model(allocation_mb: int, metrics: dict[str, float | int]) -> FakeModel:
    started = time.perf_counter()
    model = FakeModel(allocation_mb)
    metrics["model_load_seconds"] += time.perf_counter() - started
    metrics["model_load_count"] += 1
    return model


def _infer(model: FakeModel, frames: list[np.ndarray], metrics: dict[str, float | int]) -> None:
    started = time.perf_counter()
    model.infer(frames)
    metrics["inference_seconds"] += time.perf_counter() - started


def _render_pass(video: Path, chunk_size: int) -> float:
    started = time.perf_counter()
    for chunk in iter_frame_chunks(video, chunk_size):
        for frame in chunk.frames:
            rendered = frame.copy()
            cv2.circle(rendered, (10, 10), 3, (0, 255, 0), -1)
    return time.perf_counter() - started


def benchmark_scenario(video: Path, scenario: str, chunk_size: int, fake_model_mb: int = 8) -> BenchmarkResult:
    metadata = probe_video(video)
    metrics: dict[str, float | int] = {
        "model_load_count": 0,
        "model_load_seconds": 0.0,
        "inference_seconds": 0.0,
    }
    started = time.perf_counter()
    with sample_peak_rss() as peak:
        if scenario == "reload_per_chunk_baseline":
            for chunk in iter_frame_chunks(video, chunk_size):
                for _ in range(STAGE_COUNT):
                    model = _load_model(fake_model_mb, metrics)
                    _infer(model, chunk.frames, metrics)
                    del model
        elif scenario == "persistent_single_pass":
            models = [_load_model(fake_model_mb, metrics) for _ in range(STAGE_COUNT)]
            for chunk in iter_frame_chunks(video, chunk_size):
                for model in models:
                    _infer(model, chunk.frames, metrics)
            del models
        elif scenario == "low_memory_multi_pass":
            for _ in range(STAGE_COUNT):
                model = _load_model(fake_model_mb, metrics)
                for chunk in iter_frame_chunks(video, chunk_size):
                    _infer(model, chunk.frames, metrics)
                del model
        else:
            raise ValueError(f"unknown benchmark scenario: {scenario}")
        rendering_seconds = _render_pass(video, chunk_size)
    runtime = time.perf_counter() - started

    decode_started = time.perf_counter()
    decoded = sum(len(chunk.frames) for chunk in iter_frame_chunks(video, chunk_size))
    decode_seconds = time.perf_counter() - decode_started
    return BenchmarkResult(
        scenario=scenario,
        chunk_size=chunk_size,
        execution_mode="low_memory" if scenario == "low_memory_multi_pass" else "comparison",
        model_load_count=int(metrics["model_load_count"]),
        runtime_seconds=runtime,
        peak_rss_mb=peak[0],
        frames_per_second=decoded / runtime,
        model_load_seconds=float(metrics["model_load_seconds"]),
        inference_seconds=float(metrics["inference_seconds"]),
        video_decode_seconds=decode_seconds,
        rendering_seconds=rendering_seconds,
        frame_count=decoded,
        width=metadata.width,
        height=metadata.height,
    )


def _benchmark_child(connection, video: str, scenario: str, chunk_size: int, fake_model_mb: int) -> None:
    try:
        connection.send(asdict(benchmark_scenario(Path(video), scenario, chunk_size, fake_model_mb)))
    except BaseException as exc:
        connection.send(exc)
    finally:
        connection.close()


def benchmark_isolated(video: Path, scenario: str, chunk_size: int, fake_model_mb: int = 8) -> BenchmarkResult:
    """Run one scenario in a fresh process so peak RSS comparisons are fair."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_benchmark_child,
        args=(child, str(video), scenario, chunk_size, fake_model_mb),
    )
    process.start()
    child.close()
    payload = parent.recv()
    process.join()
    if isinstance(payload, BaseException):
        raise payload
    if process.exitcode != 0:
        raise RuntimeError(f"Benchmark child exited with code {process.exitcode}")
    return BenchmarkResult(**payload)


def _create_video(path: Path, seconds: int, fps: int, width: int, height: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create synthetic benchmark video")
    try:
        for index in range(seconds * fps):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 20 + index % max(1, width - 40)
            cv2.circle(frame, (x, height // 2), 8, (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bounded pipeline overhead with deterministic fake models")
    parser.add_argument("--input", type=Path, help="Existing input video; otherwise generate one")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--fake-model-mb", type=int, default=8)
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0 or args.fake_model_mb <= 0:
        parser.error("video dimensions, duration, FPS, and fake model size must be positive")
    if any(size <= 0 for size in args.chunk_sizes):
        parser.error("chunk sizes must be positive")

    video = args.input or args.output.with_name("benchmark-input.mp4")
    if args.input is None:
        _create_video(video, args.seconds, args.fps, args.width, args.height)

    scenarios = ["reload_per_chunk_baseline", "persistent_single_pass", "low_memory_multi_pass"]
    results = [
        benchmark_isolated(video, scenario, chunk_size, args.fake_model_mb)
        for chunk_size in args.chunk_sizes
        for scenario in scenarios
    ]
    payload = {
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "warning": "Fake-model pipeline-overhead benchmark; do not interpret as neural inference performance.",
        "results": [asdict(result) for result in results],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
