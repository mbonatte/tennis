from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

from tennis_analyzer.errors import InvalidVideoError, VideoProcessingError
from tennis_analyzer.schemas import VideoMetadata

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ALLOWED_CODECS = {"h264", "hevc", "mpeg4", "vp8", "vp9", "av1", "mjpeg"}


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise VideoProcessingError("Video inspection or conversion failed") from exc


def probe_video(path: Path, *, timeout: int = 30, allowed_codecs: set[str] | None = None) -> VideoMetadata:
    if not path.is_file() or path.stat().st_size == 0:
        raise InvalidVideoError("The uploaded file is empty or missing")
    command = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    try:
        data: dict[str, Any] = json.loads(_run(command, timeout).stdout)
    except (json.JSONDecodeError, VideoProcessingError) as exc:
        raise InvalidVideoError("The uploaded file is not a valid video") from exc
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video or int(video.get("width", 0)) <= 0 or int(video.get("height", 0)) <= 0:
        raise InvalidVideoError("The file does not contain a readable video stream")
    codec = str(video.get("codec_name", "unknown"))
    if codec not in (allowed_codecs or ALLOWED_CODECS):
        raise InvalidVideoError(f"Unsupported video codec: {codec}")
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
    numerator, denominator = (float(part) for part in rate.split("/", 1))
    fps = numerator / denominator if denominator else 0.0
    if not math.isfinite(fps) or fps <= 0:
        raise InvalidVideoError("The video frame rate could not be determined")
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    frames = video.get("nb_frames")
    return VideoMetadata(
        duration_seconds=duration,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        frame_count=int(frames) if frames and str(frames).isdigit() else None,
        container=str(data.get("format", {}).get("format_name", "unknown")),
        video_codec=codec,
        audio_codec=str(audio.get("codec_name")) if audio else None,
        file_size=path.stat().st_size,
    )


def validate_video(metadata: VideoMetadata, *, max_duration: int, max_width: int, max_height: int) -> None:
    if metadata.duration_seconds <= 0:
        raise InvalidVideoError("The video duration could not be determined")
    if metadata.duration_seconds > max_duration:
        raise InvalidVideoError(f"Video exceeds the {max_duration}-second duration limit")
    if metadata.width > max_width or metadata.height > max_height:
        raise InvalidVideoError(f"Video exceeds the {max_width}x{max_height} resolution limit")


def normalize_video(raw_video: Path, source_video: Path, destination: Path, *, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-shortest",
        str(temporary),
    ]
    try:
        _run(command, timeout)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise VideoProcessingError("Video conversion produced no output")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
