from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from analysis import (
    compute_match_stats,
    detect_scene_cuts,
    draw_stats_overlay,
    save_stats,
)
from ball import draw_track, track_ball
from bounce_detector import BounceDetector
from court import draw_court, draw_keypoints, track_court
from player import HybridPlayerTracker
from tracking_postprocess import stabilize_player_roles


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AnalyzerConfig:
    """Runtime switches for the tennis analysis pipeline."""

    input_path: Path = PROJECT_ROOT / "input.mp4"
    output_path: Path = PROJECT_ROOT / "output.mp4"
    stats_path: Path = PROJECT_ROOT / "analysis_stats.json"
    ball_history_plot_path: Path | None = None
    cache_dir: Path = PROJECT_ROOT / ".cache"
    point_output_dir: Path | None = None

    # Stats need court projection; caching keeps repeated runs manageable.
    track_players: bool = False
    track_court: bool = True
    detect_bounces: bool = True
    detect_scene_cuts: bool = True
    compute_stats: bool = True
    use_cache: bool = True

    draw_ball_track: bool = True
    draw_players: bool = False
    draw_court_overlay: bool = False
    draw_court_keypoints: bool = False
    draw_bounces: bool = True
    draw_frame_number: bool = True
    draw_stats: bool = True

    ball_trace_length: int = 7
    min_point_frames: int = 10


def read_image_as_video(path_image: Path | str) -> tuple[list, int]:
    """Load a single image and return it in the same format as a video."""
    img = cv2.imread(str(path_image))

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path_image}")

    return [img], 1


def read_video(path_video: Path | str) -> tuple[list, int]:
    """Read all video frames into memory and return frames plus FPS."""
    cap = cv2.VideoCapture(str(path_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path_video}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames were read from video: {path_video}")

    return frames, fps


def save_frames(frames: Sequence, output_folder: Path | str | None = None) -> None:
    """Save every processed frame as an image for debugging."""
    if output_folder is None:
        output_folder = PROJECT_ROOT / "frames"
    else:
        output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        frame_filename = output_folder / f"frame_{i:05d}.png"
        cv2.imwrite(str(frame_filename), frame)


def save_video(frames: Sequence, path_output_video: Path | str, fps: int) -> None:
    """Save processed frames to an MP4 video."""
    if not frames:
        raise ValueError("No frames to save.")

    height, width = frames[0].shape[:2]
    output_path = Path(path_output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    for frame in frames:
        out.write(frame)

    out.release()


def scene_segments(num_frames: int, scene_cuts: Sequence[int], min_frames: int = 1) -> list[tuple[int, int]]:
    """Return inclusive-exclusive frame ranges split by hard scene cuts."""
    cuts = sorted(cut for cut in scene_cuts if 0 < cut < num_frames)
    starts = [0, *cuts]
    ends = [*cuts, num_frames]
    return [
        (start, end)
        for start, end in zip(starts, ends)
        if end - start >= min_frames
    ]


def save_point_videos(
    frames: Sequence,
    fps: int,
    scene_cuts: Sequence[int],
    output_dir: Path | str,
    min_frames: int = 10,
) -> list[Path]:
    """Save one MP4 per detected point/scene and return output paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    segments = scene_segments(len(frames), scene_cuts, min_frames=min_frames)
    if not segments:
        segments = [(0, len(frames))]

    for point_index, (start, end) in enumerate(segments, start=1):
        output_path = output_dir / f"point_{point_index:03d}_frames_{start:06d}_{end - 1:06d}.mp4"
        save_video(frames[start:end], output_path, fps)
        output_paths.append(output_path)

    return output_paths


def split_video_by_scene(
    input_path: Path | str,
    output_dir: Path | str,
    threshold: float = 0.55,
    min_frames: int = 10,
) -> list[Path]:
    """Stream an input video and write one raw MP4 per detected scene/point."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video dimensions: {input_path}")

    writer = None
    previous_hist = None
    point_index = 1
    segment_start = 0
    segment_frames = 0
    frame_num = 0
    output_paths = []

    def open_writer(index: int, start: int):
        path = output_dir / f"point_{index:03d}_frames_{start:06d}.mp4"
        video_writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {path}")
        return path, video_writer

    current_path, writer = open_writer(point_index, segment_start)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_hist = _frame_histogram_from_bgr(frame)
            is_cut = False
            if previous_hist is not None:
                distance = cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_BHATTACHARYYA)
                is_cut = distance >= threshold and segment_frames >= min_frames

            if is_cut:
                writer.release()
                final_path = _rename_point_video(current_path, segment_start, frame_num - 1)
                output_paths.append(final_path)
                point_index += 1
                segment_start = frame_num
                segment_frames = 0
                current_path, writer = open_writer(point_index, segment_start)

            writer.write(frame)
            segment_frames += 1
            previous_hist = current_hist
            frame_num += 1
            if frame_num % 1000 == 0:
                if total_frames:
                    print(f"Split progress: {frame_num}/{total_frames} frames, {point_index} current point(s)")
                else:
                    print(f"Split progress: {frame_num} frames, {point_index} current point(s)")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if segment_frames >= min_frames:
        final_path = _rename_point_video(current_path, segment_start, frame_num - 1)
        output_paths.append(final_path)
    elif current_path.exists():
        current_path.unlink()

    return output_paths


def _frame_histogram_from_bgr(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _rename_point_video(path: Path, start_frame: int, end_frame: int) -> Path:
    final_path = path.with_name(
        path.stem + f"_{end_frame:06d}.mp4"
    )
    if final_path != path:
        if final_path.exists():
            final_path.unlink()
        path.rename(final_path)
    return final_path


def analyze_point_scenes(
    input_path: Path | str,
    output_dir: Path | str,
    threshold: float = 0.55,
    min_frames: int = 10,
    min_analyze_frames: int = 75,
    min_shots: int = 2,
    min_bounces: int = 1,
    draw_players: bool = False,
    draw_court: bool = False,
    draw_court_keypoints: bool = False,
    keep_debug_scenes: bool = False,
    plot_ball_history: bool = False,
) -> list[tuple[Path, Path]]:
    """Split a compilation, analyze each scene, and keep scenes that look like played points."""
    output_dir = Path(output_dir)
    existing_raw_dir = output_dir / "raw_scenes"
    raw_dir = existing_raw_dir if existing_raw_dir.exists() else output_dir / ".scene_work" / "raw_scenes"
    videos_dir = output_dir / "point_videos"
    stats_dir = output_dir / "point_stats"
    plots_dir = output_dir / "ball_history_plots"
    rejected_dir = output_dir / "rejected_scenes"
    cache_dir = output_dir / ".cache"

    raw_paths = sorted(raw_dir.glob("*.mp4")) if raw_dir.exists() else []
    if raw_paths:
        print(f"Reusing {len(raw_paths)} existing raw scene clip(s) from: {raw_dir}")
    else:
        raw_paths = split_video_by_scene(
            input_path,
            raw_dir,
            threshold=threshold,
            min_frames=min_frames,
        )
    videos_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    for scene_index, raw_path in enumerate(raw_paths, start=1):
        output_path = videos_dir / raw_path.name
        stats_path = stats_dir / f"{raw_path.stem}.json"
        plot_path = plots_dir / f"{raw_path.stem}_ball_history.png"
        scene_frames = _video_frame_count(raw_path)

        if stats_path.exists():
            if _stats_look_like_played_point(stats_path, min_shots=min_shots, min_bounces=min_bounces):
                if output_path.exists():
                    print(f"Skipping already analyzed point scene: {raw_path.name}")
                    kept.append((output_path, stats_path))
                    continue
            else:
                _discard_scene(raw_path, rejected_dir, keep_debug_scenes)
                stats_path.unlink()
                if output_path.exists():
                    output_path.unlink()
                if plot_path.exists():
                    plot_path.unlink()
                print(f"Rejected already analyzed non-point scene: {raw_path.name}")
                continue

        if scene_frames is not None and scene_frames < min_analyze_frames:
            _discard_scene(raw_path, rejected_dir, keep_debug_scenes)
            if output_path.exists():
                output_path.unlink()
            if stats_path.exists():
                stats_path.unlink()
            if plot_path.exists():
                plot_path.unlink()
            print(f"Rejected short scene without full analysis: {raw_path.name}")
            continue

        print(f"Analyzing scene {scene_index}/{len(raw_paths)}: {raw_path.name}")

        scene_config = AnalyzerConfig(
            input_path=raw_path,
            output_path=output_path,
            stats_path=stats_path,
            ball_history_plot_path=plot_path if plot_ball_history else None,
            cache_dir=cache_dir,
            track_players=True,
            track_court=True,
            detect_bounces=True,
            detect_scene_cuts=True,
            compute_stats=True,
            use_cache=True,
            draw_ball_track=True,
            draw_players=draw_players,
            draw_court_overlay=draw_court,
            draw_court_keypoints=draw_court_keypoints,
            draw_bounces=True,
            draw_frame_number=True,
            draw_stats=True,
        )
        analyze_video(scene_config)

        if _stats_look_like_played_point(stats_path, min_shots=min_shots, min_bounces=min_bounces):
            kept.append((output_path, stats_path))
            if not keep_debug_scenes and raw_path.exists():
                raw_path.unlink()
            continue

        _discard_scene(raw_path, rejected_dir, keep_debug_scenes)
        if output_path.exists():
            output_path.unlink()
        if stats_path.exists():
            stats_path.unlink()
        if plot_path.exists():
            plot_path.unlink()

    if not keep_debug_scenes:
        _remove_empty_parents(raw_dir, stop_at=output_dir)

    return kept


def _video_frame_count(path: Path) -> int | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()
    return count


def _discard_scene(raw_path: Path, rejected_dir: Path, keep_debug_scenes: bool) -> None:
    if keep_debug_scenes:
        rejected_dir.mkdir(parents=True, exist_ok=True)
        raw_path.replace(rejected_dir / raw_path.name)
    elif raw_path.exists():
        raw_path.unlink()


def _remove_empty_parents(path: Path, stop_at: Path) -> None:
    path = path.resolve()
    stop_at = stop_at.resolve()
    while path != stop_at and path.exists():
        try:
            path.rmdir()
        except OSError:
            break
        path = path.parent


def _stats_look_like_played_point(stats_path: Path, min_shots: int, min_bounces: int) -> bool:
    if not stats_path.exists():
        return False

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    shots = stats.get("shot_events", [])
    bounces = stats.get("bounce_events", [])
    known_player_shots = [
        shot for shot in shots if shot.get("player_role") in {"top_player", "bottom_player"}
    ]
    playable_bounces = [
        bounce for bounce in bounces if bounce.get("phase") in {"serve", "game"}
    ]
    return len(known_player_shots) >= min_shots and len(playable_bounces) >= min_bounces


def create_player_tracker() -> HybridPlayerTracker:
    """Build the combined box and pose tracker used for player annotation."""
    return HybridPlayerTracker(
        box_model_path=str(PROJECT_ROOT / "weights" / "yolo26n.pt"),
        pose_model_path=str(PROJECT_ROOT / "weights" / "yolo26n-pose.pt"),
        conf=0.5,
        pose_conf=0.35,
    )


def track_players_by_scene(frames: Sequence, scene_cuts: Sequence[int] | None = None) -> list:
    """Track players independently per scene so IDs and anchors do not cross cuts."""
    scene_cuts = sorted(cut for cut in (scene_cuts or []) if 0 < cut < len(frames))
    starts = [0, *scene_cuts]
    ends = [*scene_cuts, len(frames)]
    player_tracks = []

    for start, end in zip(starts, ends):
        player_tracker = create_player_tracker()
        segment_tracks, _ = player_tracker.track_frames(frames[start:end])
        player_tracks.extend(segment_tracks)

    return player_tracks


def detect_ball_bounces(ball_track: Sequence[tuple]) -> set[int]:
    """Predict which frame numbers contain a bounce from the ball trajectory."""
    bounce_detector = BounceDetector(str(PROJECT_ROOT / "weights" / "bounce_model.cbm"))
    x_ball = [point[0] for point in ball_track]
    y_ball = [point[1] for point in ball_track]
    return bounce_detector.predict(x_ball, y_ball)


def cache_key(config: AnalyzerConfig, stage_name: str) -> Path:
    input_path = config.input_path.resolve()
    input_stat = input_path.stat()
    raw_key = "|".join(
        [
            stage_name,
            str(input_path),
            str(input_stat.st_size),
            str(input_stat.st_mtime_ns),
        ]
    )
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()
    return config.cache_dir / f"{stage_name}_{digest}.pkl"


def load_or_compute(config: AnalyzerConfig, stage_name: str, compute_fn):
    if not config.use_cache:
        return compute_fn()

    cache_path = cache_key(config, stage_name)
    if cache_path.exists():
        with cache_path.open("rb") as file:
            return pickle.load(file)

    result = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(result, file)
    return result


def draw_bounce_markers(frames: Sequence, ball_track: Sequence[tuple], bounces: set[int]) -> list:
    """Draw bounce labels on the video at the tracked ball position."""
    output_frames = []

    for frame_num, frame in enumerate(frames):
        annotated = frame.copy()
        if frame_num in bounces and frame_num < len(ball_track):
            point = ball_track[frame_num]
            if point[0] is None or point[1] is None:
                point = interpolate_point(ball_track, frame_num, max_gap=8)
            if has_valid_point(point):
                x = int(point[0])
                y = int(point[1])
                cv2.circle(annotated, (x, y), 14, (0, 165, 255), 3)
                cv2.putText(
                    annotated,
                    "bounce",
                    (x + 16, max(y - 16, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

        output_frames.append(annotated)

    return output_frames


def interpolate_point(points: Sequence, frame_num: int, max_gap: int):
    before = nearest_valid_point(points, frame_num - 1, step=-1, max_steps=max_gap)
    after = nearest_valid_point(points, frame_num + 1, step=1, max_steps=max_gap)
    if before is None or after is None:
        return (None, None)

    before_frame, before_point = before
    after_frame, after_point = after
    frame_span = after_frame - before_frame
    if frame_span <= 0:
        return (None, None)

    t = (frame_num - before_frame) / frame_span
    x = before_point[0] + (after_point[0] - before_point[0]) * t
    y = before_point[1] + (after_point[1] - before_point[1]) * t
    if not has_valid_point((x, y)):
        return (None, None)
    return (x, y)


def nearest_valid_point(points: Sequence, start: int, step: int, max_steps: int):
    frame = start
    checked = 0
    while 0 <= frame < len(points) and checked < max_steps:
        point = points[frame]
        if has_valid_point(point):
            return frame, point
        frame += step
        checked += 1
    return None


def has_valid_point(point) -> bool:
    if point is None or point[0] is None or point[1] is None:
        return False
    return bool(np.isfinite(point[0]) and np.isfinite(point[1]))


def draw_frame_numbers(frames: Sequence) -> list:
    """Draw frame numbers for debugging predicted events."""
    output_frames = []

    for frame_num, frame in enumerate(frames):
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"Frame: {frame_num}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        output_frames.append(annotated)

    return output_frames


def save_ball_history_plot(ball_track: Sequence, stats, output_path: Path | str) -> None:
    """Plot ball x/y pixel history with shot and bounce frame markers."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    frames = np.arange(len(ball_track))
    x_values = np.array(
        [np.nan if not has_valid_point(point) else float(point[0]) for point in ball_track]
    )
    y_values = np.array(
        [np.nan if not has_valid_point(point) else float(point[1]) for point in ball_track]
    )

    fig, left_axis = plt.subplots(figsize=(14, 6))
    right_axis = left_axis.twinx()

    y_line = left_axis.plot(frames, y_values, color="#1f77b4", linewidth=1.4, label="ball y pixel")
    x_line = right_axis.plot(frames, x_values, color="#ff7f0e", linewidth=1.1, label="ball x pixel")

    for event in stats.shot_events:
        left_axis.axvline(event.frame, color="#d62728", linewidth=1.0, alpha=0.8)
    for event in stats.bounce_events:
        left_axis.axvline(event.frame, color="#2ca02c", linewidth=1.0, linestyle="--", alpha=0.8)

    shot_proxy = plt.Line2D([0], [0], color="#d62728", linewidth=1.0, label="shot")
    bounce_proxy = plt.Line2D([0], [0], color="#2ca02c", linewidth=1.0, linestyle="--", label="bounce")
    lines = [*y_line, *x_line, shot_proxy, bounce_proxy]

    left_axis.set_title("Ball Pixel History")
    left_axis.set_xlabel("Frame")
    left_axis.set_ylabel("Y pixel", color="#1f77b4")
    right_axis.set_ylabel("X pixel", color="#ff7f0e")
    left_axis.grid(True, axis="x", alpha=0.25)
    left_axis.legend(lines, [line.get_label() for line in lines], loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def analyze_video(config: AnalyzerConfig) -> tuple[list, set[int]]:
    """Run selected analysis stages and return annotated frames plus bounces."""
    frames, fps = read_video(config.input_path)
    processed_frames = frames

    scene_cuts = []
    if config.detect_scene_cuts:
        scene_cuts = load_or_compute(
            config,
            "scene_cuts",
            lambda: detect_scene_cuts(frames),
        )

    player_tracks = None
    if config.track_players or config.draw_players:
        player_tracker = create_player_tracker()
        player_tracks = load_or_compute(
            config,
            "players_v11_scene_top_band",
            lambda: track_players_by_scene(frames, scene_cuts),
        )
        player_tracks = stabilize_player_roles(player_tracks, frames[0].shape)
        if config.draw_players:
            processed_frames = player_tracker.annotate_frames(processed_frames, player_tracks)

    ball_track = load_or_compute(config, "ball", lambda: track_ball(frames))

    homography_matrices = None
    court_keypoints = None
    if (
        config.track_court
        or config.compute_stats
        or config.draw_court_overlay
        or config.draw_court_keypoints
    ):
        homography_matrices, court_keypoints = load_or_compute(
            config,
            "court",
            lambda: track_court(frames),
        )

    bounces: set[int] = set()
    if config.detect_bounces:
        bounces = load_or_compute(
            config,
            "bounces",
            lambda: detect_ball_bounces(ball_track),
        )

    stats = None
    if config.compute_stats:
        stats = compute_match_stats(
            ball_track,
            bounces=bounces,
            fps=fps,
            homography_matrices=homography_matrices,
            player_tracks=player_tracks,
            scene_cuts=scene_cuts,
        )
        bounces = {event.frame for event in stats.bounce_events}
        save_stats(stats, config.stats_path)
        if config.ball_history_plot_path is not None:
            save_ball_history_plot(ball_track, stats, config.ball_history_plot_path)

    if config.draw_ball_track:
        processed_frames = draw_track(
            processed_frames,
            ball_track,
            trace=config.ball_trace_length,
        )

    if config.draw_bounces:
        processed_frames = draw_bounce_markers(processed_frames, ball_track, bounces)

    if config.draw_court_keypoints and court_keypoints is not None:
        processed_frames = draw_keypoints(processed_frames, court_keypoints)

    if (
        config.draw_court_overlay
        and homography_matrices is not None
        and court_keypoints is not None
    ):
        processed_frames = draw_court(
            processed_frames,
            homography_matrices,
            court_keypoints,
            ball_track=ball_track,
            bounces=bounces,
            player_tracks=player_tracks,
        )

    if config.draw_stats and stats is not None:
        processed_frames = draw_stats_overlay(processed_frames, stats)

    if config.draw_frame_number:
        processed_frames = draw_frame_numbers(processed_frames)

    save_video(processed_frames, config.output_path, fps)
    if config.point_output_dir is not None:
        save_point_videos(
            processed_frames,
            fps,
            scene_cuts,
            config.point_output_dir,
            min_frames=config.min_point_frames,
        )
    return processed_frames, bounces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a tennis video.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "input.mp4")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output.mp4")
    parser.add_argument("--stats-output", type=Path, default=PROJECT_ROOT / "analysis_stats.json")
    parser.add_argument("--split-points-dir", type=Path)
    parser.add_argument("--analyze-points-dir", type=Path)
    parser.add_argument("--split-scene-threshold", type=float, default=0.55)
    parser.add_argument("--min-point-frames", type=int, default=10)
    parser.add_argument("--min-analyze-frames", type=int, default=75)
    parser.add_argument("--min-point-shots", type=int, default=2)
    parser.add_argument("--min-point-bounces", type=int, default=1)
    parser.add_argument("--keep-debug-scenes", action="store_true")
    parser.add_argument("--plot-ball-history", action="store_true")
    parser.add_argument("--draw-players", action="store_true")
    parser.add_argument("--draw-court", action="store_true")
    parser.add_argument("--draw-court-keypoints", action="store_true")
    parser.add_argument("--no-players", action="store_true")
    parser.add_argument("--no-ball-track", action="store_true")
    parser.add_argument("--no-bounces", action="store_true")
    parser.add_argument("--no-frame-number", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--no-scene-cuts", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> AnalyzerConfig:
    compute_stats = (not args.no_stats) or args.plot_ball_history
    plot_path = args.stats_output.with_name(f"{args.stats_output.stem}_ball_history.png")
    return AnalyzerConfig(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats_output,
        ball_history_plot_path=plot_path if args.plot_ball_history else None,
        point_output_dir=args.split_points_dir,
        track_players=(compute_stats and not args.no_players) or args.draw_players,
        track_court=compute_stats or args.draw_court or args.draw_court_keypoints,
        detect_bounces=not args.no_bounces,
        detect_scene_cuts=(not args.no_scene_cuts) or args.split_points_dir is not None,
        compute_stats=compute_stats,
        use_cache=not args.no_cache,
        draw_ball_track=not args.no_ball_track,
        draw_players=args.draw_players,
        draw_court_overlay=args.draw_court,
        draw_court_keypoints=args.draw_court_keypoints,
        draw_bounces=not args.no_bounces,
        draw_frame_number=not args.no_frame_number,
        draw_stats=not args.no_stats,
        min_point_frames=args.min_point_frames,
    )


def main() -> None:
    args = parse_args()
    if args.analyze_points_dir is not None:
        kept = analyze_point_scenes(
            args.input,
            args.analyze_points_dir,
            threshold=args.split_scene_threshold,
            min_frames=args.min_point_frames,
            min_analyze_frames=args.min_analyze_frames,
            min_shots=args.min_point_shots,
            min_bounces=args.min_point_bounces,
            draw_players=args.draw_players,
            draw_court=args.draw_court,
            draw_court_keypoints=args.draw_court_keypoints,
            keep_debug_scenes=args.keep_debug_scenes,
            plot_ball_history=args.plot_ball_history,
        )
        print(f"Saved analysis for {len(kept)} played point scene(s) to: {args.analyze_points_dir}")
        return

    if args.split_points_dir is not None:
        output_paths = split_video_by_scene(
            args.input,
            args.split_points_dir,
            threshold=args.split_scene_threshold,
            min_frames=args.min_point_frames,
        )
        print(f"Saved {len(output_paths)} point video(s) to: {args.split_points_dir}")
        return

    config = config_from_args(args)
    _, bounces = analyze_video(config)

    if bounces:
        print(f"Predicted bounce frames: {sorted(bounces)}")
    else:
        print("No bounces detected.")

    print(f"Saved output video to: {config.output_path}")

if __name__ == "__main__":
    main()
