from __future__ import annotations

import argparse
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2

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
            if point[0] is not None and point[1] is not None:
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
    return (x, y)


def nearest_valid_point(points: Sequence, start: int, step: int, max_steps: int):
    frame = start
    checked = 0
    while 0 <= frame < len(points) and checked < max_steps:
        point = points[frame]
        if point is not None and point[0] is not None and point[1] is not None:
            return frame, point
        frame += step
        checked += 1
    return None


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
    parser.add_argument("--split-scene-threshold", type=float, default=0.55)
    parser.add_argument("--min-point-frames", type=int, default=10)
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
    return AnalyzerConfig(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats_output,
        point_output_dir=args.split_points_dir,
        track_players=(not args.no_stats and not args.no_players) or args.draw_players,
        track_court=(not args.no_stats) or args.draw_court or args.draw_court_keypoints,
        detect_bounces=not args.no_bounces,
        detect_scene_cuts=(not args.no_scene_cuts) or args.split_points_dir is not None,
        compute_stats=not args.no_stats,
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
