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


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class AnalyzerConfig:
    """Runtime switches for the tennis analysis pipeline."""

    input_path: Path = PROJECT_ROOT / "input.mp4"
    output_path: Path = PROJECT_ROOT / "output.mp4"
    stats_path: Path = PROJECT_ROOT / "analysis_stats.json"
    cache_dir: Path = PROJECT_ROOT / ".cache"

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


def create_player_tracker() -> HybridPlayerTracker:
    """Build the combined box and pose tracker used for player annotation."""
    return HybridPlayerTracker(
        box_model_path=str(PROJECT_ROOT / "weights" / "yolo26n.pt"),
        pose_model_path=str(PROJECT_ROOT / "weights" / "yolo26n-pose.pt"),
        conf=0.5,
        pose_conf=0.35,
    )


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
            "players",
            lambda: player_tracker.track_frames(frames)[0],
        )
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
    return processed_frames, bounces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a tennis video.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "input.mp4")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output.mp4")
    parser.add_argument("--stats-output", type=Path, default=PROJECT_ROOT / "analysis_stats.json")
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
        track_players=(not args.no_stats and not args.no_players) or args.draw_players,
        track_court=(not args.no_stats) or args.draw_court or args.draw_court_keypoints,
        detect_bounces=not args.no_bounces,
        detect_scene_cuts=not args.no_scene_cuts,
        compute_stats=not args.no_stats,
        use_cache=not args.no_cache,
        draw_ball_track=not args.no_ball_track,
        draw_players=args.draw_players,
        draw_court_overlay=args.draw_court,
        draw_court_keypoints=args.draw_court_keypoints,
        draw_bounces=not args.no_bounces,
        draw_frame_number=not args.no_frame_number,
        draw_stats=not args.no_stats,
    )


def main() -> None:
    config = config_from_args(parse_args())
    _, bounces = analyze_video(config)

    if bounces:
        print(f"Predicted bounce frames: {sorted(bounces)}")
    else:
        print("No bounces detected.")

    print(f"Saved output video to: {config.output_path}")


if __name__ == "__main__":
    main()
