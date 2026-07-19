from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from tennis_analyzer.config import ModelPaths
from tennis_analyzer.errors import AnalysisCancelled, MissingModelError, VideoProcessingError
from tennis_analyzer.pipeline.artifact import read_artifact, write_artifact
from tennis_analyzer.pipeline.ball_track import postprocess_ball_track
from tennis_analyzer.pipeline.chunks import iter_frame_chunks
from tennis_analyzer.pipeline.court_calibration import calibrated_keypoints, calibration_homography
from tennis_analyzer.pipeline.progress import WeightedProgress, WorkStage
from tennis_analyzer.pipeline.stages import StageFactories
from tennis_analyzer.schemas import AnalysisResult, PipelineOptions
from tennis_analyzer.video import normalize_video, probe_video
from tracking_postprocess import stabilize_player_roles

ProgressCallback = Callable[[str, int, str], None]
CancellationCheck = Callable[[], bool]
logger = logging.getLogger(__name__)


def _save_result(path: Path, result: AnalysisResult) -> None:
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def render_from_artifact(source, artifact_path, destination, visual, progress_callback=None, court_calibration=None):
    """Render annotations without loading any analysis model."""
    artifact = read_artifact(Path(artifact_path))
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = probe_video(source)
    expected_frames = int(artifact.get("frame_count") or 0)
    if expected_frames <= 0:
        raise VideoProcessingError("Analysis artifact contains no frames")
    ball_track = artifact["ball_track"]
    if len(ball_track) != expected_frames:
        raise VideoProcessingError("Analysis artifact ball track is not aligned with the source video")
    bounces = set(artifact["bounces"])
    homographies = [
        np.asarray(item, dtype=np.float32) if item is not None else None for item in artifact["homographies"]
    ]
    keypoints = artifact.get("court_keypoints", [])
    player_tracks = artifact.get("player_tracks", [])
    summary = artifact.get("summary", {})
    if court_calibration:
        corrected_homography = calibration_homography(court_calibration)
        corrected_keypoints = calibrated_keypoints(court_calibration)
        homographies = [corrected_homography] * expected_frames
        keypoints = [corrected_keypoints] * expected_frames
    raw_output = destination / ".rendered.mp4"
    output = destination / "rendered.mp4"
    writer = cv2.VideoWriter(
        str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), metadata.fps, (metadata.width, metadata.height)
    )
    if not writer.isOpened():
        raise VideoProcessingError("Could not create render output")
    total = 0
    frames_complete = False
    bounce_history: list[tuple[object, object]] = []
    try:
        for chunk in iter_frame_chunks(source, 128):
            for offset, frame in enumerate(chunk.frames):
                index = chunk.start_frame + offset
                if index >= expected_frames:
                    raise VideoProcessingError("Source video has more frames than the analysis artifact")
                annotated = frame.copy()
                if visual.ball_trail:
                    _draw_ball_trail(annotated, ball_track, index, visual)
                if visual.court_overlay and index < len(homographies):
                    from court import draw_court_overlay_in_place

                    draw_court_overlay_in_place(
                        annotated,
                        homographies[index],
                        ball_track[index],
                        index in bounces,
                        bounce_history=bounce_history,
                    )
                if visual.bounce_markers and index in bounces:
                    point = ball_track[index]
                    if point[0] is not None and point[1] is not None:
                        cv2.circle(annotated, (int(point[0]), int(point[1])), 14, (0, 165, 255), 3)
                if visual.court_keypoints and index < len(keypoints) and keypoints[index] is not None:
                    for number, point in enumerate(keypoints[index]):
                        x, y = int(point[0][0]), int(point[0][1])
                        cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)
                        cv2.putText(
                            annotated, str(number), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                        )
                if index < len(player_tracks) and (visual.player_boxes or visual.player_poses):
                    _draw_saved_players(
                        annotated,
                        player_tracks[index],
                        visual.player_boxes,
                        visual.player_poses,
                        visual.top_player_label,
                        visual.bottom_player_label,
                        visual.bgr_color("player_box_color"),
                    )
                if visual.statistics_overlay:
                    _draw_saved_summary(annotated, summary)
                if visual.frame_number:
                    cv2.putText(annotated, f"Frame: {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                writer.write(annotated)
                if index in bounces:
                    bounce_history.append(
                        (ball_track[index], homographies[index] if index < len(homographies) else None)
                    )
                total += 1
            if progress_callback:
                progress_callback(
                    "rendering", min(99, int(total * 100 / max(1, len(ball_track)))), "Rendering saved analysis"
                )
        if total != expected_frames:
            raise VideoProcessingError("Source video frame count does not match the analysis artifact")
        frames_complete = True
    finally:
        writer.release()
        if not frames_complete:
            raw_output.unlink(missing_ok=True)
    try:
        normalize_video(raw_output, source, output, timeout=max(120, int(metadata.duration_seconds * 5)))
    finally:
        raw_output.unlink(missing_ok=True)
    rendered = probe_video(output)
    if (rendered.width, rendered.height) != (metadata.width, metadata.height):
        output.unlink(missing_ok=True)
        raise VideoProcessingError("Rendered video dimensions do not match the source")
    if progress_callback:
        progress_callback("completed", 100, "Render completed")
    return output


def _draw_ball_trail(frame, ball_track, index: int, visual) -> None:
    color = visual.bgr_color("ball_trail_color")
    for age in range(visual.ball_trail_length):
        point_index = index - age
        if point_index < 0:
            break
        point = ball_track[point_index]
        if point[0] is None or point[1] is None:
            continue
        radius = max(1, round(visual.ball_trail_size * (visual.ball_trail_length - age) / visual.ball_trail_length))
        cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1)


def _draw_saved_players(
    frame, players, boxes: bool, poses: bool, top_label: str, bottom_label: str, box_color=None
) -> None:
    skeleton = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    for player in players:
        role = player.get("role", "player")
        color = box_color or ((255, 80, 40) if role == "top_player" else (40, 200, 80))
        if boxes and player.get("bbox"):
            x1, y1, x2, y2 = map(int, player["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = top_label.strip() if role == "top_player" else bottom_label.strip()
            cv2.putText(
                frame,
                label or role.replace("_", " "),
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        points, confidences = player.get("keypoints_xy"), player.get("keypoints_conf")
        if poses and points:
            for first, second in skeleton:
                if confidences and (confidences[first] < 0.25 or confidences[second] < 0.25):
                    continue
                a, b = points[first], points[second]
                cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), color, 2)


def _draw_saved_summary(frame, summary) -> None:
    lines = [
        f"Shots: {summary.get('shot_count', 0)}",
        f"Bounces: {summary.get('bounce_count', 0)}",
        f"Max ball speed: {summary.get('max_ball_speed_kmh') or 0:.1f} km/h",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 45), (330, 135), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (22, 72 + index * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def _histogram(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _cancelled(check: CancellationCheck | None) -> None:
    if check and check():
        raise AnalysisCancelled("Analysis was cancelled")


def _require_models(options: PipelineOptions, models: ModelPaths) -> None:
    required: dict[Path, str] = {}
    analysis = options.analysis
    if analysis.ball_tracking:
        required[models.ball] = "ball tracking"
    if analysis.bounce_detection:
        required[models.bounce] = "bounce detection"
    if analysis.court_detection:
        required[models.court] = "court detection"
    if analysis.player_tracking:
        required[models.player] = "player tracking"
    if analysis.pose_tracking:
        required[models.pose] = "pose tracking"
    missing = [f"{path.name} ({feature})" for path, feature in required.items() if not path.is_file()]
    if missing:
        raise MissingModelError("Missing required model file(s): " + ", ".join(missing))


def progress_stages(analysis, include_render: bool = True) -> list[WorkStage]:
    """Return the exact weighted stage plan for the selected analysis options."""
    stages = [WorkStage("scanning", 1)]
    if analysis.ball_tracking:
        stages.append(WorkStage("ball_tracking", 6))
    if analysis.court_detection:
        stages.append(WorkStage("court_detection", 6))
    if analysis.player_tracking:
        stages.append(
            WorkStage(
                "pose_tracking" if analysis.pose_tracking else "player_tracking", 8 if analysis.pose_tracking else 5
            )
        )
    if analysis.bounce_detection or analysis.statistics:
        stages.append(WorkStage("events", 1))
    if include_render:
        stages.append(WorkStage("rendering", 3))
        if analysis.point_analysis:
            stages.append(WorkStage("point_analysis", 2))
        stages.append(WorkStage("normalizing", 3))
    return stages


def _release_stage(stage) -> None:
    try:
        stage.close()
    except Exception:
        logger.exception("Model stage cleanup failed")
    finally:
        gc.collect()


def _require_aligned(name: str, results: list, frame_count: int) -> None:
    if len(results) != frame_count:
        raise VideoProcessingError(f"{name} produced {len(results)} results for {frame_count} source frames")


def analyze_video(
    input_path: Path | str,
    output_dir: Path | str,
    options: PipelineOptions | None = None,
    progress_callback: ProgressCallback | None = None,
    cancellation_check: CancellationCheck | None = None,
    model_root: Path | str | None = None,
    stage_factories: StageFactories | None = None,
) -> AnalysisResult:
    """Analyze a video in bounded, job-scoped model passes.

    Per-frame result index ``i`` always describes zero-based source frame ``i``.
    Source chunks never overlap; temporal model adapters add and trim their own
    context while returning exactly one result per source frame.
    """
    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selected = (options or PipelineOptions()).validated()
    factories = stage_factories or StageFactories()
    models = ModelPaths.from_root(Path(model_root or os.environ.get("MODEL_ROOT", "models")).resolve())
    _require_models(selected, models)
    metadata = probe_video(source)
    estimated_frames = metadata.frame_count or max(1, int(metadata.duration_seconds * metadata.fps))

    def emit(stage: str, percent: int, message: str) -> None:
        logger.info("%s", message, extra={"stage": stage, "progress": percent})
        if progress_callback:
            progress_callback(stage, percent, message)

    progress = WeightedProgress(progress_stages(selected.analysis, selected.render_output), emit)
    logger.info(
        "Starting low-memory analysis",
        extra={
            "device": selected.device,
            "chunk_frames": selected.chunk_size,
            "ball_batch_size": selected.ball_batch_size,
            "execution_mode": selected.execution_mode,
            "resolution": f"{metadata.width}x{metadata.height}",
        },
    )

    analysis = selected.analysis
    visual = selected.visualization
    scene_cuts: list[int] = []
    previous_hist: np.ndarray | None = None
    decoded_frames = 0
    for chunk in iter_frame_chunks(source, selected.chunk_size):
        _cancelled(cancellation_check)
        for frame in chunk.frames:
            if frame.shape[:2] != (metadata.height, metadata.width):
                raise VideoProcessingError("Decoded frame dimensions changed during the video")
        if analysis.scene_cut_detection:
            for offset, frame in enumerate(chunk.frames):
                current = _histogram(frame)
                if (
                    previous_hist is not None
                    and cv2.compareHist(previous_hist, current, cv2.HISTCMP_BHATTACHARYYA) >= 0.55
                ):
                    scene_cuts.append(chunk.start_frame + offset)
                previous_hist = current
        decoded_frames = chunk.end_frame
        progress.update("scanning", decoded_frames, estimated_frames, f"Scanned {decoded_frames} frames")

    frame_total = decoded_frames
    if metadata.frame_count is not None and frame_total != metadata.frame_count:
        raise VideoProcessingError(
            f"Decoded {frame_total} frames but the video declares {metadata.frame_count}; refusing partial analysis"
        )
    ball_track: list[tuple[float | None, float | None]]
    if analysis.ball_tracking:
        raw_ball_track = []
        ball_stage = factories.ball(models.ball, selected.device, selected.ball_batch_size)
        try:
            for chunk in iter_frame_chunks(source, selected.chunk_size):
                _cancelled(cancellation_check)
                raw_ball_track.extend(ball_stage.process_chunk(chunk.frames))
                progress.update(
                    "ball_tracking", chunk.end_frame, frame_total, f"Tracked ball through frame {chunk.end_frame}"
                )
        finally:
            _release_stage(ball_stage)
        _require_aligned("Ball tracking", raw_ball_track, frame_total)
        ball_track = postprocess_ball_track(raw_ball_track)
        del raw_ball_track
    else:
        ball_track = [(None, None)] * frame_total

    homographies: list[object] = []
    keypoints: list[object] = []
    if analysis.court_detection:
        court_stage = factories.court(models.court, selected.device)
        try:
            for chunk in iter_frame_chunks(source, selected.chunk_size):
                _cancelled(cancellation_check)
                chunk_homographies, chunk_keypoints = court_stage.process_chunk(chunk.frames)
                homographies.extend(chunk_homographies)
                keypoints.extend(chunk_keypoints)
                progress.update(
                    "court_detection", chunk.end_frame, frame_total, f"Detected court through frame {chunk.end_frame}"
                )
        finally:
            _release_stage(court_stage)
        _require_aligned("Court homographies", homographies, frame_total)
        _require_aligned("Court keypoints", keypoints, frame_total)

    player_tracks: list[list[object]] = []
    player_stage = None
    if analysis.player_tracking:
        player_stage = factories.player(models.player, models.pose, selected.device, analysis.pose_tracking)
        player_stage_name = "pose_tracking" if analysis.pose_tracking else "player_tracking"
        try:
            for chunk in iter_frame_chunks(source, selected.chunk_size):
                _cancelled(cancellation_check)
                player_tracks.extend(player_stage.process_chunk(chunk.frames))
                progress.update(
                    player_stage_name,
                    chunk.end_frame,
                    frame_total,
                    f"Tracked players through frame {chunk.end_frame}",
                )
        finally:
            _release_stage(player_stage)
        _require_aligned("Player tracking", player_tracks, frame_total)
        player_tracks = stabilize_player_roles(player_tracks, (metadata.height, metadata.width))

    _cancelled(cancellation_check)
    bounces: set[int] = set()
    if analysis.bounce_detection:
        detector = factories.bounce(models.bounce)
        bounces = set(detector.predict([p[0] for p in ball_track], [p[1] for p in ball_track]))

    stats = None
    shots: list[dict] = []
    bounce_events: list[dict] = []
    player_statistics: dict = {}
    summary: dict = {"frames_processed": len(ball_track), "scene_count": len(scene_cuts) + 1}
    if analysis.statistics:
        from analysis import compute_match_stats

        stats = compute_match_stats(
            ball_track,
            bounces,
            int(round(metadata.fps)),
            homography_matrices=homographies,
            player_tracks=player_tracks,
            scene_cuts=scene_cuts,
        )
        stats_data = stats.to_dict()
        shots = stats_data["shot_events"]
        bounce_events = stats_data["bounce_events"]
        player_statistics = stats_data["player_stats"]
        bounces = {event["frame"] for event in bounce_events}
        summary.update(
            shot_count=len(shots),
            bounce_count=len(bounce_events),
            average_ball_speed_kmh=stats_data["average_ball_speed_kmh"],
            max_ball_speed_kmh=stats_data["max_ball_speed_kmh"],
        )
    else:
        bounce_events = [
            {"frame": frame, "classification": "unclassified", "experimental": True} for frame in sorted(bounces)
        ]
        summary.update(shot_count=0, bounce_count=len(bounce_events))
    if analysis.bounce_detection or analysis.statistics:
        progress.update("events", 1, 1, "Computed events and summary statistics")

    artifact_path = destination / "analysis-artifact.json"
    write_artifact(
        artifact_path,
        {
            "metadata": metadata,
            "analysis_options": asdict(analysis),
            "frame_count": frame_total,
            "ball_track": ball_track,
            "homographies": homographies,
            "court_keypoints": keypoints,
            "player_tracks": player_tracks,
            "bounces": sorted(bounces),
            "shots": shots,
            "bounce_events": bounce_events,
            "player_statistics": player_statistics,
            "summary": summary,
            "scene_cuts": scene_cuts,
        },
    )

    if not selected.render_output:
        result_path = destination / "result.json"
        result = AnalysisResult(
            input_filename=source.name,
            output_video=None,
            result_json=result_path.name,
            metadata=metadata,
            analysis_options=asdict(analysis),
            visualization_options={},
            analysis_artifact=artifact_path.name,
            shots=shots,
            bounces=bounce_events,
            player_statistics=player_statistics,
            summary=summary,
            scene_cuts=scene_cuts,
            warnings=["Speeds, event classification, and in/out calls are experimental estimates."],
        )
        _save_result(result_path, result)
        progress.complete()
        return result

    raw_output = destination / ".annotated.mp4"
    raw_output.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(raw_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        writer.release()
        raise VideoProcessingError("Could not create the intermediate video")
    render_succeeded = False
    try:
        for chunk in iter_frame_chunks(source, selected.chunk_size):
            start, frames = chunk.start_frame, chunk.frames
            _cancelled(cancellation_check)
            for offset, frame in enumerate(frames):
                index = start + offset
                annotated = frame.copy()
                if visual.ball_trail:
                    for age in range(7):
                        track_index = index - age
                        if track_index < 0:
                            break
                        point = ball_track[track_index]
                        if point[0] is None or point[1] is None:
                            continue
                        cv2.circle(annotated, (int(point[0]), int(point[1])), 2, (0, 0, 255), max(1, 7 - age))
                if visual.bounce_markers and index in bounces and ball_track[index][0] is not None:
                    x, y = int(ball_track[index][0]), int(ball_track[index][1])
                    cv2.circle(annotated, (x, y), 14, (0, 165, 255), 3)
                    cv2.putText(
                        annotated, "bounce", (x + 15, max(20, y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
                    )
                if (visual.player_boxes or visual.player_poses) and player_stage:
                    annotated = player_stage.draw(annotated, player_tracks[index])
                if visual.court_keypoints and index < len(keypoints) and keypoints[index] is not None:
                    for number, point in enumerate(keypoints[index]):
                        x, y = int(point[0, 0]), int(point[0, 1])
                        cv2.circle(annotated, (x, y), 6, (0, 0, 255), -1)
                        cv2.putText(
                            annotated, str(number), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
                        )
                if visual.court_overlay and index < len(homographies):
                    from court import draw_court_overlay_in_place

                    annotated = draw_court_overlay_in_place(
                        annotated,
                        homographies[index],
                        ball_point=ball_track[index],
                        bounce=index in bounces,
                        players=player_tracks[index] if index < len(player_tracks) else None,
                    )
                if visual.statistics_overlay and stats is not None:
                    from analysis import draw_stats_overlay

                    annotated = draw_stats_overlay([annotated], stats)[0]
                if visual.frame_number:
                    cv2.putText(annotated, f"Frame: {index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                writer.write(annotated)
            progress.update("rendering", chunk.end_frame, frame_total, f"Rendered {chunk.end_frame} frames")
        render_succeeded = True
    finally:
        writer.release()
        if not render_succeeded:
            try:
                raw_output.unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not remove failed intermediate video")
    if not raw_output.is_file() or raw_output.stat().st_size == 0:
        raise VideoProcessingError("The intermediate video writer produced no output")

    plots: list[str] = []
    if visual.ball_history_plot and stats is not None:
        from main import save_ball_history_plot

        plot = destination / "ball_history.png"
        save_ball_history_plot(ball_track, stats, plot)
        plots.append(plot.name)

    if analysis.point_analysis:
        progress.update("point_analysis", 0, 1, "Creating point videos")
    points = _create_points(
        source, destination, scene_cuts, metadata, frame_total, analysis.point_analysis, cancellation_check
    )
    if analysis.point_analysis:
        progress.update("point_analysis", 1, 1, "Created point videos")
    output_video = destination / "analyzed.mp4"
    progress.update("normalizing", 0, 1, "Encoding browser-compatible output")
    try:
        normalize_video(raw_output, source, output_video, timeout=max(120, int(metadata.duration_seconds * 5)))
    finally:
        raw_output.unlink(missing_ok=True)
    normalized = probe_video(output_video)
    if (normalized.width, normalized.height) != (metadata.width, metadata.height):
        output_video.unlink(missing_ok=True)
        raise VideoProcessingError("Final video dimensions do not match the source")
    if normalized.frame_count is not None and normalized.frame_count != frame_total:
        output_video.unlink(missing_ok=True)
        raise VideoProcessingError("Final video frame count does not match the source")
    expected_duration = frame_total / metadata.fps
    duration_tolerance = max(0.1, 1.5 / metadata.fps)
    if abs(normalized.duration_seconds - expected_duration) > duration_tolerance:
        output_video.unlink(missing_ok=True)
        raise VideoProcessingError("Final video duration does not match the decoded source")
    progress.update("normalizing", 1, 1, "Finalized browser-compatible output")

    result_path = destination / "result.json"
    result = AnalysisResult(
        input_filename=source.name,
        output_video=output_video.name,
        result_json=result_path.name,
        metadata=metadata,
        analysis_options=asdict(analysis),
        visualization_options=asdict(visual),
        analysis_artifact=artifact_path.name,
        shots=shots,
        bounces=bounce_events,
        player_statistics=player_statistics,
        summary=summary,
        scene_cuts=scene_cuts,
        points=points,
        plots=plots,
        warnings=["Speeds, event classification, and in/out calls are experimental estimates."],
    )
    _save_result(result_path, result)
    progress.complete()
    return result


def _create_points(
    source: Path,
    output_dir: Path,
    cuts: list[int],
    metadata,
    frame_total: int,
    enabled: bool,
    check,
) -> list[dict]:
    if not enabled:
        return []
    boundaries = [0, *cuts, frame_total]
    point_dir = output_dir / "points"
    point_dir.mkdir(exist_ok=True)
    points = []
    for number, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False), 1):
        _cancelled(check)
        if end <= start:
            continue
        name = f"point_{number:03d}.mp4"
        temporary = point_dir / f".{name}"
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start / metadata.fps:.3f}",
            "-i",
            str(source),
            "-t",
            f"{(end - start) / metadata.fps:.3f}",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        try:
            subprocess.run(
                command, check=True, capture_output=True, timeout=max(60, int((end - start) / metadata.fps * 3))
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise VideoProcessingError("Point video conversion produced no output")
        except (OSError, subprocess.SubprocessError) as exc:
            temporary.unlink(missing_ok=True)
            raise VideoProcessingError("Could not create point video") from exc
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        final = point_dir / name
        temporary.replace(final)
        points.append(
            {
                "number": number,
                "start_frame": start,
                "end_frame": end - 1,
                "video": f"points/{name}",
                "experimental": True,
            }
        )
    return points
