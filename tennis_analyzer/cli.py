from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tennis_analyzer.pipeline import analyze_video
from tennis_analyzer.schemas import AnalysisOptions, PipelineOptions, VisualizationOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a tennis video")
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-frames", type=int, default=128)
    parser.add_argument("--ball-batch-size", type=int, default=4)
    parser.add_argument("--execution-mode", default="low_memory")
    parser.add_argument("--full", action="store_true", help="Enable all model-based analysis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    analysis = AnalysisOptions(
        ball_tracking=args.full,
        court_detection=args.full,
        player_tracking=args.full,
        pose_tracking=args.full,
        bounce_detection=args.full,
        statistics=args.full,
    )
    visuals = VisualizationOptions(
        ball_trail=args.full,
        bounce_markers=args.full,
        frame_number=True,
        player_boxes=args.full,
        player_poses=args.full,
        statistics_overlay=args.full,
    )
    result = analyze_video(
        args.input,
        args.output_dir,
        PipelineOptions(
            analysis,
            visuals,
            chunk_size=args.chunk_frames,
            ball_batch_size=args.ball_batch_size,
            device=args.device,
            execution_mode=args.execution_mode,
        ),
        model_root=args.models,
    )
    logging.getLogger(__name__).info("Wrote %s", result.output_video)


if __name__ == "__main__":
    main()
