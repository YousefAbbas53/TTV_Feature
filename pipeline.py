from __future__ import annotations

import argparse
from pathlib import Path

from analyzer import run_analysis
from video_generator import generate_video_from_prompts


def run_pipeline(
    book_path: str | Path,
    output_video_path: str | Path = "final_video.mp4",
    *,
    prompts_output_path: str | Path = "temp_prompts.json",
    max_scenes: int = 3,
    scene_window: int = 3,
) -> str:
    prompts_file = run_analysis(
        book_path=book_path,
        output_json_path=prompts_output_path,
        max_scenes=max_scenes,
        scene_window=scene_window,
    )
    return generate_video_from_prompts(prompts_file, output_video_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", required=True, help="Path to PDF or TXT file")
    parser.add_argument("--output_video", default="final_video.mp4")
    parser.add_argument("--prompts_output", default="temp_prompts.json")
    parser.add_argument("--max_scenes", type=int, default=3)
    parser.add_argument("--scene_window", type=int, default=3)
    args = parser.parse_args()

    print("Step 1: Analyzing book...")
    print("Step 2: Generating video from prompts...")
    video_path = run_pipeline(
        book_path=args.book,
        output_video_path=args.output_video,
        prompts_output_path=args.prompts_output,
        max_scenes=args.max_scenes,
        scene_window=args.scene_window,
    )
    print(f"Done! Video saved at {video_path}")


if __name__ == "__main__":
    main()
