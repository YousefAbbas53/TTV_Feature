from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from config import settings
from video_service import VideoGenerationError, VideoGenerator

# FIX: hard cap — keeps every clip within the non-sliding-window range.
# 5 s × 16 fps = 80 frames exactly.  Sliding window fires above 80 and
# produces ghosting / warped-motion artifacts on short clips.
MAX_FRAMES = 80


def _ffmpeg_concat_reencode(mp4_list: list[Path], out_path: Path, fps: int) -> bool:
    if not mp4_list:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat_file = out_path.parent / f"concat_{out_path.stem}.txt"
    with concat_file.open("w", encoding="utf-8") as handle:
        for item in mp4_list:
            safe_path = str(item).replace("'", "'\\''")
            handle.write(f"file '{safe_path}'\n")

    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-vf", f"fps={int(fps)},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-vsync", "cfr",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return proc.returncode == 0 and out_path.exists()






def generate_video_from_prompts(
    prompts_json_path: str | Path,
    output_video_path: str | Path = "final_video.mp4",
) -> str:
    prompts_path = Path(prompts_json_path)
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    with prompts_path.open("r", encoding="utf-8") as handle:
        prompt_items = json.load(handle)

    return generate_video_from_prompt_items(prompt_items, output_video_path)


def generate_video_from_prompt_items(
    prompt_items: list[dict],
    output_video_path: str | Path = "final_video.mp4",
) -> str:
    if not isinstance(prompt_items, list) or not prompt_items:
        raise VideoGenerationError("Prompts JSON must contain a non-empty list.")

    generator      = VideoGenerator(settings)
    rendered_paths: list[Path] = []
    target_fps     = settings.default_fps

    for index, item in enumerate(prompt_items):
        if not isinstance(item, dict):
            continue

        try:
            clip_fps = int(item.get("fps", settings.default_fps))
        except (TypeError, ValueError):
            clip_fps = settings.default_fps

        try:
            duration_seconds = int(item.get("duration_seconds", 5))
        except (TypeError, ValueError):
            duration_seconds = 5

        # FIX: cap frames so we never trigger the sliding-window path
        # (which causes ghosting / warped motion on short clips).
        # If the caller explicitly passed num_frames, respect it but still cap.
        if item.get("num_frames") is not None:
            try:
                clip_num_frames = min(int(item["num_frames"]), MAX_FRAMES)
            except (TypeError, ValueError):
                clip_num_frames = MAX_FRAMES
        else:
            clip_num_frames = min(max(1, duration_seconds * clip_fps), MAX_FRAMES)

        result = generator.generate(
            prompt=str(item.get("prompt", "")).strip(),
            negative_prompt=item.get("negative_prompt"),
            num_frames=clip_num_frames,
            fps=clip_fps,
            steps=item.get("steps",  settings.default_steps),
            cfg=item.get("cfg",      settings.default_cfg),
            seed=item.get("seed"),
            width=item.get("width",  settings.default_width),
            height=item.get("height", settings.default_height),
        )
        rendered_paths.append(Path(result.video_path))

        if index == 0:
            target_fps = clip_fps

    output_path = Path(output_video_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rendered_paths) == 1:
        shutil.copy2(rendered_paths[0], output_path)
        return str(output_path)

    merged = _ffmpeg_concat_reencode(rendered_paths, output_path, fps=target_fps)
    if not merged:
        raise VideoGenerationError("Rendered clips were created, but final merge failed.")
    return str(output_path)
