from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from config import settings
from video_generator import generate_video_from_prompt_items


def _probe_video(video_path: Path) -> dict[str, float | int | None]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return {
            "duration_seconds": None,
            "size_bytes": video_path.stat().st_size if video_path.exists() else None,
            "width": None,
            "height": None,
            "fps": None,
        }

    payload = json.loads(proc.stdout or "{}")
    fmt = payload.get("format", {}) or {}
    streams = payload.get("streams", []) or []
    stream = streams[0] if streams else {}

    fps = None
    frame_rate = stream.get("r_frame_rate")
    if isinstance(frame_rate, str) and "/" in frame_rate:
        num, den = frame_rate.split("/", 1)
        try:
            fps = round(float(num) / float(den), 3)
        except (TypeError, ValueError, ZeroDivisionError):
            fps = None

    try:
        duration = float(fmt.get("duration")) if fmt.get("duration") is not None else None
    except (TypeError, ValueError):
        duration = None

    try:
        size_bytes = int(fmt.get("size")) if fmt.get("size") is not None else None
    except (TypeError, ValueError):
        size_bytes = None

    return {
        "duration_seconds": duration,
        "size_bytes": size_bytes,
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
    }


def _load_selected_scenes(preview_path: Path, scene_ids: list[str]) -> list[dict]:
    with preview_path.open("r", encoding="utf-8") as handle:
        preview = json.load(handle)

    scenes = preview.get("scenes", [])
    if not isinstance(scenes, list):
        raise ValueError("Preview file contains no valid scenes array.")

    selected = []
    wanted = set(scene_ids)
    for scene in scenes:
        if isinstance(scene, dict) and str(scene.get("scene_id")) in wanted:
            selected.append(scene)

    found_ids = {str(scene.get("scene_id")) for scene in selected}
    missing = [scene_id for scene_id in scene_ids if scene_id not in found_ids]
    if missing:
        raise ValueError(f"Missing scene_ids in preview: {', '.join(missing)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-path", required=True)
    parser.add_argument("--scene-id", action="append", dest="scene_ids", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--report-path", required=True)
    args = parser.parse_args()

    preview_path = Path(args.preview_path).resolve()
    output_path = Path(args.output_path).resolve()
    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_scenes = _load_selected_scenes(preview_path, args.scene_ids)

    started_at = time.time()
    final_video_path = Path(generate_video_from_prompt_items(selected_scenes, output_path))
    elapsed_seconds = time.time() - started_at
    video_info = _probe_video(final_video_path)

    payload = {
        "video_backend": "wan",
        "video_model_preset": settings.video_model_preset,
        "default_model_type": settings.default_model_type,
        "defaults": {
            "width": settings.default_width,
            "height": settings.default_height,
            "num_frames": settings.default_num_frames,
            "fps": settings.default_fps,
            "steps": settings.default_steps,
            "cfg": settings.default_cfg,
        },
        "preview_path": str(preview_path),
        "scene_ids": args.scene_ids,
        "selected_scene_count": len(selected_scenes),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "video_path": str(final_video_path),
        "video_info": video_info,
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
