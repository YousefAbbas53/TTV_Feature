from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from config import settings
from model_profiles import MODEL_PRESETS


PRESET_ENV_OVERRIDES: dict[str, dict[str, str]] = {
    "wan_t2v_1_3b": {
        "VIDEO_MODEL_PRESET": "wan_t2v_1_3b",
        "VIDEO_MODEL_TYPE": "t2v_1.3B",
        "VIDEO_WIDTH": "768",
        "VIDEO_HEIGHT": "432",
        "VIDEO_NUM_FRAMES": "80",
        "VIDEO_FPS": "16",
        "VIDEO_STEPS": "40",
        "VIDEO_CFG": "4.8",
    },
    "wan_t2v_14b": {
        "VIDEO_MODEL_PRESET": "wan_t2v_14b",
        "VIDEO_BACKEND": "wan",
        "VIDEO_MODEL_TYPE": "t2v",
        "VIDEO_WIDTH": "832",
        "VIDEO_HEIGHT": "480",
        "VIDEO_NUM_FRAMES": "49",
        "VIDEO_FPS": "16",
        "VIDEO_STEPS": "32",
        "VIDEO_CFG": "4.0",
    },
    "hunyuan_video_1_5": {
        "VIDEO_MODEL_PRESET": "hunyuan_video_1_5",
        "VIDEO_BACKEND": "hunyuan",
        "HUNYUAN_MODEL_ID": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "VIDEO_WIDTH": "848",
        "VIDEO_HEIGHT": "480",
        "VIDEO_NUM_FRAMES": "49",
        "VIDEO_FPS": "16",
        "VIDEO_STEPS": "20",
        "VIDEO_CFG": "1.5",
    },
}


def _resolve_preview_path(preview_id: str | None, preview_path: str | None) -> Path:
    if preview_path:
        path = Path(preview_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Preview file not found: {path}")
        return path

    if not preview_id:
        raise ValueError("Provide either --preview-id or --preview-path.")

    path = (settings.temp_dir / "previews" / f"{preview_id}.json").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Preview file not found for preview_id: {preview_id}")
    return path


def _read_gpu_memory() -> dict[str, object] | None:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return None

    gpus = []
    peak_used = 0
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            used = int(parts[2])
            total = int(parts[3])
        except ValueError:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_used_mb": used,
                "memory_total_mb": total,
            }
        )
        peak_used = max(peak_used, used)
    return {"gpus": gpus, "peak_used_mb": peak_used}


def _run_single_benchmark(
    preset_name: str,
    preview_path: Path,
    scene_ids: list[str],
    output_dir: Path,
    poll_interval: float,
) -> dict[str, object]:
    env = os.environ.copy()
    env.update(PRESET_ENV_OVERRIDES[preset_name])

    output_path = output_dir / f"{preset_name}_{preview_path.stem}.mp4"
    report_path = output_dir / f"{preset_name}_{preview_path.stem}.json"

    command = [
        sys.executable,
        "benchmark_render_once.py",
        "--preview-path",
        str(preview_path),
        "--output-path",
        str(output_path),
        "--report-path",
        str(report_path),
    ]
    for scene_id in scene_ids:
        command.extend(["--scene-id", scene_id])

    started_at = time.time()
    proc = subprocess.Popen(
        command,
        cwd=settings.app_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    peak_vram_mb = 0
    gpu_snapshot = None
    output_tail = ""
    while proc.poll() is None:
        snapshot = _read_gpu_memory()
        if snapshot is not None:
            gpu_snapshot = snapshot
            peak_vram_mb = max(peak_vram_mb, int(snapshot["peak_used_mb"]))
        time.sleep(poll_interval)

    stdout_text = ""
    if proc.stdout is not None:
        stdout_text = proc.stdout.read() or ""
        output_tail = stdout_text[-4000:]

    finished_at = time.time()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark for preset '{preset_name}' failed with exit code {proc.returncode}.\n{output_tail}"
        )

    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    report["wall_time_seconds"] = round(finished_at - started_at, 3)
    report["peak_vram_mb_estimate"] = peak_vram_mb or None
    report["gpu_snapshot"] = gpu_snapshot
    report["stdout_tail"] = output_tail
    report["manual_review"] = {
        "motion_coherence_score": None,
        "prompt_adherence_score": None,
        "notes": "",
    }
    return report


def _add_cost_metrics(report: dict[str, object], hourly_cost: float | None) -> None:
    if hourly_cost is None:
        report["cost_estimate_usd"] = None
        report["cost_per_video_second_usd"] = None
        return

    elapsed_seconds = float(report.get("wall_time_seconds") or report.get("elapsed_seconds") or 0.0)
    video_info = report.get("video_info", {}) or {}
    duration = video_info.get("duration_seconds")
    total_cost = (elapsed_seconds / 3600.0) * hourly_cost
    report["cost_estimate_usd"] = round(total_cost, 4)
    if duration:
        report["cost_per_video_second_usd"] = round(total_cost / float(duration), 4)
    else:
        report["cost_per_video_second_usd"] = None


def _write_csv(summary_path: Path, reports: list[dict[str, object]]) -> None:
    fieldnames = [
        "video_model_preset",
        "video_backend",
        "selected_scene_count",
        "wall_time_seconds",
        "peak_vram_mb_estimate",
        "duration_seconds",
        "width",
        "height",
        "fps",
        "size_bytes",
        "cost_estimate_usd",
        "cost_per_video_second_usd",
        "video_path",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            video_info = report.get("video_info", {}) or {}
            writer.writerow(
                {
                    "video_model_preset": report.get("video_model_preset"),
                    "video_backend": report.get("video_backend"),
                    "selected_scene_count": report.get("selected_scene_count"),
                    "wall_time_seconds": report.get("wall_time_seconds"),
                    "peak_vram_mb_estimate": report.get("peak_vram_mb_estimate"),
                    "duration_seconds": video_info.get("duration_seconds"),
                    "width": video_info.get("width"),
                    "height": video_info.get("height"),
                    "fps": video_info.get("fps"),
                    "size_bytes": video_info.get("size_bytes"),
                    "cost_estimate_usd": report.get("cost_estimate_usd"),
                    "cost_per_video_second_usd": report.get("cost_per_video_second_usd"),
                    "video_path": report.get("video_path"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-id")
    parser.add_argument("--preview-path")
    parser.add_argument("--scene-id", action="append", dest="scene_ids", required=True)
    parser.add_argument(
        "--preset",
        action="append",
        dest="presets",
        help="Preset to benchmark. Repeatable. Defaults to wan_t2v_1_3b.",
    )
    parser.add_argument("--output-dir", default="benchmark_runs")
    parser.add_argument("--hourly-cost", type=float, default=None)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()

    preview_path = _resolve_preview_path(args.preview_id, args.preview_path)
    presets = args.presets or ["wan_t2v_1_3b"]
    unknown = [preset for preset in presets if preset not in MODEL_PRESETS]
    if unknown:
        available = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown presets: {', '.join(unknown)}. Available: {available}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for preset in presets:
        print(f"[benchmark] Running preset: {preset}")
        report = _run_single_benchmark(
            preset_name=preset,
            preview_path=preview_path,
            scene_ids=args.scene_ids,
            output_dir=output_dir,
            poll_interval=max(0.25, args.poll_interval),
        )
        _add_cost_metrics(report, args.hourly_cost)
        reports.append(report)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_json = output_dir / f"benchmark_summary_{timestamp}.json"
    summary_csv = output_dir / f"benchmark_summary_{timestamp}.csv"

    summary_payload = {
        "preview_path": str(preview_path),
        "scene_ids": args.scene_ids,
        "presets": presets,
        "hourly_cost": args.hourly_cost,
        "reports": reports,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    _write_csv(summary_csv, reports)

    print(f"[benchmark] Summary JSON: {summary_json}")
    print(f"[benchmark] Summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
