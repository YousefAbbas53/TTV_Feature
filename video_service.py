from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import copy
import glob
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
import uuid

from config import Settings


NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, out of focus, jpeg artifacts, "
    "compression artifacts, flicker, jitter, strobing, temporal inconsistency, "
    "ghosting, frame blending, shaky camera, warped motion, morphing, deformed, "
    "bad anatomy, bad hands, extra fingers, extra limbs, text, watermark, logo, subtitles"
)


class VideoGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoResult:
    video_path: str
    filename: str
    run_dir: str


class VideoGenerator:
    _wan_render_lock = threading.Lock()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        self.settings.huggingface_cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        fps: int | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        image_start: str | None = None,
        mode: str = "t2v",
    ) -> VideoResult:
        clean_prompt = self._clean_text(prompt)
        if not clean_prompt:
            raise VideoGenerationError("Prompt must not be empty.")

        backend = self.settings.video_backend
        if backend == "wan":
            return self._generate_wan(
                prompt=clean_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                steps=steps,
                cfg=cfg,
                seed=seed,
                image_start=image_start,
                mode=mode,
            )
        if backend == "hunyuan":
            return self._generate_hunyuan(
                prompt=clean_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps,
                steps=steps,
                cfg=cfg,
                seed=seed,
            )
        raise VideoGenerationError(f"Unsupported VIDEO_BACKEND '{backend}'.")

    def _generate_wan(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        width: int | None,
        height: int | None,
        num_frames: int | None,
        fps: int | None,
        steps: int | None,
        cfg: float | None,
        seed: int | None,
        image_start: str | None = None,
        mode: str = "t2v",
    ) -> VideoResult:
        self._validate_wan_runtime()

        request_id = uuid.uuid4().hex
        run_dir = self.settings.temp_dir / f"run_{request_id}"
        settings_dir = run_dir / "settings"
        render_dir = run_dir / "rendered"
        outputs_link = self.settings.wan_repo_dir / "outputs"

        settings_dir.mkdir(parents=True, exist_ok=True)
        render_dir.mkdir(parents=True, exist_ok=True)

        runtime_settings = self._build_wan_runtime_settings(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=seed,
            image_start=image_start,
            mode=mode,
        )

        settings_path = settings_dir / f"{request_id}.json"
        with settings_path.open("w", encoding="utf-8") as handle:
            json.dump(runtime_settings, handle, ensure_ascii=False, indent=2)

        with self._wan_renderer_session(outputs_link, run_dir / "outputs"):
            rendered_path = self._run_wan_renderer(settings_path, run_dir / "outputs")

        final_name = f"{request_id}.mp4"
        final_path = self.settings.outputs_dir / final_name
        shutil.move(str(rendered_path), final_path)

        return VideoResult(
            video_path=str(final_path),
            filename=final_name,
            run_dir=str(run_dir),
        )

    def _validate_wan_runtime(self) -> None:
        if not self.settings.wan_repo_dir.exists():
            raise VideoGenerationError(
                "Wan2GP repository is missing. Set WAN_REPO_DIR to an existing checkout."
            )
        if not self.settings.wan_headless_script.exists():
            raise VideoGenerationError(
                "Headless renderer script is missing. Set WAN_HEADLESS_SCRIPT correctly."
            )
        if not self.settings.wan_template_path.exists():
            raise VideoGenerationError(
                "Template file is missing. Set WAN_TEMPLATE_PATH to Wan2GP defaults/t2v.json."
            )

    def _build_wan_runtime_settings(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        width: int | None,
        height: int | None,
        num_frames: int | None,
        fps: int | None,
        steps: int | None,
        cfg: float | None,
        seed: int | None,
        image_start: str | None = None,
        mode: str = "t2v",
    ) -> dict:
        # Use i2v template when image_start is provided
        if mode == "i2v" and image_start:
            i2v_template = self.settings.wan_repo_dir / "defaults" / "i2v.json"
            template_path = i2v_template if i2v_template.exists() else self.settings.wan_template_path
        else:
            template_path = self.settings.wan_template_path

        with template_path.open("r", encoding="utf-8") as handle:
            base = json.load(handle)

        payload = copy.deepcopy(base) if isinstance(base, dict) else {}
        render_seed = seed if seed is not None else random.randint(1, 2**31 - 1)

        # Use i2v model type when doing image-to-video
        model_type = "i2v" if (mode == "i2v" and image_start) else self.settings.default_model_type

        runtime = {
            "model_type": model_type,
            "base_model_type": model_type,
            "mode": mode,
            "seed": int(render_seed),
            "num_inference_steps": int(steps or self.settings.default_steps),
            "guidance_scale": float(cfg or self.settings.default_cfg),
            "cfg_scale": float(cfg or self.settings.default_cfg),
            "width": int(width or self.settings.default_width),
            "height": int(height or self.settings.default_height),
            "video_length": int(num_frames or self.settings.default_num_frames),
            "fps": int(fps or self.settings.default_fps),
        }

        # Inject anchor image for I2V
        if mode == "i2v" and image_start:
            runtime["image_start"] = str(Path(image_start).resolve())
            runtime["image_prompt_type"] = "S"
        runtime["resolution"] = f"{runtime['width']}x{runtime['height']}"
        runtime["video_width"] = runtime["width"]
        runtime["video_height"] = runtime["height"]
        runtime["image_width"] = runtime["width"]
        runtime["image_height"] = runtime["height"]
        runtime["num_frames"] = runtime["video_length"]
        runtime["frames"] = runtime["video_length"]
        runtime["num_steps"] = runtime["num_inference_steps"]
        runtime["steps"] = runtime["num_inference_steps"]

        # Wan2GP rejects sliding windows for text-to-video because each window
        # cannot see prior frames. Keep the window equal to the full clip.
        runtime["use_sliding_window"] = False
        runtime["sliding_window_size"] = runtime["video_length"]
        runtime["sliding_window_stride"] = max(1, runtime["video_length"] - 1)
        runtime["sliding_window_overlap"] = 1
        runtime["sliding_window_overlap_noise"] = 0.0
        runtime["sliding_window_discard_last_frames"] = 0

        prompt_value = prompt[:1200]
        negative_value = self._clean_text(negative_prompt) or NEGATIVE_PROMPT

        for key in ("prompt", "positive_prompt", "text", "caption"):
            payload[key] = prompt_value
        for key in ("negative_prompt", "neg_prompt", "negative", "n_prompt"):
            payload[key] = negative_value
        payload.update(runtime)

        for nested_key in ("params", "inputs", "override_params", "override_inputs"):
            nested = payload.get(nested_key)
            if not isinstance(nested, dict):
                nested = {}
                payload[nested_key] = nested
            for key in ("prompt", "positive_prompt", "text", "caption"):
                nested[key] = prompt_value
            for key in ("negative_prompt", "neg_prompt", "negative", "n_prompt"):
                nested[key] = negative_value
            nested.update(runtime)

        return payload

    def _prepare_wan_outputs_link(self, outputs_link: Path, run_outputs_dir: Path) -> None:
        run_outputs_dir.mkdir(parents=True, exist_ok=True)
        try:
            if outputs_link.is_symlink() or outputs_link.is_file():
                outputs_link.unlink()
            elif outputs_link.exists():
                shutil.rmtree(outputs_link)
            os.symlink(run_outputs_dir, outputs_link, target_is_directory=True)
        except OSError:
            if outputs_link.exists():
                shutil.rmtree(outputs_link)
            shutil.copytree(run_outputs_dir, outputs_link, dirs_exist_ok=True)

    @contextmanager
    def _wan_renderer_session(self, outputs_link: Path, run_outputs_dir: Path):
        with self._wan_render_lock:
            self._prepare_wan_outputs_link(outputs_link, run_outputs_dir)
            yield

    def _run_wan_renderer(self, settings_path: Path, output_dir: Path) -> Path:
        start_time = time.time()
        proc = subprocess.run(
            ["python", str(self.settings.wan_headless_script), "--process", str(settings_path)],
            cwd=self.settings.wan_repo_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            raise VideoGenerationError(
                "Renderer failed with exit code "
                f"{proc.returncode}. Output:\n{proc.stdout[-4000:]}"
            )

        rendered = self._detect_latest_mp4(start_time, output_dir)
        if rendered is None:
            raise VideoGenerationError("Renderer finished but no MP4 was produced.")
        return rendered

    def _detect_latest_mp4(self, start_time: float, output_dir: Path) -> Path | None:
        matches: list[Path] = []
        for raw in glob.glob(str(output_dir / "**" / "*.mp4"), recursive=True):
            candidate = Path(raw)
            try:
                if candidate.stat().st_mtime >= start_time:
                    matches.append(candidate)
            except OSError:
                continue
        if not matches:
            return None
        matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return matches[0]

    @staticmethod
    def _clean_text(value: str | None) -> str:
        collapsed = re.sub(r"\s+", " ", (value or "").replace("\r", " ").replace("\n", " "))
        return collapsed.strip()
