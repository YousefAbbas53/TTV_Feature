from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from model_profiles import resolve_model_preset


@dataclass(frozen=True)
class Settings:
    app_root: Path
    outputs_dir: Path
    temp_dir: Path
    wan_repo_dir: Path
    wan_headless_script: Path
    wan_template_path: Path
    huggingface_cache_dir: Path
    api_key: str | None
    video_model_preset: str
    video_backend: str
    default_model_type: str = "t2v_1.3B"
    default_width: int = 768
    default_height: int = 432
    default_num_frames: int = 80
    default_fps: int = 16
    default_steps: int = 40
    default_cfg: float = 4.8


def _resolve_path(env_name: str, fallback: Path) -> Path:
    raw = os.getenv(env_name)
    return Path(raw).expanduser().resolve() if raw else fallback.resolve()


def _resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def _get_int(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def _get_float(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


APP_ROOT = Path(__file__).resolve().parent
_preset = resolve_model_preset(os.getenv("VIDEO_MODEL_PRESET", "wan_t2v_1_3b"))
_default_wan_repo = _resolve_existing_path(APP_ROOT / "Wan2GP", APP_ROOT / "Wan2GP-main")

settings = Settings(
    app_root=APP_ROOT,
    outputs_dir=_resolve_path("VIDEO_OUTPUT_DIR", APP_ROOT / "outputs"),
    temp_dir=_resolve_path("VIDEO_TEMP_DIR", APP_ROOT / "tmp"),
    wan_repo_dir=_resolve_path("WAN_REPO_DIR", _default_wan_repo),
    wan_headless_script=_resolve_path(
        "WAN_HEADLESS_SCRIPT",
        _default_wan_repo / "wgp_headless_t2v_fix.py",
    ),
    wan_template_path=_resolve_path(
        "WAN_TEMPLATE_PATH",
        _default_wan_repo / "defaults" / "t2v.json",
    ),
    huggingface_cache_dir=_resolve_path("HF_HOME", APP_ROOT / "hf_cache"),
    api_key=os.getenv("VIDEO_API_KEY") or None,
    video_model_preset=_preset.name,
    video_backend=os.getenv("VIDEO_BACKEND", "wan").strip().lower(),
    default_model_type=os.getenv("VIDEO_MODEL_TYPE", _preset.model_type).strip() or "t2v_1.3B",
    default_width=_get_int("VIDEO_WIDTH", _preset.width),
    default_height=_get_int("VIDEO_HEIGHT", _preset.height),
    default_num_frames=_get_int("VIDEO_NUM_FRAMES", _preset.num_frames),
    default_fps=_get_int("VIDEO_FPS", _preset.fps),
    default_steps=_get_int("VIDEO_STEPS", _preset.steps),
    default_cfg=_get_float("VIDEO_CFG", _preset.cfg),
)
