from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    app_root: Path
    outputs_dir: Path
    temp_dir: Path
    wan_repo_dir: Path
    wan_headless_script: Path
    wan_template_path: Path
    api_key: str | None
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


APP_ROOT = Path(__file__).resolve().parent

settings = Settings(
    app_root=APP_ROOT,
    outputs_dir=_resolve_path("VIDEO_OUTPUT_DIR", APP_ROOT / "outputs"),
    temp_dir=_resolve_path("VIDEO_TEMP_DIR", APP_ROOT / "tmp"),
    wan_repo_dir=_resolve_path("WAN_REPO_DIR", APP_ROOT / "Wan2GP"),
    wan_headless_script=_resolve_path(
        "WAN_HEADLESS_SCRIPT",
        APP_ROOT / "Wan2GP" / "wgp_headless_t2v_fix.py",
    ),
    wan_template_path=_resolve_path(
        "WAN_TEMPLATE_PATH",
        APP_ROOT / "Wan2GP" / "defaults" / "t2v.json",
    ),
    api_key=os.getenv("VIDEO_API_KEY") or None,
)
