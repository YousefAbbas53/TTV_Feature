from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    name: str
    description: str
    model_type: str
    width: int
    height: int
    num_frames: int
    fps: int
    steps: int
    cfg: float


MODEL_PRESETS: dict[str, ModelPreset] = {
    "wan_t2v_1_3b": ModelPreset(
        name="wan_t2v_1_3b",
        description="Kaggle baseline Wan text-to-video profile that previously produced the best results.",
        model_type="t2v_1.3B",
        width=768,
        height=432,
        num_frames=80,
        fps=16,
        steps=40,
        cfg=4.8,
    ),
}


def resolve_model_preset(name: str | None) -> ModelPreset:
    preset_name = (name or "wan_t2v_1_3b").strip().lower()
    preset = MODEL_PRESETS.get(preset_name)
    if preset is None:
        available = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown VIDEO_MODEL_PRESET '{preset_name}'. Available: {available}")
    return preset
