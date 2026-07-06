from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("VOXCPM_MODEL_NAME", "openbmb/VoxCPM2")
    load_denoiser: bool = _bool_env("VOXCPM_LOAD_DENOISER", False)
    optimize: bool = _bool_env("VOXCPM_OPTIMIZE", False)
    device: str = os.getenv("VOXCPM_DEVICE", "auto")
    voices_dir: Path = Path(os.getenv("VOXCPM_VOICES_DIR", "./voices")).expanduser().resolve()
    default_cfg_value: float = 1.5
    default_inference_timesteps: int = 30
    api_base_url: str = os.getenv("VOXCPM_API_URL", "http://127.0.0.1:8000")
    default_voice_id: str = os.getenv("VOXCPM_DEFAULT_VOICE_ID", "")


settings = Settings()
