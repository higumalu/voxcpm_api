from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_name: str = "openbmb/VoxCPM2"
    load_denoiser: bool = False
    optimize: bool = False
    default_seed: int = 42
    default_cfg_value: float = 1.5
    default_inference_timesteps: int = 30
    default_pause_seconds: float = 0.3
    default_native_subtype: str = "PCM_16"
    output_dir: Path = Path(os.getenv("VOXCPM_OUTPUT_DIR", "./tts_outputs")).expanduser().resolve()
    voices_dir: Path = Path(os.getenv("VOXCPM_VOICES_DIR", "./voices")).expanduser().resolve()
    default_reference_filename: str = "voice_design_female_v1.wav"
    default_reference_text: str = (
        "嘿，大家好，欢迎回到《不期而遇》。我是你们的朋友小夏。是忙碌的一周过去了，"
        "今天想跟大家聊聊那个困扰我们很久的话题。希望接下来的这三十分钟，能给你带来一些温柔的力量。我们开始吧。"
    )
    job_ttl_seconds: int = 7200
    max_concurrent_gpu_jobs: int = 1

    @property
    def default_reference_wav_path(self) -> Path:
        return self.voices_dir / self.default_reference_filename


settings = Settings()

