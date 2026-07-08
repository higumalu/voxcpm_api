from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from voxcpm_api.config import settings


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration_sec: float


class TTSEngine:
    """Thin wrapper around VoxCPM with lazy model loading."""

    def __init__(self, model_loader: Callable[[], Any] | None = None) -> None:
        self._model: Any | None = None
        self._lock = threading.Lock()
        self._model_loader = model_loader or self._default_model_loader

    @staticmethod
    def _default_model_loader() -> Any:
        from voxcpm_api.cuda_compat import patch_safetensors_cuda_loading

        patch_safetensors_cuda_loading()
        from voxcpm import VoxCPM

        patch_safetensors_cuda_loading()
        return VoxCPM.from_pretrained(
            settings.model_name,
            load_denoiser=settings.load_denoiser,
            optimize=settings.optimize,
        )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._model_loader()
        return self._model

    def generate(
        self,
        *,
        text: str,
        reference_wav_path: Path | str,
        reference_text: str,
        cfg_value: float,
        inference_timesteps: int,
    ) -> SynthesisResult:
        if not text or not text.strip():
            raise ValueError("text 不可為空")

        model = self._get_model()
        ref_path = str(reference_wav_path)
        with self._lock:
            wav = model.generate(
                text=text,
                reference_wav_path=ref_path,
                prompt_wav_path=ref_path,
                prompt_text=reference_text,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=True,
            )

        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=0)

        sample_rate = int(model.tts_model.sample_rate)
        duration_sec = float(len(wav) / sample_rate) if sample_rate else 0.0
        return SynthesisResult(audio=wav, sample_rate=sample_rate, duration_sec=duration_sec)
