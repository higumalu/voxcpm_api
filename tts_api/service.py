from __future__ import annotations

import io
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import soundfile as sf
from opencc import OpenCC
from scipy import signal

from tts_api.config import settings


@dataclass
class VoiceConfig:
    reference_wav_path: str
    reference_text: str


@dataclass
class InferConfig:
    seed: int
    cfg_value: float
    inference_timesteps: int
    pause_seconds: float = settings.default_pause_seconds


@dataclass
class SynthesisResult:
    job_id: str
    combined_audio: np.ndarray
    sample_rate: int
    duration_sec: float
    total_chars: int
    num_paragraphs: int
    processing_time_sec: float
    output_native_path: str | None = None
    output_16k_path: str | None = None


class TTSService:
    def __init__(
        self,
        model_loader: Callable[[], Any] | None = None,
        seed_setter: Callable[[int], None] | None = None,
    ) -> None:
        self._model = None
        self._cc = OpenCC("t2s")
        self._model_loader = model_loader or self._default_model_loader
        self._seed_setter = seed_setter or self._default_seed_setter
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    def _default_model_loader(self) -> Any:
        from voxcpm import VoxCPM

        return VoxCPM.from_pretrained(
            settings.model_name,
            load_denoiser=settings.load_denoiser,
            optimize=settings.optimize,
        )

    def _default_seed_setter(self, seed: int) -> None:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            return

    @property
    def is_model_ready(self) -> bool:
        return self._model is not None

    def get_or_load_model(self) -> Any:
        if self._model is None:
            self._model = self._model_loader()
        return self._model

    def normalize_paragraphs(self, text: str | None, paragraphs: list[str] | None) -> list[str]:
        if paragraphs:
            base = [p.strip() for p in paragraphs if p and p.strip()]
        else:
            base = [line.strip() for line in (text or "").splitlines() if line.strip()]
            if not base and text and text.strip():
                base = [text.strip()]
        return [self._cc.convert(p) for p in base if p]

    def synthesize_paragraphs(self, paragraphs: list[str], voice_config: VoiceConfig, infer_config: InferConfig) -> tuple[np.ndarray, int, int]:
        if not paragraphs:
            raise ValueError("paragraphs 不可為空")

        model = self.get_or_load_model()
        sample_rate = model.tts_model.sample_rate
        chunks: list[np.ndarray] = []
        total_chars = 0

        for index, paragraph in enumerate(paragraphs):
            self._seed_setter(infer_config.seed)
            wav = model.generate(
                text=paragraph,
                reference_wav_path=voice_config.reference_wav_path,
                prompt_wav_path=voice_config.reference_wav_path,
                prompt_text=voice_config.reference_text,
                cfg_value=infer_config.cfg_value,
                inference_timesteps=infer_config.inference_timesteps,
                normalize=True,
            )
            if wav.ndim == 2:
                wav = wav.mean(axis=0)
            chunks.append(wav.astype(np.float32))
            total_chars += len(paragraph)
            if index < len(paragraphs) - 1:
                chunks.append(np.zeros(int(infer_config.pause_seconds * sample_rate), dtype=np.float32))

        combined = self.concat_and_normalize(chunks)
        return combined, sample_rate, total_chars

    @staticmethod
    def concat_and_normalize(chunks: list[np.ndarray]) -> np.ndarray:
        if not chunks:
            raise ValueError("audio chunks 不可為空")
        combined = np.concatenate(chunks).astype(np.float32)
        peak = np.max(np.abs(combined)) if combined.size else 0
        if peak > 0:
            combined = combined / peak * 0.95
        return combined

    @staticmethod
    def resample_to_16k(audio: np.ndarray, source_sr: int) -> np.ndarray:
        if source_sr == 16000:
            return audio.astype(np.float32)
        target_length = int(len(audio) * 16000 / source_sr)
        return signal.resample(audio, target_length).astype(np.float32)

    def export_wav_variants(self, job_id: str, audio: np.ndarray, sample_rate: int) -> tuple[str, str]:
        ts = int(time.time())
        native_path = settings.output_dir / f"{job_id}_{ts}_native.wav"
        s16k_path = settings.output_dir / f"{job_id}_{ts}_16k.wav"

        sf.write(native_path, audio.astype(np.float32), sample_rate, subtype=settings.default_native_subtype)
        data_16k = self.resample_to_16k(audio, sample_rate)
        sf.write(s16k_path, data_16k, 16000, subtype=settings.default_native_subtype)
        return str(native_path), str(s16k_path)

    def synthesize(self, *, text: str | None, paragraphs: list[str] | None, voice_config: VoiceConfig, infer_config: InferConfig, write_files: bool = True) -> SynthesisResult:
        started = time.time()
        normalized_paragraphs = self.normalize_paragraphs(text, paragraphs)
        combined, sample_rate, total_chars = self.synthesize_paragraphs(normalized_paragraphs, voice_config, infer_config)
        duration_sec = len(combined) / sample_rate
        processing_time = time.time() - started
        job_id = uuid.uuid4().hex
        output_native_path = None
        output_16k_path = None
        if write_files:
            output_native_path, output_16k_path = self.export_wav_variants(job_id, combined, sample_rate)

        return SynthesisResult(
            job_id=job_id,
            combined_audio=combined,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            total_chars=total_chars,
            num_paragraphs=len(normalized_paragraphs),
            processing_time_sec=processing_time,
            output_native_path=output_native_path,
            output_16k_path=output_16k_path,
        )

    @staticmethod
    def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        sf.write(buffer, audio.astype(np.float32), sample_rate, format="WAV", subtype=settings.default_native_subtype)
        return buffer.getvalue()


def default_voice_config(reference_wav_path: str | None, reference_text: str | None) -> VoiceConfig:
    wav_path = reference_wav_path or str(settings.default_reference_wav_path)
    text = reference_text or settings.default_reference_text
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"reference_wav_path 不存在: {wav_path}")
    return VoiceConfig(reference_wav_path=wav_path, reference_text=text)

