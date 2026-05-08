from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class DeliveryMode(str, Enum):
    stream = "stream"
    file = "file"


class SampleRateMode(str, Enum):
    native = "native"
    s16k = "16k"


class TTSRequest(BaseModel):
    text: Optional[str] = None
    paragraphs: Optional[list[str]] = None
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    cfg_value: float = Field(default=1.5, gt=0.0, le=5.0)
    inference_timesteps: int = Field(default=30, ge=1, le=200)
    reference_wav_path: Optional[str] = None
    reference_text: Optional[str] = None
    output_sample_rates: list[SampleRateMode] = Field(default_factory=lambda: [SampleRateMode.native, SampleRateMode.s16k])
    delivery: DeliveryMode = DeliveryMode.file

    @model_validator(mode="after")
    def validate_text_or_paragraphs(self) -> "TTSRequest":
        has_text = bool(self.text and self.text.strip())
        has_paragraphs = bool(self.paragraphs and any(p.strip() for p in self.paragraphs))
        if not has_text and not has_paragraphs:
            raise ValueError("必須提供 text 或 paragraphs 其中之一。")
        return self


class TTSMetadata(BaseModel):
    duration_sec: float
    num_paragraphs: int
    total_chars: int
    sample_rate: int
    processing_time_sec: float


class SyncFileResponse(BaseModel):
    job_id: str
    status: str
    file_path: str
    file_path_16k: Optional[str] = None
    download_url: str
    metadata: TTSMetadata


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: float
    updated_at: float
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    file_path_16k: Optional[str] = None
    metadata: Optional[TTSMetadata] = None

