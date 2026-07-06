from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ResponseFormat(str, Enum):
    audio = "audio"
    json = "json"


class VoiceCreateRequest(BaseModel):
    audio_base64: str = Field(..., min_length=1, description="BASE64 編碼的 WAV 檔內容")
    reference_text: str = Field(..., min_length=1, description="參考音檔對應的逐字稿")


class VoiceMetadata(BaseModel):
    voice_id: str
    reference_text: str
    sample_rate: int
    duration_sec: float
    created_at: float


class VoiceListResponse(BaseModel):
    voices: list[VoiceMetadata]


class TTSRequest(BaseModel):
    voice_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    cfg_value: float = Field(default=1.5, gt=0.0, le=5.0)
    inference_timesteps: int = Field(default=30, ge=1, le=200)
    response_format: ResponseFormat = ResponseFormat.audio


class TTSJsonResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_sec: float


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    voices_count: int
