from __future__ import annotations

from fastapi import APIRouter, Depends

from voxcpm_api.dependencies import get_tts_engine, get_voice_store
from voxcpm_api.schemas import HealthResponse
from voxcpm_api.tts_engine import TTSEngine
from voxcpm_api.voice_store import VoiceStore

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    store: VoiceStore = Depends(get_voice_store),
    engine: TTSEngine = Depends(get_tts_engine),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_ready=engine.is_ready,
        voices_count=store.count(),
    )
