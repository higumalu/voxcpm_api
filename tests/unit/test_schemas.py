from __future__ import annotations

import pytest
from pydantic import ValidationError

from voxcpm_api.schemas import ResponseFormat, TTSRequest, VoiceCreateRequest


@pytest.mark.unit
def test_voice_create_request_requires_fields() -> None:
    with pytest.raises(ValidationError):
        VoiceCreateRequest(audio_base64="", reference_text="ok")
    with pytest.raises(ValidationError):
        VoiceCreateRequest(audio_base64="abc", reference_text="")


@pytest.mark.unit
def test_tts_request_defaults() -> None:
    req = TTSRequest(voice_id="abc", text="hello")
    assert req.cfg_value == 1.5
    assert req.inference_timesteps == 30
    assert req.response_format == ResponseFormat.audio


@pytest.mark.unit
@pytest.mark.parametrize("cfg_value", [0.0, -1.0, 5.1])
def test_tts_request_cfg_value_bounds(cfg_value: float) -> None:
    with pytest.raises(ValidationError):
        TTSRequest(voice_id="abc", text="hello", cfg_value=cfg_value)


@pytest.mark.unit
@pytest.mark.parametrize("steps", [0, -1, 201])
def test_tts_request_steps_bounds(steps: int) -> None:
    with pytest.raises(ValidationError):
        TTSRequest(voice_id="abc", text="hello", inference_timesteps=steps)


@pytest.mark.unit
def test_tts_request_requires_text_and_voice_id() -> None:
    with pytest.raises(ValidationError):
        TTSRequest(voice_id="", text="hello")
    with pytest.raises(ValidationError):
        TTSRequest(voice_id="abc", text="")
