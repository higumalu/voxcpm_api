from __future__ import annotations

import pytest
from pydantic import ValidationError

from tts_api.models import TTSRequest


@pytest.mark.unit
def test_tts_request_accepts_text():
    req = TTSRequest(text="你好")
    assert req.text == "你好"


@pytest.mark.unit
def test_tts_request_accepts_paragraphs():
    req = TTSRequest(paragraphs=["第一段", "第二段"])
    assert len(req.paragraphs or []) == 2


@pytest.mark.unit
def test_tts_request_requires_text_or_paragraphs():
    with pytest.raises(ValidationError):
        TTSRequest()

