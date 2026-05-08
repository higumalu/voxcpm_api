from __future__ import annotations

import numpy as np
import pytest

from tts_api.service import TTSService


@pytest.mark.unit
def test_concat_and_normalize():
    chunks = [np.array([0.2, -0.5], dtype=np.float32), np.array([0.3], dtype=np.float32)]
    out = TTSService.concat_and_normalize(chunks)
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 0.95 + 1e-6


@pytest.mark.unit
def test_resample_to_16k():
    audio = np.random.randn(24000).astype(np.float32)
    out = TTSService.resample_to_16k(audio, source_sr=24000)
    assert out.dtype == np.float32
    assert len(out) == 16000


@pytest.mark.unit
def test_normalize_paragraphs():
    service = TTSService(model_loader=lambda: None)
    paragraphs = service.normalize_paragraphs("第一行\n\n第二行", None)
    assert len(paragraphs) == 2

