from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from tts_api.app import app


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
def test_real_model_smoke():
    if os.getenv("RUN_GPU_E2E") != "1":
        pytest.skip("未設定 RUN_GPU_E2E=1，略過真模型測試")

    reference_wav_path = os.getenv("TTS_REFERENCE_WAV")
    if not reference_wav_path:
        pytest.skip("未設定 TTS_REFERENCE_WAV，略過真模型測試")

    client = TestClient(app)
    resp = client.post(
        "/v1/tts/sync",
        json={
            "text": "你好，這是 E2E 測試。",
            "delivery": "stream",
            "reference_wav_path": reference_wav_path,
            "reference_text": "你好，這是參考語音。",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert len(resp.content) > 0

