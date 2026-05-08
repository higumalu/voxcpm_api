from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import tts_api.app as app_module


@pytest.mark.integration
def test_sync_stream(monkeypatch, reference_wav):
    def fake_synthesize(**kwargs):
        return SimpleNamespace(
            job_id="job-sync-stream",
            combined_audio=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
            duration_sec=0.1,
            total_chars=2,
            num_paragraphs=1,
            processing_time_sec=0.05,
            output_native_path=None,
            output_16k_path=None,
        )

    monkeypatch.setattr(app_module.service, "synthesize", fake_synthesize)
    client = TestClient(app_module.app)
    response = client.post(
        "/v1/tts/sync",
        json={
            "text": "你好",
            "delivery": "stream",
            "reference_wav_path": reference_wav,
            "reference_text": "hello",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert len(response.content) > 0


@pytest.mark.integration
def test_sync_file(monkeypatch, tmp_path, reference_wav):
    native = tmp_path / "a.wav"
    s16k = tmp_path / "b.wav"
    native.write_bytes(b"wav")
    s16k.write_bytes(b"wav")

    def fake_synthesize(**kwargs):
        return SimpleNamespace(
            job_id="job-sync-file",
            combined_audio=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
            duration_sec=0.1,
            total_chars=2,
            num_paragraphs=1,
            processing_time_sec=0.05,
            output_native_path=str(native),
            output_16k_path=str(s16k),
        )

    monkeypatch.setattr(app_module.service, "synthesize", fake_synthesize)
    client = TestClient(app_module.app)
    response = client.post(
        "/v1/tts/sync",
        json={
            "text": "你好",
            "delivery": "file",
            "reference_wav_path": reference_wav,
            "reference_text": "hello",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "succeeded"
    assert "download_url" in payload

