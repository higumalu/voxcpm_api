from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import tts_api.app as app_module


@pytest.mark.integration
def test_async_job_flow(monkeypatch, tmp_path, reference_wav):
    native = tmp_path / "job_native.wav"
    s16k = tmp_path / "job_16k.wav"
    native.write_bytes(b"wav")
    s16k.write_bytes(b"wav")

    def fake_synthesize(**kwargs):
        return SimpleNamespace(
            job_id="job-async",
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
    with TestClient(app_module.app) as client:
        create_resp = client.post(
            "/v1/tts/jobs",
            json={
                "text": "你好",
                "delivery": "file",
                "reference_wav_path": reference_wav,
                "reference_text": "hello",
            },
        )
        assert create_resp.status_code == 200
        job_id = create_resp.json()["job_id"]

        deadline = time.time() + 3
        status_payload = {}
        while time.time() < deadline:
            status_resp = client.get(f"/v1/tts/jobs/{job_id}")
            assert status_resp.status_code == 200
            status_payload = status_resp.json()
            if status_payload["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)

        assert status_payload["status"] == "succeeded"
        audio_resp = client.get(f"/v1/tts/jobs/{job_id}/audio")
        assert audio_resp.status_code == 200
        assert audio_resp.headers["content-type"].startswith("audio/wav")

