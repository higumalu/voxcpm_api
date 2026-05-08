from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import tts_api.app as app_module


@pytest.mark.integration
@pytest.mark.slow
def test_sync_latency_budget_with_mock(monkeypatch, reference_wav):
    def fake_synthesize(**kwargs):
        return SimpleNamespace(
            job_id="job-perf-sync",
            combined_audio=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
            duration_sec=0.1,
            total_chars=12,
            num_paragraphs=1,
            processing_time_sec=0.05,
            output_native_path=None,
            output_16k_path=None,
        )

    monkeypatch.setattr(app_module.service, "synthesize", fake_synthesize)
    client = TestClient(app_module.app)

    start = time.time()
    response = client.post(
        "/v1/tts/sync",
        json={
            "text": "短文本測試",
            "delivery": "stream",
            "reference_wav_path": reference_wav,
            "reference_text": "hello",
        },
    )
    elapsed = time.time() - start
    assert response.status_code == 200
    assert elapsed < 8.0


@pytest.mark.integration
@pytest.mark.slow
def test_async_queue_20_jobs_no_crash(monkeypatch, tmp_path, reference_wav):
    native = tmp_path / "job_native.wav"
    s16k = tmp_path / "job_16k.wav"
    native.write_bytes(b"wav")
    s16k.write_bytes(b"wav")

    def fake_synthesize(**kwargs):
        return SimpleNamespace(
            job_id="job-perf-async",
            combined_audio=np.zeros(1600, dtype=np.float32),
            sample_rate=16000,
            duration_sec=0.1,
            total_chars=12,
            num_paragraphs=1,
            processing_time_sec=0.05,
            output_native_path=str(native),
            output_16k_path=str(s16k),
        )

    monkeypatch.setattr(app_module.service, "synthesize", fake_synthesize)
    with TestClient(app_module.app) as client:
        job_ids: list[str] = []
        for _ in range(20):
            response = client.post(
                "/v1/tts/jobs",
                json={
                    "text": "排隊測試",
                    "delivery": "file",
                    "reference_wav_path": reference_wav,
                    "reference_text": "hello",
                },
            )
            assert response.status_code == 200
            job_ids.append(response.json()["job_id"])

        deadline = time.time() + 6
        while time.time() < deadline:
            statuses = [client.get(f"/v1/tts/jobs/{job_id}").json()["status"] for job_id in job_ids]
            if all(status == "succeeded" for status in statuses):
                break
            time.sleep(0.05)

        statuses = [client.get(f"/v1/tts/jobs/{job_id}").json()["status"] for job_id in job_ids]
        assert all(status == "succeeded" for status in statuses)

