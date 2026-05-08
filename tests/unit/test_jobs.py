from __future__ import annotations

import pytest

from tts_api.jobs import JobStore


@pytest.mark.unit
@pytest.mark.asyncio
async def test_job_store_create_get_update():
    store = JobStore(ttl_seconds=10)
    rec = await store.create(payload={"text": "hello"})
    assert rec.status == "queued"

    loaded = await store.get(rec.job_id)
    assert loaded is not None

    updated = await store.update(rec.job_id, status="running", progress=0.5)
    assert updated is not None
    assert updated.status == "running"
    assert updated.progress == 0.5

