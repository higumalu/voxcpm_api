from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from tts_api.config import settings


@dataclass
class JobRecord:
    job_id: str
    payload: dict[str, Any]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    progress: float = 0.0
    error_code: str | None = None
    error_message: str | None = None
    file_path: str | None = None
    file_path_16k: str | None = None
    metadata: dict[str, Any] | None = None


class JobStore:
    def __init__(self, ttl_seconds: int = settings.job_ttl_seconds) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()
        self.ttl_seconds = ttl_seconds

    async def create(self, payload: dict[str, Any]) -> JobRecord:
        async with self._lock:
            job_id = uuid.uuid4().hex
            rec = JobRecord(job_id=job_id, payload=payload)
            self._jobs[job_id] = rec
            return rec

    async def get(self, job_id: str) -> JobRecord | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job_id: str, **kwargs: Any) -> JobRecord | None:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            for key, value in kwargs.items():
                setattr(rec, key, value)
            rec.updated_at = time.time()
            return rec

    async def cleanup_expired(self) -> int:
        now = time.time()
        removed = 0
        async with self._lock:
            stale_ids = [job_id for job_id, rec in self._jobs.items() if now - rec.updated_at > self.ttl_seconds]
            for job_id in stale_ids:
                self._jobs.pop(job_id, None)
                removed += 1
        return removed

