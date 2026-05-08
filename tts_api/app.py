from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

from tts_api.config import settings
from tts_api.jobs import JobStore
from tts_api.models import (
    DeliveryMode,
    JobCreateResponse,
    JobStatusResponse,
    SampleRateMode,
    SyncFileResponse,
    TTSMetadata,
    TTSRequest,
)
from tts_api.service import InferConfig, TTSService, default_voice_config

app = FastAPI(title="VoxCPM Podcast TTS API", version="0.1.0")
service = TTSService()
job_store = JobStore()
gpu_semaphore = asyncio.Semaphore(settings.max_concurrent_gpu_jobs)


def _metadata_from_result(result: Any) -> TTSMetadata:
    return TTSMetadata(
        duration_sec=result.duration_sec,
        num_paragraphs=result.num_paragraphs,
        total_chars=result.total_chars,
        sample_rate=result.sample_rate,
        processing_time_sec=result.processing_time_sec,
    )


def _build_download_url(job_id: str, sample_rate_mode: SampleRateMode = SampleRateMode.native) -> str:
    return f"/v1/tts/jobs/{job_id}/audio?sample_rate={sample_rate_mode.value}"


@app.on_event("startup")
async def startup_event() -> None:
    app.state.job_queue = asyncio.Queue()
    app.state.worker_task = asyncio.create_task(job_worker(app.state.job_queue))


@app.on_event("shutdown")
async def shutdown_event() -> None:
    worker_task = getattr(app.state, "worker_task", None)
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


@app.get("/v1/health")
def health() -> dict[str, Any]:
    queue = getattr(app.state, "job_queue", None)
    return {
        "status": "ok",
        "model_ready": service.is_model_ready,
        "queue_size": queue.qsize() if queue else 0,
    }


@app.post("/v1/tts/sync")
async def tts_sync(payload: TTSRequest):
    if payload.delivery not in {DeliveryMode.stream, DeliveryMode.file}:
        raise HTTPException(status_code=400, detail="delivery 只支援 stream/file")

    async with gpu_semaphore:
        try:
            voice_cfg = default_voice_config(payload.reference_wav_path, payload.reference_text)
            infer_cfg = InferConfig(
                seed=payload.seed,
                cfg_value=payload.cfg_value,
                inference_timesteps=payload.inference_timesteps,
            )
            result = await asyncio.to_thread(
                service.synthesize,
                text=payload.text,
                paragraphs=payload.paragraphs,
                voice_config=voice_cfg,
                infer_config=infer_cfg,
                write_files=(payload.delivery == DeliveryMode.file),
            )
        except FileNotFoundError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except Exception as err:
            raise HTTPException(status_code=500, detail=f"TTS 推論失敗: {err}") from err

    if payload.delivery == DeliveryMode.stream:
        wav_bytes = service.to_wav_bytes(result.combined_audio, result.sample_rate)
        return Response(content=wav_bytes, media_type="audio/wav")

    if not result.output_native_path:
        raise HTTPException(status_code=500, detail="未產生輸出檔案")

    metadata = _metadata_from_result(result)
    return SyncFileResponse(
        job_id=result.job_id,
        status="succeeded",
        file_path=result.output_native_path,
        file_path_16k=result.output_16k_path,
        download_url=_build_download_url(result.job_id),
        metadata=metadata,
    )


@app.post("/v1/tts/jobs", response_model=JobCreateResponse)
async def create_job(payload: TTSRequest) -> JobCreateResponse:
    payload_dict = payload.model_dump(mode="json")
    rec = await job_store.create(payload=payload_dict)
    queue = getattr(app.state, "job_queue", None)
    if queue is None:
        raise HTTPException(status_code=503, detail="job worker 尚未啟動")
    await queue.put(rec.job_id)
    return JobCreateResponse(job_id=rec.job_id, status=rec.status)


@app.get("/v1/tts/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    rec = await job_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job_id 不存在")
    return JobStatusResponse(
        job_id=rec.job_id,
        status=rec.status,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        progress=rec.progress,
        error_code=rec.error_code,
        error_message=rec.error_message,
        file_path=rec.file_path,
        file_path_16k=rec.file_path_16k,
        metadata=rec.metadata,
    )


@app.get("/v1/tts/jobs/{job_id}/audio")
async def download_job_audio(job_id: str, sample_rate: SampleRateMode = Query(default=SampleRateMode.native)):
    rec = await job_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job_id 不存在")
    if rec.status != "succeeded":
        raise HTTPException(status_code=409, detail=f"任務尚未完成，目前狀態: {rec.status}")

    target = rec.file_path_16k if sample_rate == SampleRateMode.s16k else rec.file_path
    if not target or not Path(target).exists():
        raise HTTPException(status_code=404, detail="音檔不存在")
    return FileResponse(path=target, media_type="audio/wav", filename=os.path.basename(target))


async def job_worker(queue: asyncio.Queue[str]) -> None:
    while True:
        job_id = await queue.get()
        rec = await job_store.get(job_id)
        if rec is None:
            queue.task_done()
            continue

        await job_store.update(job_id, status="running", progress=0.1)
        try:
            payload = TTSRequest(**rec.payload)
            voice_cfg = default_voice_config(payload.reference_wav_path, payload.reference_text)
            infer_cfg = InferConfig(
                seed=payload.seed,
                cfg_value=payload.cfg_value,
                inference_timesteps=payload.inference_timesteps,
            )
            async with gpu_semaphore:
                result = await asyncio.to_thread(
                    service.synthesize,
                    text=payload.text,
                    paragraphs=payload.paragraphs,
                    voice_config=voice_cfg,
                    infer_config=infer_cfg,
                    write_files=True,
                )
            await job_store.update(
                job_id,
                status="succeeded",
                progress=1.0,
                file_path=result.output_native_path,
                file_path_16k=result.output_16k_path,
                metadata=_metadata_from_result(result).model_dump(),
            )
        except FileNotFoundError as err:
            await job_store.update(
                job_id,
                status="failed",
                progress=1.0,
                error_code="REFERENCE_WAV_NOT_FOUND",
                error_message=str(err),
            )
        except Exception as err:
            await job_store.update(
                job_id,
                status="failed",
                progress=1.0,
                error_code="INFERENCE_ERROR",
                error_message=str(err),
            )
        finally:
            await job_store.cleanup_expired()
            queue.task_done()

