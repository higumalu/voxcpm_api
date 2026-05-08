# VoxCPM Podcast TTS API

## 啟動

```bash
cd voxcpm_api
uv sync --group dev
uv run uvicorn tts_api.app:app --host 0.0.0.0 --port 8000
```

## 環境變數（避免機器綁定）

```bash
export VOXCPM_VOICES_DIR="./voices"
export VOXCPM_OUTPUT_DIR="./tts_outputs"
```

## API 範例

### 健康檢查

```bash
curl -s http://127.0.0.1:8000/v1/health
```

### 同步（串流回傳）

```bash
curl -X POST http://127.0.0.1:8000/v1/tts/sync \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，這是同步串流測試。",
    "delivery": "stream",
    "reference_wav_path": "/abs/path/ref.wav",
    "reference_text": "你好，這是參考語音。"
  }' \
  --output sync.wav
```

### 同步（檔案交付）

```bash
curl -X POST http://127.0.0.1:8000/v1/tts/sync \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，這是 file 模式測試。",
    "delivery": "file",
    "reference_wav_path": "/abs/path/ref.wav",
    "reference_text": "你好，這是參考語音。"
  }'
```

### 非同步任務

```bash
curl -X POST http://127.0.0.1:8000/v1/tts/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，這是非同步測試。",
    "delivery": "file",
    "reference_wav_path": "/abs/path/ref.wav",
    "reference_text": "你好，這是參考語音。"
  }'
```

```bash
curl -s http://127.0.0.1:8000/v1/tts/jobs/<job_id>
```

```bash
curl -L "http://127.0.0.1:8000/v1/tts/jobs/<job_id>/audio?sample_rate=16k" --output job_16k.wav
```

## 測試

```bash
cd voxcpm_api
uv run pytest tests -m "not gpu"
```

```bash
cd voxcpm_api
uv run pytest tests -m gpu
```

