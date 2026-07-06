#!/usr/bin/env python3
"""以變數控制聲線、文字與輸出路徑，直接載入 VoxCPM 模型產生語音（不經過 API 服務）。"""

from __future__ import annotations

import sys
import os

import soundfile as sf

# --- 在這裡修改 ---
REFERENCE_WAV_PATH = None   # "ref.wav"                 # 參考音檔（決定聲線）
REFERENCE_TEXT = None       # "你好，這是參考語音。"      # 參考音檔的逐字稿
TEXT_PROMPT = "大便的離去，到底是馬桶的追求，還是肛門的不挽留？"
VOICE_PROMPT = "(A young woman, slow, gentle and sweet voice)"
TEXT = f"{VOICE_PROMPT} {TEXT_PROMPT}"
OUTPUT_PATH = "scripts/output/generated.wav"

MODEL_NAME = "openbmb/VoxCPM2"
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 30

# ----------------


def main() -> int:
    if not TEXT.strip():
        print("error: TEXT 不可為空", file=sys.stderr)
        return 1

    from voxcpm import VoxCPM
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"loading model: {MODEL_NAME}", file=sys.stderr)
    model = VoxCPM.from_pretrained(MODEL_NAME, load_denoiser=False)

    wav = model.generate(
        text=TEXT,
        reference_wav_path=REFERENCE_WAV_PATH,
        prompt_wav_path=REFERENCE_WAV_PATH,
        prompt_text=REFERENCE_TEXT,
        cfg_value=CFG_VALUE,
        inference_timesteps=INFERENCE_TIMESTEPS,
        normalize=True,
    )

    sf.write(OUTPUT_PATH, wav, model.tts_model.sample_rate)
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
