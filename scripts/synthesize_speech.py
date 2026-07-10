#!/usr/bin/env python3
"""以變數控制聲線、文字與輸出路徑，直接載入 VoxCPM 模型產生語音（不經過 API 服務）。"""

from __future__ import annotations

import sys
import os
from datetime import datetime
from pathlib import Path

from numpy import True_
import soundfile as sf

_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


# --- 在這裡修改 ---
REFERENCE_WAV_PATH = None #"scripts/ref/ref.wav"
REFERENCE_PROMPT_WAV_PATH = None   # "ref.wav"                 # 參考音檔（決定聲線）
REFERENCE_PROMPT_TEXT = None       # "你好，這是參考語音。"      # 參考音檔的逐字稿
TEXT_PROMPT = _load_prompt(_SCRIPT_DIR / "prompt" / "text_prompt.txt")
VOICE_PROMPT = _load_prompt(_SCRIPT_DIR / "prompt" / "voice_prompt.txt")
TEXT = f"({VOICE_PROMPT}) {TEXT_PROMPT}".strip()
OUTPUT_DIR = "scripts/output/"

MODEL_NAME = "openbmb/VoxCPM2"
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 30
batch_number = 5

# ----------------


def main() -> int:
    if not TEXT.strip():
        print("error: TEXT 不可為空", file=sys.stderr)
        return 1

    from voxcpm_api.cuda_compat import patch_safetensors_cuda_loading

    patch_safetensors_cuda_loading()
    from voxcpm import VoxCPM

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"loading model: {MODEL_NAME}", file=sys.stderr)
    model = VoxCPM.from_pretrained(MODEL_NAME, load_denoiser=False, optimize=False)

    
    for i in range(batch_number):
        wav = model.generate(
            text=TEXT,
            reference_wav_path=REFERENCE_WAV_PATH,
            prompt_wav_path=REFERENCE_PROMPT_WAV_PATH,
            prompt_text=REFERENCE_PROMPT_TEXT,
            cfg_value=CFG_VALUE,
            inference_timesteps=INFERENCE_TIMESTEPS,
            normalize=True,
            denoise=False,
        )

        output_path = os.path.join(
            OUTPUT_DIR, f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        sf.write(output_path, wav, model.tts_model.sample_rate)
        print(f"wrote {output_path}")
        print(f"synthesized batch {i + 1} of {batch_number} successfully")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
