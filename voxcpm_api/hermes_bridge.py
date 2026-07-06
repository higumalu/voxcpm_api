#!/usr/bin/env python3
"""Hermes Agent command-type TTS bridge for the VoxCPM API service.

Hermes writes input text to a temp file and runs this script with placeholders:

    voxcpm-hermes-bridge --text-file {input_path} --out {output_path}

Environment variables:
    VOXCPM_API_URL          API base URL (default: http://127.0.0.1:8000)
    VOXCPM_DEFAULT_VOICE_ID Required voice_id registered via POST /v1/voices
    VOXCPM_CFG_VALUE        Optional cfg_value override (default: 1.5)
    VOXCPM_INFERENCE_TIMESTEPS  Optional timesteps override (default: 30)
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes → VoxCPM API TTS bridge")
    parser.add_argument("--text-file", required=True, help="UTF-8 text file from Hermes")
    parser.add_argument("--out", required=True, help="Output audio path")
    parser.add_argument("--voice-id", default=os.getenv("VOXCPM_DEFAULT_VOICE_ID", ""))
    parser.add_argument("--api-url", default=os.getenv("VOXCPM_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--cfg-value", type=float, default=_env_float("VOXCPM_CFG_VALUE", 1.5))
    parser.add_argument(
        "--inference-timesteps",
        type=int,
        default=_env_int("VOXCPM_INFERENCE_TIMESTEPS", 30),
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    if not args.voice_id:
        print("error: --voice-id or VOXCPM_DEFAULT_VOICE_ID is required", file=sys.stderr)
        return 2

    text_path = os.path.expanduser(args.text_file)
    output_path = os.path.expanduser(args.out)

    try:
        text = open(text_path, encoding="utf-8").read().strip()
    except OSError as err:
        print(f"error: cannot read text file: {err}", file=sys.stderr)
        return 1

    if not text:
        print("error: input text is empty", file=sys.stderr)
        return 1

    api_url = args.api_url.rstrip("/")
    payload = {
        "voice_id": args.voice_id,
        "text": text,
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
        "response_format": "audio",
    }

    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(f"{api_url}/v1/tts", json=payload)
            response.raise_for_status()
            audio = response.content
    except httpx.HTTPStatusError as err:
        detail = err.response.text.strip()
        print(f"error: API returned {err.response.status_code}: {detail}", file=sys.stderr)
        return 1
    except httpx.RequestError as err:
        print(f"error: API request failed: {err}", file=sys.stderr)
        return 1

    if not audio:
        print("error: API returned empty audio", file=sys.stderr)
        return 1

    try:
        with open(output_path, "wb") as handle:
            handle.write(audio)
    except OSError as err:
        print(f"error: cannot write output file: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
