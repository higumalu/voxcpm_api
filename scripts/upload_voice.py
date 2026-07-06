#!/usr/bin/env python3
"""Upload a reference WAV file and transcript to create a voice_id.

Examples:
    uv run python scripts/upload_voice.py ref.wav --reference-text "你好，這是參考語音。"
    uv run python scripts/upload_voice.py ref.wav --text-file prompt.txt
    uv run python scripts/upload_voice.py ref.wav -t "demo" --export-env
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import httpx


def _read_reference_text(args: argparse.Namespace) -> str:
    if args.reference_text and args.text_file:
        raise SystemExit("error: use only one of --reference-text or --text-file")

    if args.text_file:
        path = Path(args.text_file).expanduser()
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as err:
            raise SystemExit(f"error: cannot read text file: {err}") from err
    else:
        text = (args.reference_text or "").strip()

    if not text:
        raise SystemExit("error: reference text is empty (use --reference-text or --text-file)")
    return text


def _read_wav_base64(wav_path: Path) -> str:
    if not wav_path.exists():
        raise SystemExit(f"error: wav file not found: {wav_path}")
    if wav_path.suffix.lower() != ".wav":
        print(f"warning: file extension is not .wav: {wav_path}", file=sys.stderr)

    try:
        raw = wav_path.read_bytes()
    except OSError as err:
        raise SystemExit(f"error: cannot read wav file: {err}") from err

    if not raw:
        raise SystemExit(f"error: wav file is empty: {wav_path}")
    return base64.b64encode(raw).decode("ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upload BASE64 WAV + reference text to POST /v1/voices",
    )
    parser.add_argument("wav", help="Path to reference WAV file")
    parser.add_argument("-t", "--reference-text", help="Reference transcript for the WAV")
    parser.add_argument("--text-file", help="UTF-8 file containing the reference transcript")
    parser.add_argument("--api-url", default=os.getenv("VOXCPM_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--export-env",
        action="store_true",
        help="Print export VOXCPM_DEFAULT_VOICE_ID=... after success",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print voice_id on success",
    )
    args = parser.parse_args(argv)

    wav_path = Path(args.wav).expanduser().resolve()
    reference_text = _read_reference_text(args)
    audio_base64 = _read_wav_base64(wav_path)

    payload = {
        "audio_base64": audio_base64,
        "reference_text": reference_text,
    }
    api_url = args.api_url.rstrip("/")

    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(f"{api_url}/v1/voices", json=payload)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as err:
        detail = err.response.text.strip()
        print(f"error: API returned {err.response.status_code}: {detail}", file=sys.stderr)
        return 1
    except httpx.RequestError as err:
        print(f"error: API request failed: {err}", file=sys.stderr)
        return 1

    voice_id = result.get("voice_id", "")
    if not voice_id:
        print("error: API response missing voice_id", file=sys.stderr)
        return 1

    if args.quiet:
        print(voice_id)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.export_env:
        print(f'export VOXCPM_DEFAULT_VOICE_ID="{voice_id}"', file=sys.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
