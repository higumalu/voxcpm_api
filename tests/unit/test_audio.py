from __future__ import annotations

import base64

import numpy as np
import pytest

from voxcpm_api.audio import AudioDecodeError, decode_base64_wav, encode_wav_base64


@pytest.mark.unit
def test_decode_base64_wav_roundtrip(wav_bytes: bytes) -> None:
    encoded = base64.b64encode(wav_bytes).decode("ascii")
    raw, samples, sample_rate = decode_base64_wav(encoded)
    assert raw == wav_bytes
    assert sample_rate == 24000
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert len(samples) == 24000


@pytest.mark.unit
def test_decode_base64_wav_empty_raises() -> None:
    with pytest.raises(AudioDecodeError):
        decode_base64_wav("")


@pytest.mark.unit
def test_decode_base64_wav_invalid_base64_raises() -> None:
    with pytest.raises(AudioDecodeError):
        decode_base64_wav("not-base64!!!")


@pytest.mark.unit
def test_decode_base64_wav_invalid_payload_raises() -> None:
    payload = base64.b64encode(b"this is not a wav file").decode("ascii")
    with pytest.raises(AudioDecodeError):
        decode_base64_wav(payload)


@pytest.mark.unit
def test_encode_wav_base64_produces_valid_payload() -> None:
    audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    encoded = encode_wav_base64(audio, 16000)
    raw, samples, sample_rate = decode_base64_wav(encoded)
    assert sample_rate == 16000
    assert raw[:4] == b"RIFF"
    assert samples.shape == (16000,)
