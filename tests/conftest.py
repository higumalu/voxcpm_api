from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


@pytest.fixture
def reference_wav(tmp_path):
    path = tmp_path / "ref.wav"
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    wav = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(path, wav, sr, subtype="PCM_16")
    return str(path)

