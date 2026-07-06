from __future__ import annotations

from pathlib import Path

import pytest

from voxcpm_api.audio import AudioDecodeError
from voxcpm_api.voice_store import VoiceNotFoundError, VoiceStore


@pytest.mark.unit
def test_create_voice_persists_files(tmp_path: Path, wav_base64: str) -> None:
    store = VoiceStore(tmp_path / "voices")
    meta = store.create(wav_base64, "你好，這是一段參考語音。")

    assert len(meta.voice_id) == 32
    assert meta.sample_rate == 24000
    assert meta.duration_sec == pytest.approx(1.0, abs=0.05)

    voice_dir = store.voices_dir / meta.voice_id
    assert (voice_dir / "reference.wav").exists()
    assert (voice_dir / "metadata.json").exists()


@pytest.mark.unit
def test_get_voice_returns_record(tmp_path: Path, wav_base64: str) -> None:
    store = VoiceStore(tmp_path / "voices")
    meta = store.create(wav_base64, "demo")

    record = store.get(meta.voice_id)
    assert record.metadata.voice_id == meta.voice_id
    assert record.metadata.reference_text == "demo"
    assert record.audio_path.exists()


@pytest.mark.unit
def test_get_voice_missing_raises(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path / "voices")
    with pytest.raises(VoiceNotFoundError):
        store.get("0" * 32)


@pytest.mark.unit
def test_get_voice_invalid_id_raises(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path / "voices")
    with pytest.raises(VoiceNotFoundError):
        store.get("not-a-uuid")
    with pytest.raises(VoiceNotFoundError):
        store.get("../etc/passwd")


@pytest.mark.unit
def test_list_voices_sorted_by_created_at(tmp_path: Path, wav_base64: str) -> None:
    store = VoiceStore(tmp_path / "voices")
    a = store.create(wav_base64, "a")
    b = store.create(wav_base64, "b")

    voices = store.list()
    assert [v.voice_id for v in voices] == [a.voice_id, b.voice_id]
    assert store.count() == 2


@pytest.mark.unit
def test_delete_voice_removes_directory(tmp_path: Path, wav_base64: str) -> None:
    store = VoiceStore(tmp_path / "voices")
    meta = store.create(wav_base64, "delete me")
    store.delete(meta.voice_id)

    assert not (store.voices_dir / meta.voice_id).exists()
    assert store.count() == 0
    with pytest.raises(VoiceNotFoundError):
        store.delete(meta.voice_id)


@pytest.mark.unit
def test_create_rejects_invalid_audio(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path / "voices")
    with pytest.raises(AudioDecodeError):
        store.create("not-base64!", "x")
