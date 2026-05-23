"""Tests for the voice registry module and its API endpoints."""
import os
import pytest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.services.tts_service import app
from app.services.voice_registry import (
    VoiceInfo,
    VoiceRegistry,
    Engine,
    registry,
    _parse_kokoro_voice_id,
    _scan_kokoro_voices,
)

client = TestClient(app)


# ---- Unit tests for helpers ----

class TestParseKokoroVoiceId:
    def test_american_female(self):
        lang, gender = _parse_kokoro_voice_id("af_heart")
        assert lang == "en-US"
        assert gender == "female"

    def test_american_male(self):
        lang, gender = _parse_kokoro_voice_id("am_adam")
        assert lang == "en-US"
        assert gender == "male"

    def test_british_female(self):
        lang, gender = _parse_kokoro_voice_id("bf_emma")
        assert lang == "en-GB"
        assert gender == "female"

    def test_british_male(self):
        lang, gender = _parse_kokoro_voice_id("bm_george")
        assert lang == "en-GB"
        assert gender == "male"

    def test_japanese_female(self):
        lang, gender = _parse_kokoro_voice_id("jf_alpha")
        assert lang == "ja"
        assert gender == "female"

    def test_unknown_prefix(self):
        lang, gender = _parse_kokoro_voice_id("xx_unknown")
        assert lang == "en"  # fallback
        assert gender is None

    def test_short_id(self):
        lang, gender = _parse_kokoro_voice_id("a")
        assert lang == "en"
        assert gender is None


# ---- Unit tests for Kokoro scanning ----

class TestScanKokoroVoices:
    def test_scan_real_directory(self):
        """Scan the actual models/kokoro/voices dir if it exists."""
        voices_dir = "models/kokoro/voices"
        if not Path(voices_dir).is_dir():
            pytest.skip("Kokoro voices directory not present")
        voices = _scan_kokoro_voices(voices_dir)
        assert len(voices) > 0
        # Every result should be a VoiceInfo with engine=kokoro
        for v in voices:
            assert v.engine == Engine.kokoro
            assert not v.id.startswith("_")  # stubs filtered

    def test_scan_missing_directory(self):
        voices = _scan_kokoro_voices("/nonexistent/path")
        assert voices == []

    def test_scan_deduplicates(self, tmp_path):
        """When both .pt and .bin exist, only one entry is returned."""
        (tmp_path / "af_heart.pt").write_bytes(b"dummy")
        (tmp_path / "af_heart.bin").write_bytes(b"dummy")
        voices = _scan_kokoro_voices(str(tmp_path))
        ids = [v.id for v in voices]
        assert ids.count("af_heart") == 1

    def test_scan_skips_stubs(self, tmp_path):
        """Files starting with _ are skipped."""
        (tmp_path / "_voices_stub.npy").write_bytes(b"stub")
        (tmp_path / "_internal.pt").write_bytes(b"x")
        (tmp_path / "af_heart.pt").write_bytes(b"x")
        voices = _scan_kokoro_voices(str(tmp_path))
        ids = [v.id for v in voices]
        assert "_voices_stub" not in ids
        assert "_internal" not in ids
        assert "af_heart" in ids

    def test_scan_ignores_non_voice_files(self, tmp_path):
        """Non .pt/.bin files are ignored."""
        (tmp_path / "readme.txt").write_bytes(b"text")
        (tmp_path / "model.onnx").write_bytes(b"model")
        (tmp_path / "af_heart.pt").write_bytes(b"x")
        voices = _scan_kokoro_voices(str(tmp_path))
        assert len(voices) == 1
        assert voices[0].id == "af_heart"


# ---- Unit tests for VoiceRegistry class ----

class TestVoiceRegistry:
    def test_list_all_includes_static_voices(self, tmp_path):
        """Even with empty Kokoro dir, static voices are returned."""
        reg = VoiceRegistry(kokoro_voices_dir=str(tmp_path))
        voices = reg.list_all()
        engines = {v.engine.value for v in voices}
        assert "piper" in engines
        assert "chatterbox" in engines
        assert "dia" in engines
        assert "f5_tts" in engines

    def test_list_by_engine(self, tmp_path):
        reg = VoiceRegistry(kokoro_voices_dir=str(tmp_path))
        dia_voices = reg.list_by_engine("dia")
        assert len(dia_voices) == 2
        assert all(v.engine == Engine.dia for v in dia_voices)

    def test_list_by_engine_unknown_returns_empty(self, tmp_path):
        reg = VoiceRegistry(kokoro_voices_dir=str(tmp_path))
        assert reg.list_by_engine("nonexistent") == []

    def test_cache_invalidation(self, tmp_path):
        reg = VoiceRegistry(kokoro_voices_dir=str(tmp_path))
        first = reg.list_all()
        # Add a voice file
        (tmp_path / "am_test.pt").write_bytes(b"x")
        # Still cached
        assert len(reg.list_all()) == len(first)
        # After invalidation, rescans
        reg.invalidate()
        assert len(reg.list_all()) == len(first) + 1

    def test_piper_voice_details(self, tmp_path):
        reg = VoiceRegistry(kokoro_voices_dir=str(tmp_path))
        piper_voices = reg.list_by_engine("piper")
        assert len(piper_voices) == 1
        v = piper_voices[0]
        assert v.id == "en_US-lessac-medium"
        assert v.language == "en-US"
        assert v.sample_rate == 22050

    def test_chatterbox_voice_details(self, tmp_path):
        reg = VoiceRegistry(kokoro_voices_dir=str(tmp_path))
        cb_voices = reg.list_by_engine("chatterbox")
        assert len(cb_voices) == 1
        assert cb_voices[0].id == "default"
        assert "voice-cloning" in cb_voices[0].tags

    def test_dia_voice_details(self, tmp_path):
        reg = VoiceRegistry(kokoro_voices_dir=str(tmp_path))
        dia_voices = reg.list_by_engine("dia")
        ids = {v.id for v in dia_voices}
        assert ids == {"S1", "S2"}
        for v in dia_voices:
            assert v.sample_rate == 44100

    def test_f5_tts_voice_details(self, tmp_path):
        reg = VoiceRegistry(kokoro_voices_dir=str(tmp_path))
        f5_voices = reg.list_by_engine("f5_tts")
        assert len(f5_voices) == 1
        assert f5_voices[0].id == "default"
        assert "voice-cloning" in f5_voices[0].tags


# ---- API endpoint tests ----

class TestVoicesEndpoints:
    def test_get_voices_returns_list(self):
        response = client.get("/voices")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_get_voices_shape(self):
        response = client.get("/voices")
        data = response.json()
        required_fields = {"id", "name", "engine", "language", "sample_rate"}
        for voice in data:
            assert required_fields.issubset(voice.keys()), (
                f"Voice {voice.get('id')} missing fields: {required_fields - voice.keys()}"
            )

    def test_get_voices_by_engine_dia(self):
        response = client.get("/voices/dia")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(v["engine"] == "dia" for v in data)

    def test_get_voices_by_engine_piper(self):
        response = client.get("/voices/piper")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "en_US-lessac-medium"

    def test_get_voices_by_engine_chatterbox(self):
        response = client.get("/voices/chatterbox")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "default"

    def test_get_voices_by_engine_f5_tts(self):
        response = client.get("/voices/f5_tts")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["engine"] == "f5_tts"

    def test_get_voices_by_engine_kokoro(self):
        response = client.get("/voices/kokoro")
        assert response.status_code == 200
        data = response.json()
        # All returned voices should be kokoro
        assert all(v["engine"] == "kokoro" for v in data)

    def test_get_voices_unknown_engine_404(self):
        response = client.get("/voices/nonexistent")
        assert response.status_code == 404
        assert "Unknown engine" in response.json()["detail"]

    def test_get_voices_fish_engine_empty(self):
        """Fish engine has no static voices registered."""
        response = client.get("/voices/fish")
        assert response.status_code == 200
        data = response.json()
        assert data == []
