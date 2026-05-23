import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.config import settings
from app.audio.cache import AudioCacheManager, get_cache_key
from app.main import app

def test_generate_simulation_triggers_gating():
    client = TestClient(app)
    
    # 1. Test Mode: simulate_llm_bad_json should trigger fake bad JSON string
    with patch("app.config.settings.mode", "test"):
        resp = client.post("/generate", json={
            "prompt": "simulate_llm_bad_json",
            "max_words": 50
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/plain; charset=utf-8"
        assert resp.text == "not-a-json-string-at-all"

    # 2. Dev Mode: simulate_llm_bad_json should NOT trigger fake bad JSON response
    # Mock LLM provider to avoid requiring real model weights
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Mocked normal text reply"
    
    with patch("app.config.settings.mode", "dev"), \
         patch("app.core.orchestrator.get_llm_provider", return_value=mock_llm):
        resp = client.post("/generate", json={
            "prompt": "simulate_llm_bad_json",
            "max_words": 50
        })
        assert resp.status_code == 200
        # Should return normal GenerateResponse JSON
        data = resp.json()
        assert "text" in data
        assert data["text"] == "Mocked normal text reply"


def test_cache_key_consistency():
    # Verify that get_cache_key produces the exact same hash regardless of sample_rate parameter,
    # because sample_rate is omitted from the cache key payload to ensure consistency.
    key_no_sr = get_cache_key(
        text="Hello test text",
        voice_id="default",
        format="wav",
        engine="kokoro",
        encoder_settings="high_quality_narration"
    )
    
    key_with_sr = get_cache_key(
        text="Hello test text",
        voice_id="default",
        format="wav",
        engine="kokoro",
        sample_rate=16000,
        encoder_settings="high_quality_narration"
    )
    
    assert key_no_sr == key_with_sr

    # Verify that cache put and get resolve the same file path and data when using sample_rate overrides
    import tempfile
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        manager = AudioCacheManager(cache_dir=cache_dir)
        
        # Put audio with a non-default sample rate (e.g. 16000)
        audio_data = b"RIFFfakeaudiowavdata"
        manager.put(
            text="Hello test text",
            voice_id="default",
            format="wav",
            data=audio_data,
            duration_ms=1000,
            engine="kokoro",
            sample_rate=16000,
            encoder_settings="high_quality_narration"
        )
        
        # Retrieve audio without explicitly passing sample_rate (e.g. default)
        retrieved_data = manager.get(
            text="Hello test text",
            voice_id="default",
            format="wav",
            engine="kokoro",
            encoder_settings="high_quality_narration"
        )
        
        assert retrieved_data == audio_data
        
        # Verify sample_rate is successfully recovered from the metadata sidecar
        metadata = manager.get_metadata(
            text="Hello test text",
            voice_id="default",
            format="wav",
            engine="kokoro",
            encoder_settings="high_quality_narration"
        )
        assert metadata is not None
        assert metadata.get("sample_rate") == 16000
        assert metadata.get("engine") == "kokoro"
        assert metadata.get("profile") == "high_quality_narration"
