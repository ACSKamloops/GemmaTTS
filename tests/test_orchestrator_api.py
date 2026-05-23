import asyncio
import base64
import io
import os
import pathlib
import shutil
import struct
import tempfile
import time
import urllib.parse
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.audio.cache import AudioCacheManager, get_cache_key
from app.audio.signer import sign_audio_id, verify_signed_audio_id
from app.config import settings
from app.services.orchestrator_api import app

@pytest.fixture(autouse=True)
def force_distributed_mode():
    orig_unified = settings.unified
    settings.unified = False
    yield
    settings.unified = orig_unified

@pytest.fixture
def client():
    # Clear the test cache directory before each test
    cache_dir = settings.audio_cache_dir
    if cache_dir.exists():
        for f in cache_dir.iterdir():
            if f.is_file() or f.is_symlink():
                try:
                    f.unlink()
                except OSError:
                    pass
    return TestClient(app)

# Helper to generate dummy wav bytes
def get_dummy_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00" * 2000)  # 2000 samples
    return buffer.getvalue()

# Mock async HTTP post calls
async def mock_httpx_post(url, *args, **kwargs):
    url_str = str(url)
    if "8001/generate" in url_str:
        json_payload = kwargs.get("json", {})
        prompt = json_payload.get("prompt", "")
        
        if "simulate-llm-bad-json" in prompt:
            return httpx.Response(200, text="not-a-json-string-at-all")
        if "simulate_llm_failed_status" in prompt:
            return httpx.Response(500, text="internal server error")
            
        return httpx.Response(200, json={
            "text": f"MOCK_RESPONSE: {prompt}",
            "generation_time_ms": 120.0
        })
        
    elif "8002/synthesize" in url_str:
        json_payload = kwargs.get("json", {})
        text = json_payload.get("text", "")
        engine = json_payload.get("engine", "")
        voice_id = json_payload.get("voice_id", "")
        
        if engine == "dia" and "simulate_offline" in voice_id:
            return httpx.Response(503, text="Dia engine offline")
            
        if engine == "fish" and "enable_fish" not in voice_id:
            return httpx.Response(403, text="Fish Audio consent required")
            
        wav_bytes = get_dummy_wav()
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        
        return httpx.Response(200, json={
            "audio_bytes_base64": audio_b64,
            "format": "wav",
            "sample_rate": 24000,
            "synthesis_time_ms": 340.0
        })
        
    return httpx.Response(404)

# =====================================================================
# Unit Tests
# =====================================================================

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_rotate_key_endpoint(client):
    new_key = "new-temp-secret-key-12345-longer-version-32"
    response = client.post(f"/debug/rotate_key?new_key={new_key}")
    assert response.status_code == 200
    assert response.json() == {"status": "key rotated"}
    assert settings.secret_key == new_key

def test_update_settings_endpoint(client):
    orig_max_cache = settings.max_cache_size_bytes
    orig_max_file = settings.max_file_size_bytes
    try:
        response = client.post("/debug/update_settings?max_cache_size_bytes=50000&max_file_size_bytes=1000")
        assert response.status_code == 200
        assert response.json() == {"status": "settings updated"}
        assert settings.max_cache_size_bytes == 50000
        assert settings.max_file_size_bytes == 1000
    finally:
        settings.max_cache_size_bytes = orig_max_cache
        settings.max_file_size_bytes = orig_max_file

@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
def test_dialogue_audio_disabled(mock_post, client):
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Hello world",
        "output": {"audio": False, "format": "wav"}
    }
    response = client.post("/v1/dialogue", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "MOCK_RESPONSE: Hello world"
    assert data["audio"] is None
    assert data["metrics"]["cache_hit"] is False
    assert data["metrics"]["tts_ms"] == 0.0

@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
def test_dialogue_audio_enabled_and_cached(mock_post, client):
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Cached query",
        "output": {"audio": True, "format": "wav"}
    }
    
    # First call: Cache miss
    response1 = client.post("/v1/dialogue", json=payload)
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["metrics"]["cache_hit"] is False
    assert data1["audio"] is not None
    signed_id = data1["audio"]["audio_id"]

    # Second call: Cache hit
    response2 = client.post("/v1/dialogue", json=payload)
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["metrics"]["cache_hit"] is True
    assert data2["audio"]["audio_id"] is not None
    
    # Request Cache-Control: no-cache should force miss
    response3 = client.post("/v1/dialogue", json=payload, headers={"Cache-Control": "no-cache"})
    assert response3.status_code == 200
    data3 = response3.json()
    assert data3["metrics"]["cache_hit"] is False

@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
def test_dialogue_corrupt_cache_recovery(mock_post, client):
    # Populate cache
    text = "MOCK_RESPONSE: Corrupt me"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Corrupt me",
        "output": {"audio": True, "format": "wav"}
    }
    response1 = client.post("/v1/dialogue", json=payload)
    assert response1.status_code == 200
    assert response1.json()["metrics"]["cache_hit"] is False

    # Get the cache file path and write all zeros to corrupt it
    cache_key = get_cache_key(text, "af_heart", "wav", engine="chatterbox", encoder_settings="voice_agent_fast")
    cache_manager = AudioCacheManager()
    path = cache_manager.get_file_path(cache_key, "wav")
    assert path.exists()
    path.write_bytes(b"\x00" * 100)

    # Call again: corrupt cache recovery (should trigger miss and rewrite)
    response2 = client.post("/v1/dialogue", json=payload)
    assert response2.status_code == 200
    assert response2.json()["metrics"]["cache_hit"] is False
    assert path.read_bytes().startswith(b"RIFF")

def test_dialogue_unsupported_format(client):
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Unsupported format test",
        "output": {"audio": True},
        "tts": {"format": "flac"}
    }
    response = client.post("/v1/dialogue", json=payload)
    assert response.status_code == 422

def test_dialogue_llm_crash_simulation(client):
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "simulate_llm_crash now",
        "output": {"audio": False, "format": "wav"}
    }
    response = client.post("/v1/dialogue", json=payload)
    assert response.status_code == 503
    assert "LLM service unavailable" in response.json()["detail"]

@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
def test_dialogue_llm_schema_mismatch_fallback(mock_post, client):
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "simulate-llm-bad-json",
        "output": {"audio": False, "format": "wav"},
        "fallback_policy": "use_static_text"
    }
    response = client.post("/v1/dialogue", json=payload)
    assert response.status_code == 200
    assert "Fallback dialogue text due to schema mismatch." in response.json()["text"]

@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
def test_dialogue_llm_schema_mismatch_raise_error(mock_post, client):
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "simulate-llm-bad-json",
        "output": {"audio": False, "format": "wav"},
        "fallback_policy": "raise_error"
    }
    response = client.post("/v1/dialogue", json=payload)
    assert response.status_code == 502

@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
def test_dialogue_style_engine_selection(mock_post, client):
    engines = ["dia", "kokoro", "piper", "f5_tts"]
    for eng in engines:
        payload = {
            "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
            "user_text": f"Engine test for {eng}",
            "output": {"audio": True, "format": "wav"},
            "tts": {"engine": eng, "voice_id": "default", "format": "wav", "profile": "voice_agent_fast"}
        }
        with patch.object(settings, "enable_f5_tts", True):
            response = client.post("/v1/dialogue", json=payload)
            assert response.status_code == 200
            
            called_args = mock_post.call_args_list[-1]
            payload_sent = called_args[1]["json"]
            assert payload_sent["engine"] == eng

@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
def test_dialogue_dia_failure_fallback_to_piper(mock_post, client):
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart_dia_simulate_offline", "style": "dia"},
        "user_text": "Dia fallback to Piper test",
        "output": {"audio": True, "format": "wav"},
        "tts": {"engine": "dia"}
    }
    response = client.post("/v1/dialogue", json=payload)
    assert response.status_code == 200
    assert response.json()["audio"] is not None

@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
def test_dialogue_client_disconnect(mock_post, client):
    # Mock Request.is_disconnected to return True immediately
    async def mock_is_disconnected():
        return True

    with patch.object(Request, "is_disconnected", side_effect=mock_is_disconnected):
        payload = {
            "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
            "user_text": "simulate_client_disconnect",
            "output": {"audio": True, "format": "wav"}
        }
        response = client.post("/v1/dialogue", json=payload)
        assert response.status_code == 499

def test_audio_delivery_success(client):
    # Put a dummy file in cache
    cache_manager = AudioCacheManager()
    dummy_data = b"RIFFdummywavcontent"
    path = cache_manager.get_file_path("test_audio", "wav")
    path.write_bytes(dummy_data)
    
    # Sign it
    token = sign_audio_id("test_audio_wav")
    
    response = client.get(f"/audio/{token}")
    assert response.status_code == 200
    assert response.content == dummy_data
    assert response.headers["content-type"] == "audio/wav"

def test_audio_delivery_expired(client):
    token = sign_audio_id("test_audio_wav", expiry_seconds=-10)
    response = client.get(f"/audio/{token}")
    assert response.status_code == 403
    assert "Signature expired or invalid" in response.json()["detail"]

def test_audio_delivery_tampered(client):
    token = sign_audio_id("test_audio_wav")
    parts = token.split(".")
    tampered = f"{parts[0]}.{parts[1]}.badsignature"
    response = client.get(f"/audio/{tampered}")
    assert response.status_code == 403

def test_audio_delivery_path_traversal(client):
    traversals = [
        "../etc/passwd",
        "..\\Windows\\win.ini",
        "etc/passwd",
        "/etc/passwd"
    ]
    for path in traversals:
        # Check direct path traversal rejection
        encoded = urllib.parse.quote(path)
        response = client.get(f"/audio/{encoded}")
        assert response.status_code in (400, 403, 404)

def test_audio_symlink_safety(client, tmp_path):
    outside_file = tmp_path / "outside.wav"
    outside_file.write_bytes(b"RIFFoutsidecontent")
    
    cache_dir = settings.audio_cache_dir
    symlink_path = cache_dir / "badlink.wav"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
        
    try:
        os.symlink(str(outside_file.resolve()), str(symlink_path))
        token = sign_audio_id("badlink_wav")
        response = client.get(f"/audio/{token}")
        assert response.status_code == 403
        assert response.json()["detail"] in ("Symlink targets outside cache directory.", "Path traversal or out-of-boundary access detected.")
    finally:
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
