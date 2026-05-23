import os
import base64
import pytest
from fastapi.testclient import TestClient
from app.services.tts_service import app, rate_limiter

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "tts-service"}

@pytest.mark.parametrize("engine", ["chatterbox", "dia", "kokoro", "piper"])
def test_synthesize_success_test_mode(engine):
    payload = {
        "text": "Hello world duration_sec=1.5 size_bytes=1000",
        "voice_id": "default",
        "engine": engine,
        "test_mode": True
    }
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["format"] == "wav"
    assert data["sample_rate"] == 24000
    assert "audio_bytes_base64" in data
    
    # Verify we can decode it and it matches the size
    wav_bytes = base64.b64decode(data["audio_bytes_base64"])
    assert len(wav_bytes) == 1000

def test_validation_empty_text():
    payload = {
        "text": "",
        "engine": "chatterbox",
        "test_mode": True
    }
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 422

def test_fish_consent_blocked_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_FISH_AUDIO", raising=False)
    payload = {
        "text": "Hello",
        "voice_id": "default",
        "engine": "fish",
        "test_mode": True
    }
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 403
    assert "consent" in response.json()["detail"]

def test_fish_consent_allowed_via_voice_id(monkeypatch):
    monkeypatch.delenv("ENABLE_FISH_AUDIO", raising=False)
    payload = {
        "text": "Hello",
        "voice_id": "enable_fish_style",
        "engine": "fish",
        "test_mode": True
    }
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "audio_bytes_base64" in data

def test_fish_consent_allowed_via_env(monkeypatch):
    monkeypatch.setenv("ENABLE_FISH_AUDIO", "true")
    payload = {
        "text": "Hello",
        "voice_id": "default",
        "engine": "fish",
        "test_mode": True
    }
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "audio_bytes_base64" in data

def test_fish_consent_case_insensitive_env(monkeypatch):
    monkeypatch.setenv("ENABLE_FISH_AUDIO", "TrUe")
    payload = {
        "text": "Hello",
        "voice_id": "default",
        "engine": "fish",
        "test_mode": True
    }
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 200

def test_dia_offline_simulation():
    payload = {
        "text": "Hello",
        "voice_id": "simulate_offline_preset",
        "engine": "dia",
        "test_mode": True
    }
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 503
    assert "offline" in response.json()["detail"].lower()

def test_rate_limiting():
    # Make sure timestamps are cleared before testing
    rate_limiter.timestamps.clear()
    
    payload = {
        "text": "Hello limit test",
        "engine": "chatterbox",
        "test_mode": True
    }
    
    # First 40 should pass
    for _ in range(40):
        response = client.post("/synthesize", json=payload)
        assert response.status_code == 200
        
    # The 41st should fail with 429
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 429
    assert "too many requests" in response.json()["detail"].lower()
    
    # Clean up rate limiter state
    rate_limiter.timestamps.clear()
