import pytest
import hmac
import hashlib
import time
import io
import wave
import base64
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient

from app.config import settings
from app.audio.cache import AudioCacheManager
from app.audio.signer import verify_signed_audio_id, sign_audio_id
from app.safety.text_sanitizer import sanitize_text

# =====================================================================
# Mock Helpers
# =====================================================================

def get_dummy_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00" * 2000)  # 2000 samples
    return buffer.getvalue()

async def mock_httpx_post(url, *args, **kwargs):
    url_str = str(url)
    if "8001/generate" in url_str:
        json_payload = kwargs.get("json", {})
        prompt = json_payload.get("prompt", "")
        max_words = json_payload.get("max_words", 150)
        
        if not prompt or prompt.strip() == "":
            return httpx.Response(422, json={"detail": "Prompt cannot be empty"})
        if max_words is not None and max_words < 0:
            return httpx.Response(422, json={"detail": "max_words cannot be negative"})
            
        return httpx.Response(200, json={
            "text": f"MOCK_RESPONSE: {prompt}",
            "generation_time_ms": 120.0
        })
        
    elif "8002/synthesize" in url_str:
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
# Adversarial Tests
# =====================================================================

def test_link_sanitizer_bypass():
    """
    Test 1: Link Sanitizer Bypass (Fixed)
    Checks that extremely short hostnames (<=1 char) do NOT bypass the sanitizer,
    and are instead successfully stripped.
    """
    input_text = "Check this out: [Google](http://a)"
    sanitized = sanitize_text(input_text)
    assert "http://a" not in sanitized
    assert sanitized == "Check this out: Google"


def test_token_split_dos():
    """
    Test 2: Token Split Denial of Service
    Verifies that verify_signed_audio_id handles a very large number of dots
    without throwing exceptions or crashing.
    """
    large_payload = "." * 100000
    result = verify_signed_audio_id(large_payload)
    assert result is None


def test_cache_size_bypass():
    """
    Test 3: Cache Size Bypass via Negative/Zero Configuration
    Checks that when max_cache_size_bytes is configured to a negative value,
    calling put raises ValueError.
    """
    manager = AudioCacheManager()
    orig_max_cache = settings.max_cache_size_bytes
    orig_max_file = settings.max_file_size_bytes
    
    try:
        # Set limits to negative / zero
        settings.max_cache_size_bytes = -100
        settings.max_file_size_bytes = 1000
        
        data = b"x" * 200
        with pytest.raises(ValueError):
            manager.put("test_neg_cache", "voice", "ogg", data)
        
    finally:
        # Restore settings
        settings.max_cache_size_bytes = orig_max_cache
        settings.max_file_size_bytes = orig_max_file


def test_empty_key_rotation_and_forgery():
    """
    Test 4: Weak Cryptographic Keys via Key Rotation
    Verifies that rotating the key to an empty string or weak key is rejected with a 400 status code.
    """
    from app.services.orchestrator_api import app
    client = TestClient(app)
    orig_key = settings.secret_key
    
    try:
        # Rotate key to empty string
        resp = client.post("/debug/rotate_key?new_key=")
        assert resp.status_code == 400
        
        # Rotate key to a short string (e.g. 10 chars)
        resp_short = client.post("/debug/rotate_key?new_key=shortkey12")
        assert resp_short.status_code == 400
        
    finally:
        settings.secret_key = orig_key


@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
def test_incorrect_compressed_duration_metadata(mock_post):
    """
    Test 5: Incorrect Duration Metadata in Compressed Formats (Fixed)
    Verifies that OGG and MP3 formats yield accurate duration_ms metadata matching WAV.
    """
    from app.services.orchestrator_api import app
    client = TestClient(app)
    
    payload_wav = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Duration check",
        "output": {"audio": True, "format": "wav"}
    }
    payload_ogg = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Duration check",
        "output": {"audio": True, "format": "ogg"}
    }
    
    # 1. WAV Response duration
    resp_wav = client.post("/v1/dialogue", json=payload_wav)
    assert resp_wav.status_code == 200
    wav_duration = resp_wav.json()["audio"]["duration_ms"]
    
    # 2. OGG Response duration
    resp_ogg = client.post("/v1/dialogue", json=payload_ogg)
    assert resp_ogg.status_code == 200
    ogg_duration = resp_ogg.json()["audio"]["duration_ms"]
    
    # Both have the same 1000 sample dummy wav (41.66ms duration).
    # We assert that the calculated duration metadata is accurate and identical.
    assert wav_duration == ogg_duration
    assert wav_duration == 41


@patch("httpx.AsyncClient.post", side_effect=mock_httpx_post)
def test_swallowed_validation_errors(mock_post):
    """
    Test 6: Swallowed Client Input Validation Errors
    Verifies that the orchestrator API propagates client input errors (empty query or negative max_words)
    by returning 422 errors from the Gemma service.
    """
    from app.services.orchestrator_api import app
    client = TestClient(app)
    
    # 1. Empty user_text
    payload_empty = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "   ",
        "output": {"audio": False, "format": "wav"}
    }
    response_empty = client.post("/v1/dialogue", json=payload_empty)
    # The empty/whitespace text causes Gemma to return 422, and the orchestrator propagates it
    assert response_empty.status_code == 422
    assert "detail" in response_empty.json()

    # 2. Negative max_words
    payload_neg = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Hello",
        "max_words": -10,
        "output": {"audio": False, "format": "wav"}
    }
    response_neg = client.post("/v1/dialogue", json=payload_neg)
    # Negative max_words causes Gemma to return 422, and the orchestrator propagates it
    assert response_neg.status_code == 422
    assert "detail" in response_neg.json()


def test_piper_dependency_discrepancy():
    """
    Test 7: Missing Dependency Error Code Discrepancy
    Checks that requesting piper synthesis with test_mode=False returns a 503 error
    due to deferred import inside synthesise() raising ImportError.
    """
    from app.services.tts_service import app
    client = TestClient(app)
    
    # piper is not installed, so loading it during synthesize raises ImportError.
    # This now bubbles up as a 503 service unavailable.
    response = client.post("/synthesize", json={
        "text": "Hello",
        "engine": "piper",
        "test_mode": False
    })
    
    assert response.status_code == 503
    assert "not installed" in response.json()["detail"].lower()
