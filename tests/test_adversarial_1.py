import pytest
import time
import os
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import components
from app.audio.signer import sign_audio_id, verify_signed_audio_id
from app.audio.cache import AudioCacheManager, get_cache_key
from app.safety.text_sanitizer import sanitize_text
from app.safety.output_validator import validate_llm_json
from app.config import settings
from app.services.gemma_service import app as gemma_app, rate_limiter as gemma_rate_limiter
from app.services.orchestrator_api import app as orchestrator_app

@pytest.fixture
def gemma_client():
    gemma_rate_limiter.timestamps.clear()
    return TestClient(gemma_app)

@pytest.fixture
def orchestrator_client():
    return TestClient(orchestrator_app)


# =====================================================================
# 1. Signer Adversarial Tests
# =====================================================================

def test_signer_with_dots():
    """
    Exposes bug: Any audio_id containing dots cannot be verified.
    Since verify_signed_audio_id splits by '.', an audio_id like "my.audio.file"
    will result in len(parts) > 3, returning None.
    """
    audio_id = "my.audio.file"
    token = sign_audio_id(audio_id, expiry_seconds=60)
    
    # Verification succeeds because the id is parsed correctly using rsplit(".", 2)
    verified = verify_signed_audio_id(token)
    assert verified == "my.audio.file"


def test_signer_boundary_inputs():
    """
    Tests verify_signed_audio_id with empty, invalid, and tampered formats.
    """
    # Empty token
    assert verify_signed_audio_id("") is None
    assert verify_signed_audio_id(None) is None
    
    # Token with too few or too many parts
    assert verify_signed_audio_id("one.two") is None
    assert verify_signed_audio_id("one.two.three.four") is None
    
    # Invalid timestamp (non-integer)
    assert verify_signed_audio_id("aud_123.notanint.signature") is None
    
    # Huge timestamp overflow boundary check
    huge_timestamp = "999999999999999999999999999999"
    token = f"aud_123.{huge_timestamp}.signature"
    # Should safely return None (signature mismatch) rather than raising OverflowError
    assert verify_signed_audio_id(token) is None


# =====================================================================
# 2. Audio Cache Manager Adversarial Tests
# =====================================================================

def test_cache_empty_key_and_format():
    """
    Exposes boundary: If key and format resolve to empty after stripping
    non-alphanumeric characters, get_file_path raises ValueError.
    """
    manager = AudioCacheManager()
    
    with pytest.raises(ValueError, match="Empty key or format after sanitization"):
        manager.get_file_path("", "")


def test_cache_negative_settings():
    """
    Tests behavior when cache settings are set to negative values.
    Specifically, check that if max_cache_size_bytes is negative,
    calling put raises ValueError.
    """
    manager = AudioCacheManager()
    
    # Store original settings
    orig_max_cache = settings.max_cache_size_bytes
    orig_max_file = settings.max_file_size_bytes
    
    try:
        # Set max cache size to negative
        settings.max_cache_size_bytes = -1000
        settings.max_file_size_bytes = 100 * 1024 # 100 KB
        
        # Clear cache first
        for f in manager.cache_dir.iterdir():
            if f.is_file():
                f.unlink()
                
        # Write some data
        data = b"x" * 1000
        with pytest.raises(ValueError):
            manager.put("text1", "voice", "ogg", data)
        
    finally:
        # Restore settings
        settings.max_cache_size_bytes = orig_max_cache
        settings.max_file_size_bytes = orig_max_file


def test_cache_prune_incoming_greater_than_max():
    """
    Exposes logic gap: If incoming_bytes > max_cache_size_bytes, prune_cache
    is bypassed, but put raises ValueError to avoid writing files exceeding cache limit.
    """
    manager = AudioCacheManager()
    
    orig_max_cache = settings.max_cache_size_bytes
    orig_max_file = settings.max_file_size_bytes
    
    try:
        # Configure small max_cache_size but larger max_file_size
        settings.max_cache_size_bytes = 500
        settings.max_file_size_bytes = 1000
        
        # Clear cache
        for f in manager.cache_dir.iterdir():
            if f.is_file():
                f.unlink()
                
        # Write initial file of 400 bytes (fits under 500 limit)
        path1 = manager.put("text1", "voice", "ogg", b"y" * 400)
        assert path1.exists()
        
        # Write second file of 600 bytes (exceeds 500 but is under 1000)
        with pytest.raises(ValueError):
            manager.put("text2", "voice", "ogg", b"z" * 600)
        
    finally:
        settings.max_cache_size_bytes = orig_max_cache
        settings.max_file_size_bytes = orig_max_file


# =====================================================================
# 3. Text Sanitizer Adversarial Tests
# =====================================================================

def test_sanitizer_single_char_host_url():
    """
    Exposes boundary: URLs with single-character hostnames (e.g., http://a)
    or markdown links pointing to them (Fixed).
    They are now correctly and completely stripped.
    """
    # "http://ab" gets stripped
    assert sanitize_text("Hello http://ab") == "Hello"
    
    # "http://a" is now stripped!
    assert sanitize_text("Hello http://a") == "Hello"
    
    # Markdown link with single char host is now completely stripped without leaving dangling "("
    assert sanitize_text("Go to [link](http://a)") == "Go to link"


def test_sanitizer_trailing_traversal():
    """
    Exposes boundary: TRAVERSAL_REGEX only matches traversal sequences followed by
    a slash or backslash. If the traversal sequence is at the end of the text without
    a trailing slash (e.g. ".."), it is NOT removed.
    """
    assert sanitize_text("path/../file") == "path/file"
    
    # Trailing traversal without slash
    assert sanitize_text("path/..") == "path/.."


# =====================================================================
# 4. Output Validator Adversarial Tests
# =====================================================================

def test_output_validator_empty_fields():
    """
    Tests that validate_llm_json accepts JSON where 'text' is an empty string
    or whitespace-only. This can downstream cause synthesis failures in TTS.
    """
    # Empty string text field
    raw_empty = '{"text": ""}'
    parsed_empty = validate_llm_json(raw_empty)
    assert parsed_empty is not None
    assert parsed_empty["text"] == ""
    
    # Whitespace only text field
    raw_whitespace = '{"text": "   "}'
    parsed_ws = validate_llm_json(raw_whitespace)
    assert parsed_ws is not None
    assert parsed_ws["text"] == "   "


# =====================================================================
# 5. Gemma Service Endpoint Adversarial Tests
# =====================================================================

def test_gemma_truncation_limits(gemma_client):
    """
    Tests the truncation boundary of Gemma service prompt input.
    Prompt <= 5000 chars: not truncated.
    Prompt > 5000 chars: truncated to 1000 chars (MAX_FALLBACK_CHARS).
    """
    # Prompt of exactly 5000 characters
    prompt_5000 = "a" * 5000
    response = gemma_client.post("/generate", json={"prompt": prompt_5000, "test_mode": True})
    assert response.status_code == 200
    assert response.json()["text"] == f"MOCK_RESPONSE: {prompt_5000}"
    
    # Prompt of 5001 characters
    prompt_5001 = "a" * 5001
    response = gemma_client.post("/generate", json={"prompt": prompt_5001, "test_mode": True})
    assert response.status_code == 200
    # Gets truncated to 1000 chars
    expected_prompt = "a" * 1000
    assert response.json()["text"] == f"MOCK_RESPONSE: {expected_prompt}"


def test_gemma_huge_max_words(gemma_client):
    """
    Tests that passing a huge max_words passes Pydantic validation (which it should),
    but we verify that the service parses the integer without crashing.
    """
    huge_max_words = 999999999999999999999999999999
    response = gemma_client.post("/generate", json={
        "prompt": "Hello",
        "max_words": huge_max_words,
        "test_mode": True
    })
    assert response.status_code == 200
    assert response.json()["text"] == "MOCK_RESPONSE: Hello"
