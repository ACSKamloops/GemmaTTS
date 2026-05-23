import pytest
import time
from app.audio.signer import sign_audio_id, verify_signed_audio_id

def test_sign_and_verify_valid():
    audio_id = "aud_123"
    token = sign_audio_id(audio_id, expiry_seconds=10)
    assert token.startswith(audio_id)
    
    verified = verify_signed_audio_id(token)
    assert verified == audio_id

def test_expiry_enforcement():
    audio_id = "aud_123"
    # Sign with negative expiration (already expired)
    token = sign_audio_id(audio_id, expiry_seconds=-1)
    
    verified = verify_signed_audio_id(token)
    assert verified is None

def test_tampering_detection():
    audio_id = "aud_123"
    token = sign_audio_id(audio_id, expiry_seconds=60)
    
    parts = token.split(".")
    
    # Tamper with audio_id
    tampered_id = f"aud_999.{parts[1]}.{parts[2]}"
    assert verify_signed_audio_id(tampered_id) is None
    
    # Tamper with timestamp
    tampered_time = f"{parts[0]}.{int(parts[1])+100}.{parts[2]}"
    assert verify_signed_audio_id(tampered_time) is None
    
    # Tamper with signature
    tampered_sig = f"{parts[0]}.{parts[1]}.badsignature"
    assert verify_signed_audio_id(tampered_sig) is None
