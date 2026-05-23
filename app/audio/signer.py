import hmac
import hashlib
import time
from typing import Optional
from app.config import settings

def sign_audio_id(audio_id: str, expiry_seconds: Optional[int] = None) -> str:
    """
    Generates a signed URL token containing the audio_id, timestamp, and signature.
    Format: audio_id.expiry_timestamp.signature
    """
    if expiry_seconds is None:
        expiry_seconds = settings.signed_url_expiry_seconds
        
    expiry_time = int(time.time()) + expiry_seconds
    message = f"{audio_id}.{expiry_time}"
    
    signature = hmac.new(
        settings.secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    return f"{audio_id}.{expiry_time}.{signature}"

def verify_signed_audio_id(signed_id: str) -> Optional[str]:
    """
    Verifies signature and expiration of a signed token.
    Returns the audio_id if valid, or None if invalid or expired.
    """
    if not signed_id:
        return None
        
    parts = signed_id.split(".")
    if len(parts) != 3:
        return None
        
    audio_id, expiry_str, signature = parts
    
    try:
        expiry_time = int(expiry_str)
    except ValueError:
        return None
        
    # Check if expired
    if time.time() > expiry_time:
        return None
        
    # Recreate signature
    message = f"{audio_id}.{expiry_time}"
    expected_signature = hmac.new(
        settings.secret_key.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    if hmac.compare_digest(signature, expected_signature):
        return audio_id
        
    return None
