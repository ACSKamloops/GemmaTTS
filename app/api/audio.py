import urllib.parse
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.audio.cache import AudioCacheManager, is_safe_path
from app.audio.signer import verify_signed_audio_id

router = APIRouter(tags=["audio"])
cache_manager = AudioCacheManager()

@router.get("/audio/{signed_id:path}")
def get_audio(signed_id: str, format: Optional[str] = None):
    # Decode URL-encoded characters
    decoded_id = urllib.parse.unquote(signed_id)
    
    # Path traversal validation
    if ".." in decoded_id or "/" in decoded_id or "\\" in decoded_id:
        raise HTTPException(status_code=400, detail="Path traversal or out-of-boundary access detected.")
        
    # Verify HMAC signature and expiration
    verified = verify_signed_audio_id(decoded_id)
    if not verified:
        raise HTTPException(status_code=403, detail="Signature expired or invalid")
        
    if "_" in verified:
        key, format_str = verified.rsplit("_", 1)
    else:
        key = verified
        format_str = format or "wav"
        
    try:
        path = cache_manager.get_file_path(key, format_str)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # Check symlink safety
    if path.is_symlink():
        real_path = path.resolve()
        if not is_safe_path(real_path, cache_manager.cache_dir):
            raise HTTPException(status_code=403, detail="Symlink targets outside cache directory.")
            
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    media_types = {
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
        "pcm": "audio/l16"
    }
    media_type = media_types.get(format_str.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type)
