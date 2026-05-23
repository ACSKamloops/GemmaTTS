from typing import Optional
from fastapi import APIRouter, HTTPException
from app.config import settings

router = APIRouter()

@router.post("/debug/rotate_key")
def rotate_key(new_key: str = ""):
    if not new_key or len(new_key) < 32:
        raise HTTPException(status_code=400, detail="Secret key must be at least 32 characters long")
    settings.secret_key = new_key
    return {"status": "key rotated"}

@router.post("/debug/update_settings")
def update_settings(
    max_cache_size_bytes: Optional[int] = None,
    max_file_size_bytes: Optional[int] = None
):
    if max_cache_size_bytes is not None:
        settings.max_cache_size_bytes = max_cache_size_bytes
    if max_file_size_bytes is not None:
        settings.max_file_size_bytes = max_file_size_bytes
    return {"status": "settings updated"}
