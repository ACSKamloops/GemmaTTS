import hashlib
import os
from pathlib import Path
from typing import Optional
from app.config import settings

def get_cache_key(text: str, voice_id: str, format: str) -> str:
    """
    Generate a SHA-256 hash representing text, voice, and format.
    """
    payload = f"{text}:{voice_id}:{format}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def is_safe_path(path: Path, base_dir: Path) -> bool:
    """
    Ensures that the resolved absolute path resides strictly within the base directory,
    preventing path traversal and symlink-based jailbreaks.
    """
    try:
        resolved_base = base_dir.resolve()
        # Resolve path including symlinks
        if path.exists():
            resolved_path = path.resolve()
        else:
            resolved_path = Path(os.path.abspath(path)).resolve()
            
        return resolved_base in resolved_path.parents or resolved_base == resolved_path
    except (ValueError, OSError):
        return False

class AudioCacheManager:
    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            
    @property
    def cache_dir(self) -> Path:
        resolved = self._cache_dir or settings.audio_cache_dir
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved
        
    def get_file_path(self, key: str, format: str) -> Path:
        """
        Get absolute path to a cache file. Verifies safety boundaries before returning.
        """
        # Directly reject path traversal delimiters in the input parameters
        if "/" in key or "\\" in key or ".." in key or "/" in format or "\\" in format or ".." in format:
            raise PermissionError("Path traversal or out-of-boundary access detected.")
            
        safe_key = "".join(c for c in key if c.isalnum() or c in ("-", "_"))
        safe_format = "".join(c for c in format if c.isalnum())
        
        if not safe_key or not safe_format:
            raise ValueError("Empty key or format after sanitization")
            
        path = (self.cache_dir / f"{safe_key}.{safe_format}").resolve()
        
        if not is_safe_path(path, self.cache_dir):
            raise PermissionError("Path traversal or out-of-boundary access detected.")
            
        return path
        
    def get(self, text: str, voice_id: str, format: str) -> Optional[bytes]:
        """
        Retrieves cached audio file if it exists and is valid.
        """
        key = get_cache_key(text, voice_id, format)
        try:
            path = self.get_file_path(key, format)
            if path.exists():
                # Enforce no symlinks pointing outside the cache directory
                if path.is_symlink():
                    real_path = path.resolve()
                    if not is_safe_path(real_path, self.cache_dir):
                        raise PermissionError("Symlink targets outside cache directory.")
                return path.read_bytes()
        except Exception:
            pass
        return None
        
    def get_metadata(self, text: str, voice_id: str, format: str) -> Optional[dict]:
        """
        Retrieves cached metadata associated with a cache key.
        """
        key = get_cache_key(text, voice_id, format)
        try:
            path = self.get_file_path(key, format)
            meta_path = path.with_suffix(path.suffix + ".json")
            if meta_path.exists():
                import json
                return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None

    def put(self, text: str, voice_id: str, format: str, data: bytes, duration_ms: Optional[int] = None) -> Path:
        """
        Caches audio data, pruning old entries as needed and enforcing size limits.
        """
        if settings.max_cache_size_bytes < 0:
            raise ValueError("max_cache_size_bytes must be non-negative")
            
        if len(data) > settings.max_cache_size_bytes:
            raise ValueError("incoming data size exceeds max_cache_size_bytes")

        if len(data) > settings.max_file_size_bytes:
            raise ValueError(f"File size exceeds max_file_size_bytes ({settings.max_file_size_bytes}).")
            
        key = get_cache_key(text, voice_id, format)
        path = self.get_file_path(key, format)
        
        self.prune_cache(len(data))
        
        path.write_bytes(data)

        if duration_ms is not None:
            import json
            meta_path = path.with_suffix(path.suffix + ".json")
            meta_path.write_text(json.dumps({"duration_ms": duration_ms}), encoding="utf-8")

        return path
        
    def prune_cache(self, incoming_bytes: int):
        """
        Removes oldest cache files if the cache size exceeds maximum limits.
        """
        max_size = settings.max_cache_size_bytes
        if incoming_bytes > max_size:
            return
            
        files = []
        total_size = 0
        
        for file in self.cache_dir.iterdir():
            if file.is_file() and not file.is_symlink() and not file.name.endswith(".json"):
                try:
                    stat = file.stat()
                    files.append((stat.st_mtime_ns, stat.st_size, file))
                    total_size += stat.st_size
                except OSError:
                    pass
                    
        # Sort oldest first
        files.sort()
        
        while total_size + incoming_bytes > max_size and files:
            _, size, file_path = files.pop(0)
            try:
                meta_path = file_path.with_suffix(file_path.suffix + ".json")
                if meta_path.exists():
                    try:
                        meta_path.unlink()
                    except OSError:
                        pass
                file_path.unlink()
                total_size -= size
            except OSError:
                pass
