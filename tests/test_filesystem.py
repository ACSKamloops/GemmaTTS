import pytest
import os
from pathlib import Path
from app.audio.cache import AudioCacheManager, is_safe_path
from app.config import settings

def test_is_safe_path():
    base = Path("tests/test_audio_cache").resolve()
    assert is_safe_path(base / "abc.ogg", base)
    assert not is_safe_path(base / "../abc.ogg", base)
    assert not is_safe_path(Path("/etc/passwd"), base)

def test_cache_put_and_get():
    manager = AudioCacheManager()
    text = "Hello world"
    voice = "af_heart"
    fmt = "ogg"
    
    data = b"dummy-audio-bytes"
    path = manager.put(text, voice, fmt, data)
    assert path.exists()
    
    cached = manager.get(text, voice, fmt)
    assert cached == data

def test_path_traversal_prevention():
    manager = AudioCacheManager()
    
    with pytest.raises(PermissionError):
        manager.get_file_path("../../etc/passwd", "ogg")
        
    with pytest.raises(PermissionError):
        manager.get_file_path("key", "../passwd")

def test_symlink_prevention(tmp_path):
    manager = AudioCacheManager()
    
    # Create external file
    external_file = tmp_path / "secret.txt"
    external_file.write_bytes(b"sensitive-data")
    
    symlink_path = manager.cache_dir / "symlink_test.ogg"
    if symlink_path.exists():
        symlink_path.unlink()
        
    try:
        os.symlink(external_file, symlink_path)
    except OSError:
        pass
        
    if symlink_path.is_symlink():
        with pytest.raises(PermissionError):
            manager.get_file_path("symlink_test", "ogg")
            
def test_max_file_size():
    manager = AudioCacheManager()
    # 100 KB is our test limit (set in conftest)
    oversized_data = b"x" * (101 * 1024)
    with pytest.raises(ValueError):
        manager.put("Oversized", "voice", "ogg", oversized_data)

def test_cache_pruning():
    manager = AudioCacheManager()
    
    for f in manager.cache_dir.iterdir():
        if f.is_file():
            f.unlink()
            
    # max_cache_size_bytes is 1 MB (1024 KB). max_file_size_bytes is 100 KB.
    # We write 15 files of 80 KB each (total 1200 KB).
    chunk = b"y" * (80 * 1024)
    
    paths = []
    for i in range(15):
        path = manager.put(f"text_{i}", "voice", "ogg", chunk)
        paths.append(path)
        
    # The first file should have been pruned
    assert not paths[0].exists()
    # The latest file should exist
    assert paths[14].exists()
