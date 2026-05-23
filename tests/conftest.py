import pytest
import shutil
from pathlib import Path
from app.config import settings

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    # Override settings to use a local test cache directory and smaller caps
    test_cache_dir = Path("tests/test_audio_cache").resolve()
    settings.audio_cache_dir = test_cache_dir
    settings.max_cache_size_bytes = 1024 * 1024 # 1 MB test limit
    settings.max_file_size_bytes = 100 * 1024 # 100 KB test limit
    settings.secret_key = "test-secret-key-for-hmac-verification-operations"
    
    if test_cache_dir.exists():
        shutil.rmtree(test_cache_dir)
    test_cache_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    
    if test_cache_dir.exists():
        shutil.rmtree(test_cache_dir)
