import os
import pytest
from app.config import settings

def is_chatterbox_available() -> bool:
    try:
        import chatterbox
        return True
    except ImportError:
        return False

run_real_smoke = os.environ.get("RUN_REAL_SMOKE", "0") == "1"

@pytest.mark.skipif(not run_real_smoke, reason="Real smoke tests are skipped by default")
def test_smoke_chatterbox():
    if run_real_smoke and not is_chatterbox_available():
        pytest.fail("RUN_REAL_SMOKE=1 but Chatterbox package is not installed")
        
    from unittest.mock import patch
    from app.services.tts_service import get_worker
    
    with patch.object(settings, "mode", "dev"):
        try:
            worker = get_worker("chatterbox")
            wav_bytes, sample_rate = worker.synthesize("Hello, this is a Chatterbox smoke test.", "default")
            assert len(wav_bytes) > 1000
            assert sample_rate > 0
            assert wav_bytes.startswith(b"RIFF")
        except Exception as e:
            pytest.fail(f"Chatterbox smoke test failed: {e}")
