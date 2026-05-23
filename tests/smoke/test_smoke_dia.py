import os
import pytest
from pathlib import Path
from app.config import settings

def is_dia_available() -> bool:
    return Path("models/dia/config.json").exists()

run_real_smoke = os.environ.get("RUN_REAL_SMOKE", "0") == "1"

@pytest.mark.skipif(not run_real_smoke, reason="Real smoke tests are skipped by default")
def test_smoke_dia():
    if run_real_smoke and not is_dia_available():
        pytest.fail("RUN_REAL_SMOKE=1 but Dia model is not available at models/dia/config.json")
        
    from unittest.mock import patch
    from app.services.tts_service import get_worker
    
    with patch.object(settings, "mode", "dev"):
        try:
            worker = get_worker("dia")
            wav_bytes, sample_rate = worker.synthesize("Hello, this is a Dia smoke test.", "default")
            assert len(wav_bytes) > 1000
            assert sample_rate > 0
            assert wav_bytes.startswith(b"RIFF")
        except Exception as e:
            pytest.fail(f"Dia smoke test failed: {e}")
