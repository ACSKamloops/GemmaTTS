import os
import pytest
from pathlib import Path
from app.config import settings

def is_kokoro_available() -> bool:
    return Path("models/kokoro/onnx/model.onnx").exists()

run_real_smoke = os.environ.get("RUN_REAL_SMOKE", "0") == "1"

@pytest.mark.skipif(not run_real_smoke, reason="Real smoke tests are skipped by default")
def test_smoke_kokoro():
    if run_real_smoke and not is_kokoro_available():
        pytest.fail("RUN_REAL_SMOKE=1 but Kokoro model is not available at models/kokoro/onnx/model.onnx")
        
    from unittest.mock import patch
    from app.services.tts_service import get_worker
    
    with patch.object(settings, "mode", "dev"):
        try:
            worker = get_worker("kokoro")
            wav_bytes, sample_rate = worker.synthesize("Hello, this is a Kokoro smoke test.", "af_heart")
            assert len(wav_bytes) > 1000
            assert sample_rate > 0
            assert wav_bytes.startswith(b"RIFF")
        except Exception as e:
            pytest.fail(f"Kokoro smoke test failed: {e}")
