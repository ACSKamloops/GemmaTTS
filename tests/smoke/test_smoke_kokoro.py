import pytest
from pathlib import Path
from app.config import settings

def is_kokoro_available() -> bool:
    return Path("models/kokoro/onnx/model.onnx").exists()

@pytest.mark.skipif(not is_kokoro_available(), reason="Kokoro model not available")
def test_smoke_kokoro():
    from unittest.mock import patch
    from app.services.tts_service import get_worker
    
    with patch.object(settings, "mode", "dev"):
        worker = get_worker("kokoro")
        wav_bytes, sample_rate = worker.synthesize("Hello, this is a Kokoro smoke test.", "af_heart")
        assert len(wav_bytes) > 1000
        assert sample_rate > 0
        assert wav_bytes.startswith(b"RIFF")
