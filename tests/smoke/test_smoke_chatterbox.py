import pytest
from pathlib import Path
from app.config import settings

def is_chatterbox_available() -> bool:
    return Path("models/chatterbox/config.json").exists()

@pytest.mark.skipif(not is_chatterbox_available(), reason="Chatterbox model not available")
def test_smoke_chatterbox():
    from unittest.mock import patch
    from app.services.tts_service import get_worker
    
    with patch.object(settings, "mode", "dev"):
        worker = get_worker("chatterbox")
        wav_bytes, sample_rate = worker.synthesize("Hello, this is a Chatterbox smoke test.", "default")
        assert len(wav_bytes) > 1000
        assert sample_rate > 0
        assert wav_bytes.startswith(b"RIFF")
