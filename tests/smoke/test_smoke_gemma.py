import pytest
from pathlib import Path
from app.config import settings

def is_gemma_available() -> bool:
    return Path(settings.gemma_model_path).exists() or Path("models/gemma").exists()

@pytest.mark.skipif(not is_gemma_available(), reason="Gemma model not available")
def test_smoke_gemma():
    from unittest.mock import patch
    from app.core.orchestrator import get_llm_provider
    
    with patch.object(settings, "mode", "dev"):
        llm = get_llm_provider()
        reply = llm.generate("Hello, who are you?", max_words=50)
        assert len(reply) > 0
