import os
import pytest
from pathlib import Path
from app.config import settings

def is_gemma_available() -> bool:
    return Path(settings.gemma_model_path).exists() or Path("models/gemma").exists()

run_real_smoke = os.environ.get("RUN_REAL_SMOKE", "0") == "1"

@pytest.mark.skipif(not run_real_smoke, reason="Real smoke tests are skipped by default")
def test_smoke_gemma():
    if run_real_smoke and not is_gemma_available():
        pytest.fail("RUN_REAL_SMOKE=1 but Gemma model is not available")
        
    from unittest.mock import patch
    from app.core.orchestrator import get_llm_provider
    
    with patch.object(settings, "mode", "dev"):
        try:
            llm = get_llm_provider()
            reply = llm.generate("Hello, who are you?", max_words=50)
            assert len(reply) > 0
        except Exception as e:
            pytest.fail(f"Gemma smoke test failed: {e}")
