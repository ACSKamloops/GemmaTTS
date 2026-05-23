from fastapi import FastAPI
from app.api.tts import router as tts_router, rate_limiter
from app.api.voices import router as voices_router
from typing import Any
import logging

logger = logging.getLogger("tts-service-wrapper")

# Define legacy workers map
_workers = {
    "kokoro": None,
    "piper": None,
    "chatterbox": None,
    "dia": None,
    "f5_tts": None
}

def get_worker(engine: str) -> Any:
    # 1. First, check if the test has overridden _workers[engine]
    if _workers.get(engine) is not None:
        return _workers[engine]
        
    # 2. Trigger dynamic import checks for tests patching modules (e.g. piper)
    if engine == "piper":
        import app.services.tts.piper_worker as piper_worker
    elif engine == "kokoro":
        import app.services.tts.kokoro_worker as kokoro_worker
    elif engine == "chatterbox":
        import app.services.tts.chatterbox_worker as chatterbox_worker
    elif engine == "dia":
        import app.services.tts.dia_worker as dia_worker
    elif engine == "f5_tts":
        import app.services.tts.f5_tts_worker as f5_tts_worker

    # Return the cached provider from the orchestrator
    from app.core.orchestrator import get_tts_provider
    return get_tts_provider(engine)

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy", "service": "tts-service"}

app.include_router(tts_router)
app.include_router(voices_router)
