from fastapi import FastAPI

from app.config import settings
from app.middleware.auth import AuthMiddleware
from app.dashboard.router import router as dashboard_router
from app.api import health, dialogue, tts, jobs, audio, voices, generate

app = FastAPI(
    title="GemmaTTS Service",
    description="Unified single-process Gemma 4 LLM and Multi-engine Speech Synthesis stack.",
    version="1.0.0"
)

import os

if settings.mode == "real" and not os.getenv("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY is required in MODE=real")

# Apply global authentication middleware
app.add_middleware(AuthMiddleware)

# Include core API routers
app.include_router(health.router)
app.include_router(dialogue.router)
app.include_router(tts.router)
app.include_router(jobs.router)
app.include_router(audio.router)
app.include_router(voices.router)
app.include_router(generate.router)

if settings.debug_enabled:
    from app.api import debug
    app.include_router(debug.router)

# Mount the web dashboard router
app.include_router(dashboard_router, prefix="/dashboard")
