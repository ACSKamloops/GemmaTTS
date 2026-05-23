"""
Dashboard router — web UI for monitoring GemmaTTS services.

Provides:
  • GET  /            — HTML dashboard page
  • GET  /api/status  — JSON status for AJAX polling
  • GET  /api/voices  — Voice listing
  • POST /api/preview — Quick TTS preview
"""

import base64
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from app.audio.cache import AudioCacheManager
from app.config import settings

logger = logging.getLogger("dashboard")

router = APIRouter(tags=["dashboard"])

# --------------- In-memory job history ---------------
MAX_HISTORY = 20
_job_history: deque[dict] = deque(maxlen=MAX_HISTORY)


def record_job(job: dict) -> None:
    """Append a job record to the in-memory history ring buffer."""
    _job_history.appendleft(job)


# --------------- Helpers ---------------

async def _probe_service(url: str, timeout: float = 2.0) -> dict:
    """Ping a service health endpoint. Returns status dict."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=timeout)
        if resp.status_code == 200:
            return {"status": "healthy", "latency_ms": resp.elapsed.total_seconds() * 1000}
        return {"status": "unhealthy", "code": resp.status_code}
    except httpx.ConnectError:
        return {"status": "offline", "error": "connection refused"}
    except httpx.TimeoutException:
        return {"status": "timeout", "error": "request timed out"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _cache_stats() -> dict:
    """Gather cache directory statistics."""
    cache_dir = settings.audio_cache_dir
    if not cache_dir.exists():
        return {"total_files": 0, "total_bytes": 0, "total_mb": "0.00", "max_mb": f"{settings.max_cache_size_bytes / (1024 * 1024):.1f}"}

    total_bytes = 0
    file_count = 0
    for f in cache_dir.iterdir():
        if f.is_file():
            try:
                total_bytes += f.stat().st_size
                file_count += 1
            except OSError:
                pass

    return {
        "total_files": file_count,
        "total_bytes": total_bytes,
        "total_mb": f"{total_bytes / (1024 * 1024):.2f}",
        "max_mb": f"{settings.max_cache_size_bytes / (1024 * 1024):.1f}",
    }


# --------------- Pydantic models ---------------

class PreviewRequest(BaseModel):
    text: str
    engine: str = "kokoro"
    voice_id: str = "default"


# --------------- Routes ---------------

@router.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the single-page dashboard HTML."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if not template_path.exists():
        raise HTTPException(status_code=500, detail="Dashboard template not found")
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))


@router.get("/api/status")
async def api_status():
    """JSON status endpoint polled by the dashboard via AJAX."""
    orchestrator = await _probe_service(f"http://127.0.0.1:{settings.port}/health")
    gemma = await _probe_service("http://127.0.0.1:8001/health")
    tts = await _probe_service("http://127.0.0.1:8002/health")

    engines = ["chatterbox", "dia", "f5_tts", "fish", "kokoro", "piper"]

    return JSONResponse({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "orchestrator": orchestrator,
            "gemma": gemma,
            "tts": tts,
        },
        "engines": engines,
        "default_engine": settings.default_tts_engine,
        "cache": _cache_stats(),
        "recent_jobs": list(_job_history),
    })


@router.get("/api/voices")
async def api_voices():
    """Proxy voice list from the TTS service or return static fallback."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://127.0.0.1:8002/voices", timeout=3.0)
        if resp.status_code == 200:
            return JSONResponse(resp.json())
    except Exception as exc:
        logger.warning("Could not fetch voices from TTS service: %s", exc)

    # Fallback: use the local voice registry
    try:
        from app.services.voice_registry import registry
        voices = [v.model_dump() for v in registry.list_all()]
        return JSONResponse(voices)
    except Exception as exc:
        logger.error("Voice registry fallback failed: %s", exc)
        return JSONResponse([])


@router.post("/api/preview")
async def api_preview(req: PreviewRequest):
    """Synthesize a short text clip and return base64 audio for playback."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")

    if len(req.text) > 500:
        raise HTTPException(status_code=422, detail="Preview text limited to 500 characters")

    t_start = time.time()
    job_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": req.text[:80],
        "engine": req.engine,
        "voice_id": req.voice_id,
        "status": "pending",
    }

    try:
        payload = {
            "text": req.text,
            "voice_id": req.voice_id,
            "engine": req.engine,
            "test_mode": True,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "http://127.0.0.1:8002/synthesize",
                json=payload,
                timeout=10.0,
            )

        if resp.status_code != 200:
            job_record["status"] = "error"
            job_record["error"] = f"TTS returned {resp.status_code}"
            record_job(job_record)
            raise HTTPException(status_code=502, detail=f"TTS service returned {resp.status_code}")

        data = resp.json()
        elapsed_ms = (time.time() - t_start) * 1000

        job_record["status"] = "done"
        job_record["duration_ms"] = round(elapsed_ms, 1)
        record_job(job_record)

        return JSONResponse({
            "audio_base64": data["audio_bytes_base64"],
            "format": data.get("format", "wav"),
            "sample_rate": data.get("sample_rate", 24000),
            "synthesis_ms": round(elapsed_ms, 1),
        })

    except HTTPException:
        raise
    except httpx.ConnectError:
        job_record["status"] = "error"
        job_record["error"] = "TTS service offline"
        record_job(job_record)
        raise HTTPException(status_code=503, detail="TTS service is not running")
    except Exception as exc:
        job_record["status"] = "error"
        job_record["error"] = str(exc)
        record_job(job_record)
        logger.error("Preview synthesis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
