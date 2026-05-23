import time
import collections
import logging
import threading
from typing import Optional, Any
from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field, field_validator

from app.config import settings

router = APIRouter(tags=["generate"])
logger = logging.getLogger("generate-api")

class ThreadSafeRateLimiter:
    def __init__(self, limit: int = 40, window_seconds: float = 1.0):
        self.limit = limit
        self.window_seconds = window_seconds
        self.timestamps = collections.deque()
        self.lock = threading.Lock()

    def is_allowed(self) -> bool:
        now = time.time()
        with self.lock:
            while self.timestamps and self.timestamps[0] < now - self.window_seconds:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.limit:
                return False
            self.timestamps.append(now)
            return True

rate_limiter = ThreadSafeRateLimiter(limit=40, window_seconds=1.0)

class GenerateRequest(BaseModel):
    prompt: str
    max_words: Optional[int] = 150
    enable_thinking: Optional[bool] = False

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not isinstance(v, str) or v.strip() == "":
            raise ValueError("Prompt cannot be empty")
        return v

    @field_validator('max_words', mode='before')
    @classmethod
    def validate_max_words(cls, v: Any) -> Any:
        if v is not None:
            if isinstance(v, bool):
                raise ValueError("max_words cannot be a boolean")
            if isinstance(v, float):
                raise ValueError("max_words must be an integer")
            if isinstance(v, str):
                try:
                    val = int(v)
                except ValueError:
                    raise ValueError("max_words must be an integer")
            elif isinstance(v, int):
                val = v
            else:
                raise ValueError("max_words must be an integer")

            if val < 0:
                raise ValueError("max_words cannot be negative")
            return val
        return v

class GenerateResponse(BaseModel):
    text: str
    generation_time_ms: float

@router.post("/generate", response_model=GenerateResponse)
def post_generate(req: GenerateRequest):
    if not rate_limiter.is_allowed():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too Many Requests"
        )

    # Real mode violation check
    if settings.mode == "real":
        if "simulate_llm_bad_json" in req.prompt:
            raise HTTPException(status_code=400, detail="Simulation keywords are forbidden in production/real mode.")

    start_time = time.time()

    # Simulation triggers
    if "simulate_llm_bad_json" in req.prompt:
        return Response(content="not-a-json-string-at-all", media_type="text/plain")

    # Limit prompt input length if exceeding threshold
    prompt_text = req.prompt
    if len(prompt_text) > 5000:
        prompt_text = prompt_text[:settings.max_text_chars]

    # Resolve LLM provider
    from app.core.orchestrator import get_llm_provider
    llm = get_llm_provider()

    try:
        reply = llm.generate(prompt_text, req.max_words, req.enable_thinking)
        generation_time_ms = (time.time() - start_time) * 1000.0
        return GenerateResponse(text=reply, generation_time_ms=generation_time_ms)
    except Exception as e:
        logger.error(f"Text generation failure: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
