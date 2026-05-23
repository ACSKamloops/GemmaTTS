from typing import List, Optional
from fastapi import APIRouter, Request, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.core.orchestrator import dialogue_orchestrator

router = APIRouter(tags=["dialogue"])

class Speaker(BaseModel):
    id: str
    name: str
    voice_id: str
    style: Optional[str] = None

class Fact(BaseModel):
    id: str
    can_reveal: bool
    fact: str

class Context(BaseModel):
    location: Optional[str] = None
    facts: Optional[List[Fact]] = None

class OutputConfig(BaseModel):
    audio: bool
    format: str

class DialogueRequest(BaseModel):
    request_id: Optional[str] = None
    speaker: Speaker
    context: Optional[Context] = None
    user_text: str
    max_words: Optional[int] = 150
    output: OutputConfig
    test_mode: Optional[bool] = None  # Ignored or rejected in real mode

class AudioMetadata(BaseModel):
    audio_id: str
    sha256: str
    bytes: int
    duration_ms: int
    format: str
    sample_rate: int

class Metrics(BaseModel):
    queue_ms: float
    llm_ms: float
    tts_ms: float
    encode_ms: float
    total_ms: float
    cache_hit: bool

class DialogueResponse(BaseModel):
    job_id: str
    state: str
    text: str
    audio: Optional[AudioMetadata] = None
    metrics: Metrics

@router.post("/v1/dialogue", response_model=DialogueResponse)
async def post_dialogue(
    req: DialogueRequest,
    request: Request,
    cache_control: Optional[str] = Header(None)
):
    if req.output.audio and req.output.format not in ("wav", "ogg", "mp3"):
        raise HTTPException(status_code=422, detail=f"Unsupported format: {req.output.format}")

    # If in real mode, reject any test/simulation triggers
    if settings.mode == "real":
        if req.test_mode is not None:
            raise HTTPException(status_code=400, detail="test_mode parameter is forbidden in production/real mode.")
            
        simulation_keywords = [
            "simulate_llm_crash", "simulate_client_disconnect", "simulate_offline",
            "simulate-llm-bad-json", "simulate_llm_bad_json", "simulate_llm_failed_status"
        ]
        if any(kw in req.user_text for kw in simulation_keywords):
            raise HTTPException(status_code=400, detail="Simulation keywords are forbidden in production/real mode.")
            
        if req.speaker.voice_id and any(kw in req.speaker.voice_id for kw in ("simulate_offline", "enable_fish")):
            raise HTTPException(status_code=400, detail="Simulation keywords are forbidden in production/real mode.")
            
    # Resolve speaker context info
    location = req.context.location if req.context else None
    facts_list = []
    if req.context and req.context.facts:
        facts_list = [{"id": f.id, "can_reveal": f.can_reveal, "fact": f.fact} for f in req.context.facts]

    # Execute dialogue pipeline
    async def is_disconnected():
        return await request.is_disconnected()

    result = await dialogue_orchestrator.execute_dialogue(
        user_text=req.user_text,
        speaker_id=req.speaker.id,
        speaker_name=req.speaker.name,
        voice_id=req.speaker.voice_id,
        style=req.speaker.style,
        location=location,
        facts=facts_list,
        max_words=req.max_words,
        audio_enabled=req.output.audio,
        audio_format=req.output.format,
        cache_control=cache_control,
        client_disconnect_check=is_disconnected
    )

    if result.get("state") == "failed":
        err_msg = result.get("error", "Dialogue processing failed")
        if err_msg == "llm_schema_mismatch":
            raise HTTPException(status_code=500, detail="LLM output schema mismatch")
        raise HTTPException(status_code=503, detail="Service Unavailable")
        
    if result.get("state") == "canceled":
        # Request disconnected
        from fastapi import Response
        return Response(status_code=499)

    return DialogueResponse(**result)
