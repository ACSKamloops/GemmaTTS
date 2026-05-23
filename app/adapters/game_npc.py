"""
Game / NPC adapter contract for GemmaTTS.

Defines engine-agnostic data models (NPCRequest, NPCResponse) and a
translation layer (GameNPCAdapter) that converts them into the
orchestrator's DialogueRequest / DialogueResponse format.

Design goals
------------
* No game-engine-specific networking or filesystem assumptions.
* Interaction types are intentionally open-ended strings so that new
  game genres can extend them without modifying this module.
* Batch mode dispatches requests concurrently via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InteractionType(str, Enum):
    """Common interaction archetypes.  Consumers may pass any string; this
    enum exists for discoverability and auto-complete, not enforcement."""
    GREETING = "greeting"
    COMBAT   = "combat"
    TRADE    = "trade"
    QUEST    = "quest"
    IDLE     = "idle"
    BARK     = "bark"          # ambient / one-liner
    FAREWELL = "farewell"


# ---------------------------------------------------------------------------
# Data contracts – request
# ---------------------------------------------------------------------------

class NPCContext(BaseModel):
    """Environmental / situational context for the actor."""
    location: Optional[str] = None
    time_of_day: Optional[str] = None
    mood: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class ListenerContext(BaseModel):
    """Optional metadata about the player / listener."""
    id: Optional[str] = None
    name: Optional[str] = None
    reputation: Optional[float] = None
    faction: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class NPCRequest(BaseModel):
    """Inbound request describing what a single NPC should say."""
    actor_id: str
    actor_name: str
    voice_id: str
    context: Optional[NPCContext] = None
    listener_id: Optional[str] = None
    listener_context: Optional[ListenerContext] = None
    interaction_type: str = InteractionType.GREETING
    user_input: str = ""
    max_response_words: int = Field(default=60, ge=1, le=500)
    # Caller-assigned tag so responses can be matched in async / batch flows
    request_tag: Optional[str] = None
    # Optional TTS style hint forwarded as speaker.style to orchestrator
    tts_style: Optional[str] = None


# ---------------------------------------------------------------------------
# Data contracts – response
# ---------------------------------------------------------------------------

class NPCResponse(BaseModel):
    """Outbound payload returned to the game engine."""
    actor_id: str
    text: str
    emotion: Optional[str] = None
    audio_id: Optional[str] = None
    audio_url: Optional[str] = None
    duration_ms: int = 0
    request_tag: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class NPCBatchResponse(BaseModel):
    """Wrapper returned by batch_synthesize."""
    responses: List[NPCResponse]
    total_duration_ms: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GameNPCAdapter:
    """Translates NPC requests into the orchestrator's dialogue API.

    Parameters
    ----------
    orchestrator_url : str
        Base URL for the orchestrator gateway (e.g. ``http://127.0.0.1:8000``).
    audio_format : str
        Default output audio format (``wav``, ``ogg``, ``mp3``).
    timeout : float
        Per-request HTTP timeout in seconds.
    request_audio : bool
        Whether to ask the orchestrator for audio synthesis (set ``False``
        for text-only testing).
    """

    def __init__(
        self,
        orchestrator_url: str = "http://127.0.0.1:8000",
        audio_format: str = "wav",
        timeout: float = 15.0,
        request_audio: bool = True,
    ) -> None:
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.audio_format = audio_format
        self.timeout = timeout
        self.request_audio = request_audio

    # ---- helpers -----------------------------------------------------------

    def _build_prompt(self, req: NPCRequest) -> str:
        """Compose the text prompt forwarded to the LLM.

        Embeds context hints so that Gemma can produce in-character output.
        """
        parts: list[str] = []

        # System-level context block
        parts.append(
            f"You are {req.actor_name} (id={req.actor_id}). "
            f"Respond in character."
        )

        if req.context:
            ctx_parts: list[str] = []
            if req.context.location:
                ctx_parts.append(f"location: {req.context.location}")
            if req.context.time_of_day:
                ctx_parts.append(f"time: {req.context.time_of_day}")
            if req.context.mood:
                ctx_parts.append(f"mood: {req.context.mood}")
            if ctx_parts:
                parts.append(f"Scene context — {', '.join(ctx_parts)}.")

        parts.append(f"Interaction type: {req.interaction_type}.")

        if req.listener_context:
            lc = req.listener_context
            lc_parts: list[str] = []
            if lc.name:
                lc_parts.append(f"name={lc.name}")
            if lc.faction:
                lc_parts.append(f"faction={lc.faction}")
            if lc.reputation is not None:
                lc_parts.append(f"rep={lc.reputation}")
            if lc_parts:
                parts.append(f"Listener: {', '.join(lc_parts)}.")

        if req.user_input:
            parts.append(f'The listener says: "{req.user_input}"')

        return " ".join(parts)

    def _to_dialogue_payload(self, req: NPCRequest) -> dict:
        """Build the JSON body for ``POST /v1/dialogue``."""
        return {
            "request_id": req.request_tag or str(uuid.uuid4()),
            "speaker": {
                "id": req.actor_id,
                "name": req.actor_name,
                "voice_id": req.voice_id,
                "style": req.tts_style,
            },
            "context": {
                "location": req.context.location if req.context else None,
            },
            "user_text": self._build_prompt(req),
            "max_words": req.max_response_words,
            "output": {
                "audio": self.request_audio,
                "format": self.audio_format,
            },
            "test_mode": False,
        }

    def _parse_response(self, req: NPCRequest, data: dict) -> NPCResponse:
        """Map an orchestrator DialogueResponse dict → NPCResponse."""
        audio_meta = data.get("audio")
        audio_id = audio_meta["audio_id"] if audio_meta else None
        duration_ms = audio_meta["duration_ms"] if audio_meta else 0
        audio_url: Optional[str] = None
        if audio_id:
            audio_url = f"{self.orchestrator_url}/audio/{audio_id}"

        return NPCResponse(
            actor_id=req.actor_id,
            text=data.get("text", ""),
            emotion=req.context.mood if req.context else None,
            audio_id=audio_id,
            audio_url=audio_url,
            duration_ms=duration_ms,
            request_tag=req.request_tag,
            metadata={
                "job_id": data.get("job_id"),
                "state": data.get("state"),
                "metrics": data.get("metrics"),
            },
        )

    # ---- public API --------------------------------------------------------

    async def synthesize(self, req: NPCRequest) -> NPCResponse:
        """Send a single NPC request to the orchestrator and return the
        translated response.

        Raises
        ------
        httpx.HTTPStatusError
            When the orchestrator responds with a non-2xx status.
        httpx.TimeoutException
            When the request exceeds *timeout*.
        """
        payload = self._to_dialogue_payload(req)
        logger.debug("NPC request → orchestrator: actor_id=%s", req.actor_id)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.orchestrator_url}/v1/dialogue",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        npc_resp = self._parse_response(req, data)
        logger.info(
            "NPC response: actor_id=%s duration_ms=%d text_len=%d",
            npc_resp.actor_id,
            npc_resp.duration_ms,
            len(npc_resp.text),
        )
        return npc_resp

    async def batch_synthesize(
        self,
        requests: List[NPCRequest],
        *,
        max_concurrency: int = 4,
    ) -> NPCBatchResponse:
        """Process multiple NPC requests concurrently.

        Parameters
        ----------
        requests : list[NPCRequest]
            Batch of NPC dialogue requests.
        max_concurrency : int
            Maximum number of orchestrator calls in flight at once.

        Returns
        -------
        NPCBatchResponse
            Aggregated responses and any per-request errors.
        """
        semaphore = asyncio.Semaphore(max_concurrency)
        responses: list[NPCResponse] = []
        errors: list[dict[str, Any]] = []

        async def _one(req: NPCRequest) -> None:
            async with semaphore:
                try:
                    resp = await self.synthesize(req)
                    responses.append(resp)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Batch item failed: actor_id=%s error=%s",
                        req.actor_id,
                        exc,
                    )
                    errors.append({
                        "actor_id": req.actor_id,
                        "request_tag": req.request_tag,
                        "error": str(exc),
                    })

        await asyncio.gather(*(_one(r) for r in requests))

        total_dur = sum(r.duration_ms for r in responses)
        return NPCBatchResponse(
            responses=responses,
            total_duration_ms=total_dur,
            errors=errors,
        )
