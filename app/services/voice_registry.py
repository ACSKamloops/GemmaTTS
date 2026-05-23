"""
Voice Registry — catalogue of all available TTS voices across engines.

Provides:
  • VoiceInfo Pydantic model with metadata for each voice
  • Dynamic scanning of Kokoro .pt/.bin voice embeddings
  • Static definitions for Piper, Chatterbox, Dia, and F5-TTS voices
  • FastAPI router with GET /voices and GET /voices/{engine}
"""
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger("voice-registry")

# --------------- Constants ---------------
KOKORO_VOICES_DIR = os.environ.get(
    "KOKORO_VOICES_DIR", "models/kokoro/voices"
)

# --------------- Enums & Models ---------------

class Engine(str, Enum):
    chatterbox = "chatterbox"
    dia = "dia"
    f5_tts = "f5_tts"
    fish = "fish"
    kokoro = "kokoro"
    piper = "piper"


class VoiceInfo(BaseModel):
    """Metadata for a single TTS voice."""

    id: str = Field(..., description="Unique voice identifier used in synthesize requests")
    name: str = Field(..., description="Human-readable display name")
    engine: Engine = Field(..., description="TTS engine that owns this voice")
    language: str = Field(default="en", description="BCP-47 language code")
    gender: Optional[str] = Field(default=None, description="male / female / neutral / None")
    description: str = Field(default="", description="Short description of the voice")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    sample_rate: int = Field(default=24000, description="Output sample rate in Hz")


# --------------- Kokoro voice helpers ---------------

# Known Kokoro voice prefixes → language/gender mapping.
# Prefix convention: {lang_code}{gender_initial}_{name}
#   a = American English, b = British English, e = Spanish,
#   f = French, h = Hindi, i = Italian, j = Japanese,
#   p = Brazilian Portuguese, z = Mandarin Chinese
_KOKORO_LANG_MAP: dict[str, str] = {
    "a": "en-US",
    "b": "en-GB",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-BR",
    "z": "zh",
}

_KOKORO_GENDER_MAP: dict[str, str] = {
    "f": "female",
    "m": "male",
}


def _parse_kokoro_voice_id(voice_id: str) -> tuple[str, Optional[str]]:
    """Derive language and gender from a Kokoro voice ID prefix."""
    if len(voice_id) < 2:
        return "en", None
    lang_char = voice_id[0]
    gender_char = voice_id[1]
    language = _KOKORO_LANG_MAP.get(lang_char, "en")
    gender = _KOKORO_GENDER_MAP.get(gender_char)
    return language, gender


def _scan_kokoro_voices(voices_dir: str = KOKORO_VOICES_DIR) -> list[VoiceInfo]:
    """Scan the Kokoro voices directory for .pt and .bin embeddings."""
    voices_path = Path(voices_dir)
    if not voices_path.is_dir():
        logger.warning("Kokoro voices directory not found at %s", voices_dir)
        return []

    seen: set[str] = set()
    voices: list[VoiceInfo] = []

    for entry in sorted(voices_path.iterdir()):
        if entry.suffix not in (".pt", ".bin"):
            continue
        voice_id = entry.stem
        # Skip internal stubs
        if voice_id.startswith("_"):
            continue
        # Deduplicate — prefer .pt over .bin when both exist
        if voice_id in seen:
            continue
        seen.add(voice_id)

        language, gender = _parse_kokoro_voice_id(voice_id)
        voices.append(
            VoiceInfo(
                id=voice_id,
                name=voice_id.replace("_", " ").title(),
                engine=Engine.kokoro,
                language=language,
                gender=gender,
                description=f"Kokoro voice embedding ({entry.suffix})",
                tags=["kokoro", language],
                sample_rate=24000,
            )
        )

    logger.info("Kokoro: discovered %d voice(s) in %s", len(voices), voices_dir)
    return voices


# --------------- Static voice definitions ---------------

_PIPER_VOICES: list[VoiceInfo] = [
    VoiceInfo(
        id="en_US-lessac-medium",
        name="Lessac Medium",
        engine=Engine.piper,
        language="en-US",
        gender="neutral",
        description="Piper en_US-lessac-medium ONNX voice",
        tags=["piper", "en-US", "lessac"],
        sample_rate=22050,
    ),
]

_CHATTERBOX_VOICES: list[VoiceInfo] = [
    VoiceInfo(
        id="default",
        name="Chatterbox Default",
        engine=Engine.chatterbox,
        language="en",
        gender="neutral",
        description="Chatterbox default voice. Voice cloning supported via .wav reference file.",
        tags=["chatterbox", "voice-cloning"],
        sample_rate=24000,
    ),
]

_DIA_VOICES: list[VoiceInfo] = [
    VoiceInfo(
        id="S1",
        name="Dia Speaker 1",
        engine=Engine.dia,
        language="en",
        gender=None,
        description="Dia [S1] primary speaker. Use [S1] tag in text.",
        tags=["dia", "dialogue", "S1"],
        sample_rate=44100,
    ),
    VoiceInfo(
        id="S2",
        name="Dia Speaker 2",
        engine=Engine.dia,
        language="en",
        gender=None,
        description="Dia [S2] secondary speaker. Use [S2] tag in text.",
        tags=["dia", "dialogue", "S2"],
        sample_rate=44100,
    ),
]

_F5_TTS_VOICES: list[VoiceInfo] = [
    VoiceInfo(
        id="default",
        name="F5-TTS Default",
        engine=Engine.f5_tts,
        language="en",
        gender="neutral",
        description="F5-TTS default voice with reference audio cloning.",
        tags=["f5_tts", "voice-cloning"],
        sample_rate=24000,
    ),
]


# --------------- Registry class ---------------

class VoiceRegistry:
    """Central registry aggregating voices from all engines."""

    def __init__(self, kokoro_voices_dir: str = KOKORO_VOICES_DIR):
        self._kokoro_voices_dir = kokoro_voices_dir
        self._cache: list[VoiceInfo] | None = None

    def _build(self) -> list[VoiceInfo]:
        """Build the full voice list (scans filesystem for Kokoro)."""
        voices: list[VoiceInfo] = []
        voices.extend(_scan_kokoro_voices(self._kokoro_voices_dir))
        voices.extend(_PIPER_VOICES)
        voices.extend(_CHATTERBOX_VOICES)
        voices.extend(_DIA_VOICES)
        voices.extend(_F5_TTS_VOICES)
        return voices

    def list_all(self) -> list[VoiceInfo]:
        """Return all registered voices (cached after first call)."""
        if self._cache is None:
            self._cache = self._build()
        return self._cache

    def list_by_engine(self, engine: str) -> list[VoiceInfo]:
        """Return voices for a specific engine."""
        return [v for v in self.list_all() if v.engine.value == engine]

    def invalidate(self) -> None:
        """Clear the cache so the next call re-scans the filesystem."""
        self._cache = None


# Module-level default instance
registry = VoiceRegistry()


# --------------- FastAPI Router ---------------

router = APIRouter(tags=["voices"])


@router.get(
    "/voices",
    response_model=list[VoiceInfo],
    summary="List all available TTS voices",
)
def list_voices():
    """Return metadata for every voice across all TTS engines."""
    return registry.list_all()


@router.get(
    "/voices/{engine}",
    response_model=list[VoiceInfo],
    summary="List voices for a specific engine",
)
def list_voices_by_engine(engine: str):
    """Return metadata for voices belonging to the given engine."""
    # Validate engine name
    valid_engines = {e.value for e in Engine}
    if engine not in valid_engines:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown engine '{engine}'. Valid engines: {sorted(valid_engines)}",
        )
    voices = registry.list_by_engine(engine)
    return voices
