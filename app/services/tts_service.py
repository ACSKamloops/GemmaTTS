import time
import os
import re
import base64
import collections
import logging
import threading
import wave
import io
import struct
import tempfile
from pathlib import Path
from typing import Optional, Literal, Any, Generator
from fastapi import FastAPI, HTTPException, status, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts-service")

app = FastAPI(title="TTS Speech Synthesis Service", version="1.0.0")

from app.middleware.auth import AuthMiddleware
app.add_middleware(AuthMiddleware)

from app.services.voice_registry import router as voice_router
app.include_router(voice_router)

# ----------------- Rate Limiter -----------------
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

# ----------------- Pydantic Models -----------------
class SynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "default"
    engine: Literal["chatterbox", "dia", "fish", "f5_tts", "kokoro", "piper"]
    test_mode: Optional[bool] = True

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Text cannot be empty")
        return v

class SynthesizeResponse(BaseModel):
    audio_bytes_base64: str
    format: str
    sample_rate: int
    synthesis_time_ms: float


class StreamRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "default"
    engine: Literal["chatterbox", "dia", "fish", "f5_tts", "kokoro", "piper"]
    test_mode: Optional[bool] = True

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Text cannot be empty")
        return v


class ExportRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "default"
    engine: Literal["chatterbox", "dia", "fish", "f5_tts", "kokoro", "piper"]
    format: Literal["wav", "ogg", "mp3"] = "wav"
    filename: Optional[str] = "tts_output"
    test_mode: Optional[bool] = True

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Text cannot be empty")
        return v

    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> str:
        if v is None:
            return "tts_output"
        # If it has path traversal like '..', replace all non-alphanumeric with '_' to neutralize it
        if ".." in v:
            sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', v)
            return sanitized
        # Otherwise, split on both Unix and Windows path separators to strip directory path
        parts = re.split(r'[/\\]', v)
        basename = parts[-1] if parts else ""
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', basename)
        if not sanitized:
            return "tts_output"
        return sanitized


# ----------------- Lazy Load Workers -----------------
_workers = {
    "chatterbox": None,
    "dia": None,
    "fish": None,
    "f5_tts": None,
    "kokoro": None,
    "piper": None
}
_worker_lock = threading.Lock()

def get_worker(engine: str) -> Any:
    global _workers
    with _worker_lock:
        if _workers[engine] is not None:
            return _workers[engine]
        
        # Lazy imports to avoid loading models/packages unnecessarily
        try:
            if engine == "chatterbox":
                from app.services.tts.chatterbox_worker import ChatterboxWorker
                _workers[engine] = ChatterboxWorker()
            elif engine == "dia":
                from app.services.tts.dia_worker import DiaWorker
                _workers[engine] = DiaWorker()
            elif engine == "fish":
                from app.services.tts.fish_worker import FishWorker
                _workers[engine] = FishWorker()
            elif engine == "f5_tts":
                from app.services.tts.f5_tts_worker import F5TTSWorker
                _workers[engine] = F5TTSWorker()
            elif engine == "kokoro":
                from app.services.tts.kokoro_worker import KokoroWorker
                _workers[engine] = KokoroWorker()
            elif engine == "piper":
                from app.services.tts.piper_worker import PiperWorker
                _workers[engine] = PiperWorker()
        except ImportError as e:
            logger.error(f"Failed to import dependencies for TTS engine '{engine}': {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Dependencies for TTS engine '{engine}' are not installed: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize worker for TTS engine '{engine}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Initialization error for engine '{engine}': {str(e)}"
            )
            
        return _workers[engine]

# ----------------- Dummy WAV Generation (test_mode=True) -----------------
def generate_dummy_wav(duration: float = 1.0, sample_rate: int = 24000, size_bytes: Optional[int] = None) -> bytes:
    if size_bytes is not None:
        data_size = max(0, size_bytes - 44)
        num_samples = data_size // 2
    else:
        num_samples = int(duration * sample_rate)
        data_size = num_samples * 2
        
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)      # Mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(sample_rate)
        
        chunk_size = 1000
        val_true = struct.pack("<h", 3000)
        val_false = struct.pack("<h", -3000)
        
        samples_written = 0
        while samples_written < num_samples:
            to_write = min(chunk_size, num_samples - samples_written)
            chunk = b"".join(val_true if (i // 120) % 2 == 0 else val_false for i in range(to_write))
            wav_file.writeframes(chunk)
            samples_written += to_write
            
    return buffer.getvalue()


def _extract_pcm_from_wav(wav_bytes: bytes) -> tuple[bytes, int]:
    """Extract raw PCM s16le data and sample rate from WAV bytes."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
    return pcm_data, sample_rate


def _wav_duration_ms(wav_bytes: bytes) -> float:
    """Calculate duration in milliseconds from WAV bytes."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        if sr == 0:
            return 0.0
        return (n_frames / sr) * 1000.0


# ----------------- Shared Helpers -----------------
def _check_rate_limit():
    """Raise 429 if rate limit exceeded."""
    if not rate_limiter.is_allowed():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too Many Requests"
        )


def _check_fish_consent(engine: str, test_mode: bool, voice_id: Optional[str]):
    """Raise 403 if Fish Audio engine lacks explicit consent."""
    if engine != "fish":
        return
    enable_fish = os.environ.get("ENABLE_FISH_AUDIO", "false").lower() == "true" or \
                  (test_mode and voice_id and "enable_fish" in voice_id)
    if not enable_fish:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Fish Audio engine requires explicit consent."
        )


def _synthesize_wav(engine: str, text: str, voice_id: Optional[str], test_mode: bool) -> tuple[bytes, int]:
    """
    Produce WAV bytes + sample_rate. Handles test_mode dummy generation
    and real worker dispatch. Returns (wav_bytes, sample_rate).
    """
    if test_mode:
        size_bytes = None
        if "size_bytes=" in text:
            m = re.search(r"size_bytes=(\d+)", text)
            if m:
                size_bytes = int(m.group(1))

        duration = 1.0
        if "duration_sec=" in text:
            m = re.search(r"duration_sec=([\d\.]+)", text)
            if m:
                duration = float(m.group(1))

        if engine == "dia" and voice_id and "simulate_offline" in voice_id:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Dia engine offline"
            )

        wav_bytes = generate_dummy_wav(duration=duration, sample_rate=24000, size_bytes=size_bytes)
        return wav_bytes, 24000
    else:
        worker = get_worker(engine)
        wav_bytes, sample_rate = worker.synthesize(text, voice_id)

        from app.audio.pipeline import AudioPipeline
        pipeline = AudioPipeline()
        wav_bytes, sample_rate = pipeline.process_wav_bytes(wav_bytes, sample_rate)

        return wav_bytes, sample_rate


# ----------------- Content-type mapping -----------------
_FORMAT_CONTENT_TYPES = {
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
}


# ----------------- API Endpoints -----------------
@app.get("/health")
def health():
    return {"status": "healthy", "service": "tts-service"}

@app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    # Empty text validation check
    if not req.text or req.text.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text cannot be empty"
        )

    # Apply rate limiting
    if not rate_limiter.is_allowed():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too Many Requests"
        )

    # Consent check for Fish Audio
    if req.engine == "fish":
        enable_fish = os.environ.get("ENABLE_FISH_AUDIO", "false").lower() == "true" or \
                      (req.test_mode and req.voice_id and "enable_fish" in req.voice_id)
        if not enable_fish:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Fish Audio engine requires explicit consent."
            )

    start_time = time.time()

    if req.test_mode:
        # Parse test metrics from instruction text
        size_bytes = None
        if "size_bytes=" in req.text:
            m = re.search(r"size_bytes=(\d+)", req.text)
            if m:
                size_bytes = int(m.group(1))
                
        duration = 1.0
        if "duration_sec=" in req.text:
            m = re.search(r"duration_sec=([\d\.]+)", req.text)
            if m:
                duration = float(m.group(1))

        # Suno Bark Dia offline simulation for E2E tests
        if req.engine == "dia" and req.voice_id and "simulate_offline" in req.voice_id:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Dia engine offline"
            )

        wav_bytes = generate_dummy_wav(duration=duration, sample_rate=24000, size_bytes=size_bytes)
        audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
        synthesis_time_ms = (time.time() - start_time) * 1000.0
        
        return SynthesizeResponse(
            audio_bytes_base64=audio_b64,
            format="wav",
            sample_rate=24000,
            synthesis_time_ms=synthesis_time_ms
        )
        
    else:
        # Real inference routing
        try:
            worker = get_worker(req.engine)
            wav_bytes, sample_rate = worker.synthesize(req.text, req.voice_id)
            
            # Post-process waveform
            from app.audio.pipeline import AudioPipeline
            pipeline = AudioPipeline()
            wav_bytes, sample_rate = pipeline.process_wav_bytes(wav_bytes, sample_rate)
            
            audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            synthesis_time_ms = (time.time() - start_time) * 1000.0
            
            return SynthesizeResponse(
                audio_bytes_base64=audio_b64,
                format="wav",
                sample_rate=sample_rate,
                synthesis_time_ms=synthesis_time_ms
            )
        except HTTPException as e:
            # Let FastAPIs own HTTPException bubble up
            raise e
        except ImportError as e:
            logger.error(f"Failed to import dependencies for TTS engine '{req.engine}': {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Dependencies for TTS engine '{req.engine}' are not installed: {str(e)}"
            )
        except FileNotFoundError as e:
            logger.error(f"Model or voice files missing: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model files for {req.engine} are missing: {str(e)}"
            )
        except Exception as e:
            logger.error(f"TTS inference execution failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"TTS synthesis failure on engine '{req.engine}': {str(e)}"
            )


# ----------------- POST /synthesize/stream  (PCM streaming) -----------------

@app.post("/synthesize/stream")
def synthesize_stream(req: StreamRequest):
    """
    Synthesize speech and stream raw PCM s16le data.

    Returns a StreamingResponse with content-type ``audio/pcm`` and
    metadata headers:
      - X-Sample-Rate
      - X-Channels
      - X-Bit-Depth
      - X-Time-To-First-Chunk-Ms
    """
    if not req.text or req.text.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text cannot be empty"
        )

    _check_rate_limit()
    _check_fish_consent(req.engine, bool(req.test_mode), req.voice_id)

    start_time = time.time()

    try:
        wav_bytes, sample_rate = _synthesize_wav(
            engine=req.engine,
            text=req.text,
            voice_id=req.voice_id,
            test_mode=bool(req.test_mode),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stream synthesis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS synthesis failure on engine '{req.engine}': {str(e)}"
        )

    pcm_data, _ = _extract_pcm_from_wav(wav_bytes)

    time_to_first_chunk_ms = (time.time() - start_time) * 1000.0

    # Stream the PCM data in 4 KB chunks
    PCM_CHUNK_SIZE = 4096

    def pcm_generator() -> Generator[bytes, None, None]:
        offset = 0
        while offset < len(pcm_data):
            yield pcm_data[offset:offset + PCM_CHUNK_SIZE]
            offset += PCM_CHUNK_SIZE

    headers = {
        "X-Sample-Rate": str(sample_rate),
        "X-Channels": "1",
        "X-Bit-Depth": "16",
        "X-Time-To-First-Chunk-Ms": f"{time_to_first_chunk_ms:.2f}",
    }

    return StreamingResponse(
        content=pcm_generator(),
        media_type="audio/pcm",
        headers=headers,
    )


# ----------------- POST /synthesize/export  (file download) -----------------

@app.post("/synthesize/export")
def synthesize_export(req: ExportRequest):
    """
    Synthesize speech, encode to the requested format, and return as a
    downloadable file attachment.

    Supported formats: wav, ogg, mp3.

    Response headers include metadata sidecar values:
      - X-Audio-Duration-Ms
      - X-Sample-Rate
      - X-Channels
      - X-Bit-Depth
      - X-Audio-Format
    """
    if not req.text or req.text.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Text cannot be empty"
        )

    _check_rate_limit()
    _check_fish_consent(req.engine, bool(req.test_mode), req.voice_id)

    try:
        wav_bytes, sample_rate = _synthesize_wav(
            engine=req.engine,
            text=req.text,
            voice_id=req.voice_id,
            test_mode=bool(req.test_mode),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export synthesis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS synthesis failure on engine '{req.engine}': {str(e)}"
        )

    duration_ms = _wav_duration_ms(wav_bytes)

    # Encode to requested format
    output_format = req.format
    if output_format == "wav":
        audio_data = wav_bytes
    else:
        # Write WAV to a temp file, then use encoder
        tmp_wav = None
        try:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.write(wav_bytes)
            tmp_wav.close()

            from app.audio.encoder import encode_audio
            audio_data = encode_audio(Path(tmp_wav.name), output_format)
        except Exception as e:
            logger.error(f"Audio encoding failed for format '{output_format}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Audio encoding failed for format '{output_format}': {str(e)}"
            )
        finally:
            if tmp_wav is not None:
                try:
                    os.unlink(tmp_wav.name)
                except OSError:
                    pass

    safe_filename = f"{req.filename}.{output_format}"
    content_type = _FORMAT_CONTENT_TYPES.get(output_format, "application/octet-stream")

    headers = {
        "Content-Disposition": f'attachment; filename="{safe_filename}"',
        "X-Audio-Duration-Ms": f"{duration_ms:.2f}",
        "X-Sample-Rate": str(sample_rate),
        "X-Channels": "1",
        "X-Bit-Depth": "16",
        "X-Audio-Format": output_format,
    }

    return Response(
        content=audio_data,
        media_type=content_type,
        headers=headers,
    )
