import base64
import collections
import hashlib
import io
import os
import pathlib
import re
import tempfile
import threading
import time
import wave
from typing import Optional, Literal, Generator, Any
from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.audio.cache import AudioCacheManager, get_cache_key, is_safe_path
from app.audio.encoder import encode_audio
from app.audio.signer import sign_audio_id
from app.audio.probe import get_audio_duration_ms
from app.core.job_store import job_store
from app.safety.text_sanitizer import sanitize_text

router = APIRouter(tags=["tts"])
cache_manager = AudioCacheManager()

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

# ----------------- Lazy Loaded Providers -----------------
# Dynamically imports and fetches providers
def get_tts_provider(engine: str) -> Any:
    from app.core.orchestrator import get_tts_provider as orchestrator_get_tts
    return orchestrator_get_tts(engine)

# ----------------- Pydantic Models -----------------
class SynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "default"
    engine: Literal["chatterbox", "dia", "fish", "f5_tts", "kokoro", "piper"]
    test_mode: Optional[bool] = None

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
    test_mode: Optional[bool] = None

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
    test_mode: Optional[bool] = None

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
        if ".." in v:
            sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', v)
            return sanitized
        parts = re.split(r'[/\\]', v)
        basename = parts[-1] if parts else ""
        sanitized = re.sub(r'[^a-zA-Z0-9_\-]', '_', basename)
        if not sanitized:
            return "tts_output"
        return sanitized

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "default"
    engine: Optional[Literal["chatterbox", "dia", "fish", "f5_tts", "kokoro", "piper"]] = "kokoro"
    format: Optional[str] = "wav"
    test_mode: Optional[bool] = None

    @field_validator('format')
    @classmethod
    def validate_format(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("wav", "ogg", "mp3"):
            raise ValueError(f"Unsupported format: {v}")
        return v

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

class TTSResponse(BaseModel):
    job_id: str
    state: str
    text: str
    audio: AudioMetadata
    metrics: Metrics

# ----------------- Shared Helpers -----------------
def _check_real_mode_violation(text: str, voice_id: Optional[str], test_mode: Optional[bool]):
    if settings.mode == "real":
        if test_mode is not None:
            raise HTTPException(status_code=400, detail="test_mode parameter is forbidden in production/real mode.")
        simulation_keywords = [
            "simulate_offline", "simulate_client_disconnect", "duration_sec=", "size_bytes="
        ]
        if any(kw in text for kw in simulation_keywords):
            raise HTTPException(status_code=400, detail="Simulation keywords are forbidden in production/real mode.")
        if voice_id and any(kw in voice_id for kw in ("simulate_offline", "enable_fish")):
            raise HTTPException(status_code=400, detail="Simulation keywords are forbidden in production/real mode.")

def _check_fish_consent(engine: str, voice_id: Optional[str]):
    if engine != "fish":
        return
    enable_fish = os.environ.get("ENABLE_FISH_AUDIO", "false").lower() == "true" or \
                  (settings.mode == "test" and voice_id and "enable_fish" in voice_id)
    if not enable_fish:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Fish Audio engine requires explicit consent."
        )

def _synthesize_wav_in_process(engine: str, text: str, voice_id: Optional[str], test_mode: Optional[bool] = None) -> tuple[bytes, int]:
    """Helper calling in-process TTS providers."""
    if settings.mode == "test" and test_mode is not False:
        tts = get_tts_provider(engine)
    else:
        from app.services.tts_service import get_worker
        tts = get_worker(engine)
    return tts.synthesize(text, voice_id or "default")

# ----------------- Endpoints -----------------

@router.post("/synthesize", response_model=SynthesizeResponse)
def post_synthesize(req: SynthesizeRequest):
    if not rate_limiter.is_allowed():
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too Many Requests")
    _check_real_mode_violation(req.text, req.voice_id, req.test_mode)
    _check_fish_consent(req.engine, req.voice_id)

    start_time = time.time()
    
    try:
        wav_bytes, sample_rate = _synthesize_wav_in_process(req.engine, req.text, req.voice_id, req.test_mode)
        
        # Audio post processing pipeline
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
        raise e
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"TTS engine '{req.engine}' is not installed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS synthesis failure on engine '{req.engine}': {str(e)}"
        )

@router.post("/synthesize/stream")
def post_synthesize_stream(req: StreamRequest):
    if not rate_limiter.is_allowed():
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too Many Requests")
    _check_real_mode_violation(req.text, req.voice_id, req.test_mode)
    _check_fish_consent(req.engine, req.voice_id)

    start_time = time.time()
    try:
        wav_bytes, sample_rate = _synthesize_wav_in_process(req.engine, req.text, req.voice_id, req.test_mode)
        from app.audio.pipeline import AudioPipeline
        pipeline = AudioPipeline()
        wav_bytes, sample_rate = pipeline.process_wav_bytes(wav_bytes, sample_rate)
    except HTTPException as e:
        raise e
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"TTS engine '{req.engine}' is not installed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS stream failure: {str(e)}"
        )

    # Extract raw PCM s16le bytes from WAV
    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            pcm_data = wf.readframes(wf.getnframes())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read WAV PCM bytes: {e}")

    time_to_first_chunk_ms = (time.time() - start_time) * 1000.0
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

@router.post("/synthesize/export")
def post_synthesize_export(req: ExportRequest):
    if not rate_limiter.is_allowed():
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too Many Requests")
    _check_real_mode_violation(req.text, req.voice_id, req.test_mode)
    _check_fish_consent(req.engine, req.voice_id)

    try:
        wav_bytes, sample_rate = _synthesize_wav_in_process(req.engine, req.text, req.voice_id, req.test_mode)
        from app.audio.pipeline import AudioPipeline
        pipeline = AudioPipeline()
        wav_bytes, sample_rate = pipeline.process_wav_bytes(wav_bytes, sample_rate)
    except HTTPException as e:
        raise e
    except ImportError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"TTS engine '{req.engine}' is not installed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS export failure: {str(e)}"
        )

    duration_ms = get_audio_duration_ms(wav_bytes, "wav")

    # Encode to output format
    output_format = req.format
    if output_format == "wav":
        audio_data = wav_bytes
    else:
        tmp_wav = None
        try:
            tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_wav.write(wav_bytes)
            tmp_wav.close()

            audio_data = encode_audio(pathlib.Path(tmp_wav.name), output_format)
        except Exception as e:
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
    content_type = {
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg"
    }.get(output_format, "application/octet-stream")

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

# ----------------- v1/tts API (Orchestrated) -----------------
@router.post("/v1/tts", response_model=TTSResponse)
async def post_v1_tts(
    req: TTSRequest,
    request: Request,
    cache_control: Optional[str] = Header(None)
):
    if not rate_limiter.is_allowed():
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too Many Requests")
    _check_real_mode_violation(req.text, req.voice_id, req.test_mode)
    engine = req.engine or "kokoro"
    _check_fish_consent(engine, req.voice_id)

    t_start = time.time()
    sanitized_text_val = sanitize_text(req.text)
    
    job_id = job_store.create_job()
    
    # Check cache
    use_cache = True
    if cache_control and "no-cache" in cache_control.lower():
        use_cache = False
        
    cache_key = get_cache_key(sanitized_text_val, req.voice_id, req.format, engine=engine)
    cached_data = None
    if use_cache:
        cached_data = cache_manager.get(sanitized_text_val, req.voice_id, req.format, engine=engine)
        
    # Detect corruption
    is_corrupt = False
    if cached_data is not None:
        if len(cached_data) == 0:
            is_corrupt = True
        elif req.format == "wav" and (len(cached_data) < 44 or not cached_data.startswith(b"RIFF")):
            is_corrupt = True
        elif cached_data.count(b"\x00") == len(cached_data):
            is_corrupt = True
            
    if is_corrupt:
        cached_data = None
        
    cache_hit = cached_data is not None
    tts_ms = 0.0
    encode_ms = 0.0
    duration_ms = 0
    encoded_data = None
    sample_rate = 24000
    
    if cache_hit:
        encoded_data = cached_data
        try:
            metadata = cache_manager.get_metadata(sanitized_text_val, req.voice_id, req.format, engine=engine)
            if metadata and "duration_ms" in metadata:
                duration_ms = metadata["duration_ms"]
            else:
                duration_ms = get_audio_duration_ms(encoded_data, req.format)
        except Exception:
            duration_ms = get_audio_duration_ms(encoded_data, req.format)
    else:
        # Synthesis
        t_tts_start = time.time()
        try:
            wav_bytes, sample_rate = _synthesize_wav_in_process(engine, sanitized_text_val, req.voice_id, req.test_mode)
        except Exception as e:
            # Fallback to piper for Dia failures
            if engine == "dia":
                logger.warning(f"Dia synthesis failed, falling back to Piper: {e}")
                engine = "piper"
                try:
                    wav_bytes, sample_rate = _synthesize_wav_in_process("piper", sanitized_text_val, req.voice_id, req.test_mode)
                except Exception as inner_e:
                    job_store.update_job(job_id, {"state": "failed", "error": str(inner_e)})
                    if isinstance(inner_e, ImportError):
                        raise HTTPException(status_code=503, detail=f"TTS engine 'piper' is not installed: {str(inner_e)}")
                    raise HTTPException(status_code=503, detail="Fallback TTS service failed")
            else:
                job_store.update_job(job_id, {"state": "failed", "error": str(e)})
                if isinstance(e, ImportError):
                    raise HTTPException(status_code=503, detail=f"TTS engine '{engine}' is not installed: {str(e)}")
                raise HTTPException(status_code=503, detail=f"TTS service synthesis failure: {str(e)}")
                
        tts_ms = (time.time() - t_tts_start) * 1000.0
        duration_ms = get_audio_duration_ms(wav_bytes, "wav")
        
        # Encode audio
        t_enc_start = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(wav_bytes)
            temp_wav_path = pathlib.Path(temp_wav.name)
            
        try:
            encoded_data = encode_audio(temp_wav_path, req.format)
        except Exception as e:
            job_store.update_job(job_id, {"state": "failed", "error": "encoding_failed"})
            raise HTTPException(status_code=500, detail=f"Audio encoding failed: {str(e)}")
        finally:
            if temp_wav_path.exists():
                try:
                    os.unlink(temp_wav_path)
                except OSError:
                    pass
                    
        encode_ms = (time.time() - t_enc_start) * 1000.0
        
        # Save to cache
        try:
            cache_manager.put(
                sanitized_text_val,
                req.voice_id,
                req.format,
                encoded_data,
                duration_ms=duration_ms,
                engine=engine
            )
        except ValueError as e:
            job_store.update_job(job_id, {"state": "failed", "error": str(e)})
            raise HTTPException(status_code=400, detail=str(e))

    # Generate signed URL/id
    audio_id = f"{cache_key}_{req.format}"
    signed_token = sign_audio_id(audio_id)
    
    t_total = (time.time() - t_start) * 1000.0
    sha256_hash = hashlib.sha256(encoded_data).hexdigest()
    
    response_obj = TTSResponse(
        job_id=job_id,
        state="ready",
        text=sanitized_text_val,
        audio=AudioMetadata(
            audio_id=signed_token,
            sha256=sha256_hash,
            bytes=len(encoded_data),
            duration_ms=duration_ms,
            format=req.format,
            sample_rate=sample_rate
        ),
        metrics=Metrics(
            queue_ms=0.0,
            llm_ms=0.0,
            tts_ms=tts_ms,
            encode_ms=encode_ms,
            total_ms=t_total,
            cache_hit=cache_hit
        )
    )
    job_store.update_job(job_id, response_obj.model_dump())
    return response_obj
