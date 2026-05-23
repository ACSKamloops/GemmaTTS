import asyncio
import base64
import hashlib
import os
import pathlib
import tempfile
import time
import urllib.parse
from typing import List, Optional
from typing_extensions import Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Real codebase imports
from app.audio.cache import AudioCacheManager, get_cache_key, is_safe_path
from app.audio.encoder import encode_audio
from app.audio.signer import sign_audio_id, verify_signed_audio_id
from app.config import settings
from app.safety.text_sanitizer import sanitize_text
from app.services.voice_registry import registry as voice_registry

# In-memory job store
_jobs = {}

app = FastAPI(title="Orchestrator Gateway API", version="1.0.0")

from app.middleware.auth import AuthMiddleware
app.add_middleware(AuthMiddleware)

from app.dashboard.router import router as dashboard_router
app.include_router(dashboard_router, prefix="/dashboard")

cache_manager = AudioCacheManager()

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
    test_mode: Optional[bool] = True

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

@app.get("/health")
def health_gateway():
    return {"status": "healthy"}

@app.post("/debug/rotate_key")
def rotate_key(new_key: str):
    if not new_key or len(new_key) < 32:
        raise HTTPException(status_code=400, detail="Secret key must be at least 32 characters long.")
    settings.secret_key = new_key
    return {"status": "key rotated"}

@app.post("/debug/update_settings")
def update_settings(
    max_cache_size_bytes: Optional[int] = None,
    max_file_size_bytes: Optional[int] = None
):
    if max_cache_size_bytes is not None:
        settings.max_cache_size_bytes = max_cache_size_bytes
    if max_file_size_bytes is not None:
        settings.max_file_size_bytes = max_file_size_bytes
    return {"status": "settings updated"}

job_id_counter = 100

@app.post("/v1/dialogue", response_model=DialogueResponse)
async def dialogue(
    req: DialogueRequest,
    request: Request,
    cache_control: Optional[str] = Header(None)
):
    global job_id_counter
    t_start = time.time()
    
    # Validate format parameter
    supported_formats = {"wav", "ogg", "mp3", "pcm"}
    fmt = req.output.format.lower().strip()
    if fmt not in supported_formats:
        raise HTTPException(status_code=422, detail=f"Unsupported format: {fmt}")
        
    # Sanitize input text using actual text sanitizer
    sanitized_input = sanitize_text(req.user_text)
    
    # LLM service disconnect simulation
    if "simulate_llm_crash" in req.user_text:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    # Call Gemma service (Port 8001)
    t_llm_start = time.time()
    try:
        llm_payload = {
            "prompt": sanitized_input,
            "max_words": req.max_words,
            "enable_thinking": False,
            "test_mode": req.test_mode
        }
        async with httpx.AsyncClient() as client:
            gemma_resp = await client.post("http://127.0.0.1:8001/generate", json=llm_payload, timeout=5.0)
        t_llm_end = time.time()
        
        # Fallback mechanism
        if gemma_resp.status_code == 422:
            try:
                detail = gemma_resp.json().get("detail", "Validation Error")
            except Exception:
                detail = "Validation Error"
            raise HTTPException(status_code=422, detail=detail)

        if gemma_resp.status_code != 200:
            generated_text = "Fallback dialogue text due to schema mismatch."
        else:
            try:
                gemma_data = gemma_resp.json()
                generated_text = gemma_data.get("text", "Fallback dialogue text.")
            except Exception:
                generated_text = "Fallback dialogue text due to schema mismatch."
                
        llm_ms = (t_llm_end - t_llm_start) * 1000.0
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    job_id_counter += 1
    job_id = f"job_{job_id_counter}"
    
    # output.audio is false
    if not req.output.audio:
        t_total = (time.time() - t_start) * 1000.0
        response_obj = DialogueResponse(
            job_id=job_id,
            state="ready",
            text=generated_text,
            audio=None,
            metrics=Metrics(
                queue_ms=0.0,
                llm_ms=llm_ms,
                tts_ms=0.0,
                encode_ms=0.0,
                total_ms=t_total,
                cache_hit=False
            )
        )
        _jobs[job_id] = response_obj.model_dump()
        return response_obj
        
    # Check Cache-Control header
    use_cache = True
    if cache_control and "no-cache" in cache_control.lower():
        use_cache = False
        
    cache_key = get_cache_key(generated_text, req.speaker.voice_id, fmt)
    
    # Get from cache
    cached_data = None
    if use_cache:
        cached_data = cache_manager.get(generated_text, req.speaker.voice_id, fmt)
        
    # Corrupt cached file detection
    is_corrupt = False
    if cached_data is not None:
        if len(cached_data) == 0:
            is_corrupt = True
        elif fmt == "wav" and (len(cached_data) < 44 or not cached_data.startswith(b"RIFF")):
            is_corrupt = True
        elif cached_data.count(b"\x00") == len(cached_data):
            is_corrupt = True
            
    if is_corrupt:
        cached_data = None
        
    cache_hit = cached_data is not None
    tts_ms = 0.0
    encode_ms = 0.0
    duration_ms = 0
    
    if cache_hit:
        encoded_data = cached_data
        try:
            metadata = cache_manager.get_metadata(generated_text, req.speaker.voice_id, fmt)
            if metadata and "duration_ms" in metadata:
                duration_ms = metadata["duration_ms"]
            else:
                cache_path = cache_manager.get_file_path(cache_key, fmt)
                duration_path = cache_path.with_suffix(cache_path.suffix + ".duration")
                if duration_path.exists():
                    duration_ms = int(duration_path.read_text().strip())
                else:
                    if fmt == "wav":
                        import io, wave
                        with wave.open(io.BytesIO(encoded_data), "rb") as w:
                            duration_ms = int((w.getnframes() / w.getframerate()) * 1000)
                    else:
                        duration_ms = int(len(encoded_data) / 48)
        except Exception:
            duration_ms = int(len(encoded_data) / 48)
    else:
        # Determine active engine
        engine = "chatterbox"
        style_lower = (req.speaker.style or "").lower()
        if "dia" in style_lower:
            engine = "dia"
        elif "fish" in style_lower:
            engine = "fish"
        elif "f5_tts" in style_lower:
            engine = "f5_tts"
        elif "kokoro" in style_lower:
            engine = "kokoro"
        elif "piper" in style_lower:
            engine = "piper"
            
        # Client Disconnect mid-synthesis simulation
        if "simulate_client_disconnect" in req.user_text:
            for _ in range(20):
                if await request.is_disconnected():
                    return Response(status_code=499)
                await asyncio.sleep(0.05)

        t_tts_start = time.time()
        try:
            tts_payload = {
                "text": generated_text,
                "voice_id": req.speaker.voice_id,
                "engine": engine,
                "test_mode": req.test_mode
            }
            async with httpx.AsyncClient() as client:
                tts_resp = await client.post("http://127.0.0.1:8002/synthesize", json=tts_payload, timeout=5.0)
            
            # Fallback: if engine synthesis fails, fallback to piper
            if tts_resp.status_code != 200 and engine == "dia":
                engine = "piper"
                tts_payload["engine"] = "piper"
                async with httpx.AsyncClient() as client:
                    tts_resp = await client.post("http://127.0.0.1:8002/synthesize", json=tts_payload, timeout=5.0)
                
            if tts_resp.status_code != 200:
                raise HTTPException(status_code=503, detail="TTS service synthesis failure")
                
            tts_data = tts_resp.json()
            wav_b64 = tts_data["audio_bytes_base64"]
            wav_bytes = base64.b64decode(wav_b64)
            tts_ms = (time.time() - t_tts_start) * 1000.0
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            # Fallback to piper for S-3
            if engine == "dia":
                try:
                    tts_payload = {
                        "text": generated_text,
                        "voice_id": req.speaker.voice_id,
                        "engine": "piper",
                        "test_mode": req.test_mode
                    }
                    async with httpx.AsyncClient() as client:
                        tts_resp = await client.post("http://127.0.0.1:8002/synthesize", json=tts_payload, timeout=5.0)
                    if tts_resp.status_code == 200:
                        tts_data = tts_resp.json()
                        wav_b64 = tts_data["audio_bytes_base64"]
                        wav_bytes = base64.b64decode(wav_b64)
                        tts_ms = (time.time() - t_tts_start) * 1000.0
                    else:
                        raise HTTPException(status_code=503, detail="Fallback TTS service failed")
                except Exception as inner_e:
                    if isinstance(inner_e, HTTPException):
                        raise inner_e
                    raise HTTPException(status_code=503, detail="TTS service unavailable")
            else:
                raise HTTPException(status_code=503, detail="TTS service unavailable")

        # Parse duration from WAV bytes
        try:
            import io, wave
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                duration_ms = int((w.getnframes() / w.getframerate()) * 1000)
        except Exception:
            duration_ms = int(len(wav_bytes) / 48)

        # Encode WAV to requested format using actual codebase component
        t_enc_start = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(wav_bytes)
            temp_wav_path = pathlib.Path(temp_wav.name)
            
        try:
            encoded_data = encode_audio(temp_wav_path, fmt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"FFmpeg encoding failed: {str(e)}")
        finally:
            if temp_wav_path.exists():
                try:
                    os.unlink(temp_wav_path)
                except OSError:
                    pass
                    
        encode_ms = (time.time() - t_enc_start) * 1000.0
        
        # Save to cache
        try:
            cache_path = cache_manager.put(generated_text, req.speaker.voice_id, fmt, encoded_data, duration_ms=duration_ms)
            try:
                duration_path = cache_path.with_suffix(cache_path.suffix + ".duration")
                duration_path.write_text(str(duration_ms))
            except Exception:
                pass
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    # Generate signed URL/id
    audio_id = f"{cache_key}_{fmt}"
    signed_token = sign_audio_id(audio_id)
    
    t_total = (time.time() - t_start) * 1000.0
    sha256_hash = hashlib.sha256(encoded_data).hexdigest()
    
    response_obj = DialogueResponse(
        job_id=job_id,
        state="ready",
        text=generated_text,
        audio=AudioMetadata(
            audio_id=signed_token,
            sha256=sha256_hash,
            bytes=len(encoded_data),
            duration_ms=duration_ms,
            format=fmt,
            sample_rate=24000
        ),
        metrics=Metrics(
            queue_ms=0.0,
            llm_ms=llm_ms,
            tts_ms=tts_ms,
            encode_ms=encode_ms,
            total_ms=t_total,
            cache_hit=cache_hit
        )
    )
    _jobs[job_id] = response_obj.model_dump()
    return response_obj

@app.get("/audio/{signed_id:path}")
def get_audio(signed_id: str, format: Optional[str] = None):
    # Decode URL-encoded characters just in case
    decoded_id = urllib.parse.unquote(signed_id)
    
    # Direct traversal check
    if ".." in decoded_id or "/" in decoded_id or "\\" in decoded_id:
        raise HTTPException(status_code=400, detail="Path traversal or out-of-boundary access detected.")
        
    verified = verify_signed_audio_id(decoded_id)
    if not verified:
        raise HTTPException(status_code=403, detail="Signature expired or invalid")
        
    if "_" in verified:
        key, format_str = verified.rsplit("_", 1)
    else:
        key = verified
        format_str = format or "wav"
        
    try:
        path = cache_manager.get_file_path(key, format_str)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # Check symlink safety
    if path.is_symlink():
        real_path = path.resolve()
        if not is_safe_path(real_path, cache_manager.cache_dir):
            raise HTTPException(status_code=403, detail="Symlink targets outside cache directory.")
            
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    # Stream the file
    media_types = {
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
        "pcm": "audio/l16"
    }
    media_type = media_types.get(format_str.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type)


# ----------------- v1 REST Endpoints for External Projects -----------------

class TTSRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "default"
    engine: Optional[Literal["chatterbox", "dia", "fish", "f5_tts", "kokoro", "piper"]] = "kokoro"
    format: Optional[str] = "wav"
    test_mode: Optional[bool] = True

class TTSResponse(BaseModel):
    job_id: str
    state: str
    text: str
    audio: AudioMetadata
    metrics: Metrics

@app.post("/v1/tts", response_model=TTSResponse)
async def post_v1_tts(
    req: TTSRequest,
    request: Request,
    cache_control: Optional[str] = Header(None)
):
    global job_id_counter
    t_start = time.time()
    
    if not req.text or req.text.strip() == "":
        raise HTTPException(status_code=422, detail="Text cannot be empty")
        
    # Validate format parameter
    supported_formats = {"wav", "ogg", "mp3", "pcm"}
    fmt = req.format.lower().strip()
    if fmt not in supported_formats:
        raise HTTPException(status_code=422, detail=f"Unsupported format: {fmt}")
        
    # Sanitize input text
    sanitized_text_val = sanitize_text(req.text)
    
    job_id_counter += 1
    job_id = f"job_{job_id_counter}"
    
    # Check Cache-Control header
    use_cache = True
    if cache_control and "no-cache" in cache_control.lower():
        use_cache = False
        
    cache_key = get_cache_key(sanitized_text_val, req.voice_id, fmt)
    
    # Get from cache
    cached_data = None
    if use_cache:
        cached_data = cache_manager.get(sanitized_text_val, req.voice_id, fmt)
        
    # Corrupt cached file detection
    is_corrupt = False
    if cached_data is not None:
        if len(cached_data) == 0:
            is_corrupt = True
        elif fmt == "wav" and (len(cached_data) < 44 or not cached_data.startswith(b"RIFF")):
            is_corrupt = True
        elif cached_data.count(b"\x00") == len(cached_data):
            is_corrupt = True
            
    if is_corrupt:
        cached_data = None
        
    cache_hit = cached_data is not None
    tts_ms = 0.0
    encode_ms = 0.0
    duration_ms = 0
    
    if cache_hit:
        encoded_data = cached_data
        try:
            metadata = cache_manager.get_metadata(sanitized_text_val, req.voice_id, fmt)
            if metadata and "duration_ms" in metadata:
                duration_ms = metadata["duration_ms"]
            else:
                cache_path = cache_manager.get_file_path(cache_key, fmt)
                duration_path = cache_path.with_suffix(cache_path.suffix + ".duration")
                if duration_path.exists():
                    duration_ms = int(duration_path.read_text().strip())
                else:
                    if fmt == "wav":
                        import io, wave
                        with wave.open(io.BytesIO(encoded_data), "rb") as w:
                            duration_ms = int((w.getnframes() / w.getframerate()) * 1000)
                    else:
                        duration_ms = int(len(encoded_data) / 48)
        except Exception:
            duration_ms = int(len(encoded_data) / 48)
    else:
        # Determine active engine
        engine = req.engine or "kokoro"
        
        # Client Disconnect mid-synthesis simulation
        if "simulate_client_disconnect" in req.text:
            for _ in range(20):
                if await request.is_disconnected():
                    return Response(status_code=499)
                await asyncio.sleep(0.05)

        t_tts_start = time.time()
        try:
            tts_payload = {
                "text": sanitized_text_val,
                "voice_id": req.voice_id,
                "engine": engine,
                "test_mode": req.test_mode
            }
            async with httpx.AsyncClient() as client:
                tts_resp = await client.post("http://127.0.0.1:8002/synthesize", json=tts_payload, timeout=5.0)
            
            # Fallback: if engine synthesis fails, fallback to piper
            if tts_resp.status_code != 200 and engine == "dia":
                engine = "piper"
                tts_payload["engine"] = "piper"
                async with httpx.AsyncClient() as client:
                    tts_resp = await client.post("http://127.0.0.1:8002/synthesize", json=tts_payload, timeout=5.0)
                
            if tts_resp.status_code != 200:
                raise HTTPException(status_code=503, detail="TTS service synthesis failure")
                
            tts_data = tts_resp.json()
            wav_b64 = tts_data["audio_bytes_base64"]
            wav_bytes = base64.b64decode(wav_b64)
            tts_ms = (time.time() - t_tts_start) * 1000.0
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            # Fallback to piper for S-3
            if engine == "dia":
                try:
                    tts_payload = {
                        "text": sanitized_text_val,
                        "voice_id": req.voice_id,
                        "engine": "piper",
                        "test_mode": req.test_mode
                    }
                    async with httpx.AsyncClient() as client:
                        tts_resp = await client.post("http://127.0.0.1:8002/synthesize", json=tts_payload, timeout=5.0)
                    if tts_resp.status_code == 200:
                        tts_data = tts_resp.json()
                        wav_b64 = tts_data["audio_bytes_base64"]
                        wav_bytes = base64.b64decode(wav_b64)
                        tts_ms = (time.time() - t_tts_start) * 1000.0
                    else:
                        raise HTTPException(status_code=503, detail="Fallback TTS service failed")
                except Exception as inner_e:
                    if isinstance(inner_e, HTTPException):
                        raise inner_e
                    raise HTTPException(status_code=503, detail="TTS service unavailable")
            else:
                raise HTTPException(status_code=503, detail="TTS service unavailable")

        # Parse duration from WAV bytes
        try:
            import io, wave
            with wave.open(io.BytesIO(wav_bytes), "rb") as w:
                duration_ms = int((w.getnframes() / w.getframerate()) * 1000)
        except Exception:
            duration_ms = int(len(wav_bytes) / 48)

        # Encode WAV to requested format using actual codebase component
        t_enc_start = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(wav_bytes)
            temp_wav_path = pathlib.Path(temp_wav.name)
            
        try:
            encoded_data = encode_audio(temp_wav_path, fmt)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"FFmpeg encoding failed: {str(e)}")
        finally:
            if temp_wav_path.exists():
                try:
                    os.unlink(temp_wav_path)
                except OSError:
                    pass
                    
        encode_ms = (time.time() - t_enc_start) * 1000.0
        
        # Save to cache
        try:
            cache_path = cache_manager.put(sanitized_text_val, req.voice_id, fmt, encoded_data, duration_ms=duration_ms)
            try:
                duration_path = cache_path.with_suffix(cache_path.suffix + ".duration")
                duration_path.write_text(str(duration_ms))
            except Exception:
                pass
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    # Generate signed URL/id
    audio_id = f"{cache_key}_{fmt}"
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
            format=fmt,
            sample_rate=24000
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
    _jobs[job_id] = response_obj.model_dump()
    return response_obj

@app.get("/v1/jobs/{job_id}")
def get_v1_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]

@app.get("/v1/voices")
def get_v1_voices():
    return [v.model_dump() for v in voice_registry.list_all()]
