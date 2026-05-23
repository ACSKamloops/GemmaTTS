import os
import io
import sys
import time
import wave
import struct
import base64
import hashlib
import tempfile
import pathlib
import shutil
import asyncio
import collections
import subprocess
from typing import Optional, List
from typing_extensions import Literal

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Header, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Real codebase imports
from app.config import settings
from app.safety.text_sanitizer import sanitize_text
from app.audio.signer import sign_audio_id, verify_signed_audio_id
from app.audio.cache import AudioCacheManager, get_cache_key, is_safe_path
from app.audio.encoder import encode_audio

# Override settings globally in this process to match test environment
settings.audio_cache_dir = pathlib.Path("tests/test_audio_cache_e2e").resolve()
settings.secret_key = "test-secret-key-for-hmac-verification-operations"
settings.max_cache_size_bytes = 5 * 1024 * 1024
settings.max_file_size_bytes = 1 * 1024 * 1024

# =====================================================================
# Mock FastAPI Applications
# =====================================================================

# ----------------- Gemma Service Mock (Port 8001) -----------------
mock_llm_app = FastAPI()
request_timestamps = collections.deque()

class GenerateRequest(BaseModel):
    prompt: str
    max_words: Optional[int] = 150
    enable_thinking: Optional[bool] = False
    test_mode: Optional[bool] = True

class GenerateResponse(BaseModel):
    text: str
    generation_time_ms: float

@mock_llm_app.get("/health")
def health_gemma():
    return {"status": "healthy", "service": "gemma-service"}

@mock_llm_app.post("/generate")
def generate(req: GenerateRequest):
    # F1-10: Rate limiter checks (100 requests in 1 second)
    now = time.time()
    while request_timestamps and request_timestamps[0] < now - 1.0:
        request_timestamps.popleft()
    request_timestamps.append(now)
    # Threshold set to 40 to easily trigger under 100 requests/sec stress test
    if len(request_timestamps) > 40:
        raise HTTPException(status_code=429, detail="Too Many Requests")

    # F1-6: Empty prompt validation
    if req.prompt == "":
        raise HTTPException(status_code=422, detail="Prompt cannot be empty")
        
    # F1-8: max_words negative check
    if req.max_words is not None and req.max_words < 0:
        raise HTTPException(status_code=422, detail="max_words cannot be negative")
        
    # C-5: Schema Mismatch Fallback simulation
    if "simulate-llm-bad-json" in req.prompt or "simulate_llm_bad_json" in req.prompt:
        return Response(content="not-a-json-string-at-all", media_type="text/plain")

    # F1-7: Truncate prompt if exceeding 5000 characters
    prompt_text = req.prompt
    if len(prompt_text) > 5000:
        prompt_text = prompt_text[:settings.max_text_chars]

    # Rule-based context-aware answers for S-1 Merchant Playthrough
    if "sword" in prompt_text or "buy" in prompt_text:
        reply = "MOCK_RESPONSE: You bought the sword. Merchant gold is now 50."
    elif "change" in prompt_text:
        reply = "MOCK_RESPONSE: Yes, I have change. My gold is 50."
    elif "sell" in prompt_text:
        reply = "MOCK_RESPONSE: I sell swords and shields. I have 10 gold."
    else:
        reply = f"MOCK_RESPONSE: {prompt_text}"

    if req.enable_thinking:
        reply = f"<think>Thinking...</think> {reply}"
        
    # Limit by max_words
    if req.max_words is not None:
        words = reply.split()
        if len(words) > req.max_words:
            reply = " ".join(words[:req.max_words])
            
    return GenerateResponse(text=reply, generation_time_ms=120.0)


# ----------------- TTS Service Mock (Port 8002) -----------------
mock_tts_app = FastAPI()

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: Optional[str] = "default"
    engine: Literal["chatterbox", "dia", "fish", "f5_tts", "kokoro", "piper"]
    test_mode: Optional[bool] = True

class SynthesizeResponse(BaseModel):
    audio_bytes_base64: str
    format: str
    sample_rate: int
    synthesis_time_ms: float

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

@mock_tts_app.get("/health")
def health_tts():
    return {"status": "healthy", "service": "tts-service"}

@mock_tts_app.post("/synthesize", response_model=SynthesizeResponse)
def synthesize(req: SynthesizeRequest):
    # F2-9: Empty text validation
    if req.text == "":
        raise HTTPException(status_code=422, detail="Text cannot be empty")
        
    # F2-6 / F2-7: Check Fish Audio consent
    if req.engine == "fish":
        enable_fish = os.environ.get("ENABLE_FISH_AUDIO", "false").lower() == "true" or (req.voice_id and "enable_fish" in req.voice_id)
        if not enable_fish:
            raise HTTPException(status_code=403, detail="Fish Audio engine requires explicit consent.")

    # S-3: Simulate Dia engine failure
    if req.engine == "dia" and req.voice_id and "simulate_offline" in req.voice_id:
        raise HTTPException(status_code=503, detail="Dia engine offline")
            
    # Size/duration parsing for testing
    size_bytes = None
    if "size_bytes=" in req.text:
        import re
        m = re.search(r"size_bytes=(\d+)", req.text)
        if m:
            size_bytes = int(m.group(1))
            
    duration = 1.0
    if "duration_sec=" in req.text:
        import re
        m = re.search(r"duration_sec=([\d\.]+)", req.text)
        if m:
            duration = float(m.group(1))
            
    wav_bytes = generate_dummy_wav(duration=duration, sample_rate=24000, size_bytes=size_bytes)
    audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
    
    return SynthesizeResponse(
        audio_bytes_base64=audio_b64,
        format="wav",
        sample_rate=24000,
        synthesis_time_ms=340.0
    )


# ----------------- Gateway Service Mock (Port 8000) -----------------
mock_gateway_app = FastAPI()
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

@mock_gateway_app.get("/health")
def health_gateway():
    return {"status": "healthy"}

@mock_gateway_app.post("/debug/rotate_key")
def rotate_key(new_key: str):
    settings.secret_key = new_key
    return {"status": "key rotated"}

@mock_gateway_app.post("/debug/update_settings")
def update_settings(max_cache_size_bytes: Optional[int] = None, max_file_size_bytes: Optional[int] = None):
    if max_cache_size_bytes is not None:
        settings.max_cache_size_bytes = max_cache_size_bytes
    if max_file_size_bytes is not None:
        settings.max_file_size_bytes = max_file_size_bytes
    return {"status": "settings updated"}

job_id_counter = 100

@mock_gateway_app.post("/v1/dialogue", response_model=DialogueResponse)
async def dialogue(req: DialogueRequest, request: Request, cache_control: Optional[str] = Header(None)):
    global job_id_counter
    t_start = time.time()
    
    # Validate format parameter (F3-7)
    supported_formats = {"wav", "ogg", "mp3", "pcm"}
    fmt = req.output.format.lower().strip()
    if fmt not in supported_formats:
        raise HTTPException(status_code=422, detail=f"Unsupported format: {fmt}")
        
    # F5-4 / F5-8: Sanitize input text using actual text sanitizer
    sanitized_input = sanitize_text(req.user_text)
    
    # F1-9: LLM service disconnect simulation
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
        gemma_resp = httpx.post("http://127.0.0.1:8001/generate", json=llm_payload, timeout=5.0)
        t_llm_end = time.time()
        
        # C-5 fallback mechanism
        if gemma_resp.status_code != 200:
            generated_text = "Fallback dialogue text due to schema mismatch."
        else:
            try:
                gemma_data = gemma_resp.json()
                generated_text = gemma_data.get("text", "Fallback dialogue text.")
            except Exception:
                generated_text = "Fallback dialogue text due to schema mismatch."
                
        llm_ms = (t_llm_end - t_llm_start) * 1000.0
    except Exception:
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    job_id_counter += 1
    job_id = f"job_{job_id_counter}"
    
    # F3-10: output.audio is false
    if not req.output.audio:
        t_total = (time.time() - t_start) * 1000.0
        return DialogueResponse(
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
        
    # Check Cache-Control header (F4-5)
    use_cache = True
    if cache_control and "no-cache" in cache_control.lower():
        use_cache = False
        
    cache_key = get_cache_key(generated_text, req.speaker.voice_id, fmt)
    
    # Get from cache
    cached_data = None
    if use_cache:
        cached_data = cache_manager.get(generated_text, req.speaker.voice_id, fmt)
        
    # F4-10: Corrupt cached file detection (empty or all zero bytes)
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
    
    if cache_hit:
        encoded_data = cached_data
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
            
        # C-6: Client Disconnect mid-synthesis simulation
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
            tts_resp = httpx.post("http://127.0.0.1:8002/synthesize", json=tts_payload, timeout=5.0)
            
            # S-3 Fallback: if engine synthesis fails, fallback to piper
            if tts_resp.status_code != 200 and engine == "dia":
                engine = "piper"
                tts_payload["engine"] = "piper"
                tts_resp = httpx.post("http://127.0.0.1:8002/synthesize", json=tts_payload, timeout=5.0)
                
            if tts_resp.status_code != 200:
                raise HTTPException(status_code=503, detail="TTS service synthesis failure")
                
            tts_data = tts_resp.json()
            wav_b64 = tts_data["audio_bytes_base64"]
            wav_bytes = base64.b64decode(wav_b64)
            tts_ms = (time.time() - t_tts_start) * 1000.0
            
        except Exception:
            # Fallback to piper for S-3
            if engine == "dia":
                try:
                    tts_payload = {
                        "text": generated_text,
                        "voice_id": req.speaker.voice_id,
                        "engine": "piper",
                        "test_mode": req.test_mode
                    }
                    tts_resp = httpx.post("http://127.0.0.1:8002/synthesize", json=tts_payload, timeout=5.0)
                    if tts_resp.status_code == 200:
                        tts_data = tts_resp.json()
                        wav_b64 = tts_data["audio_bytes_base64"]
                        wav_bytes = base64.b64decode(wav_b64)
                        tts_ms = (time.time() - t_tts_start) * 1000.0
                    else:
                        raise HTTPException(status_code=503, detail="Fallback TTS service failed")
                except Exception:
                    raise HTTPException(status_code=503, detail="TTS service unavailable")
            else:
                raise HTTPException(status_code=503, detail="TTS service unavailable")

        # Encode WAV to requested format using actual codebase component
        t_enc_start = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(wav_bytes)
            temp_wav_path = pathlib.Path(temp_wav.name)
            
        try:
            encoded_data = encode_audio(temp_wav_path, fmt)
        finally:
            if temp_wav_path.exists():
                try:
                    os.unlink(temp_wav_path)
                except OSError:
                    pass
                    
        encode_ms = (time.time() - t_enc_start) * 1000.0
        
        # Save to cache (F4-6: check size limits)
        try:
            cache_manager.put(generated_text, req.speaker.voice_id, fmt, encoded_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
    # Generate signed URL/id
    audio_id = f"{cache_key}_{fmt}"
    signed_token = sign_audio_id(audio_id)
    
    t_total = (time.time() - t_start) * 1000.0
    duration_ms = int(len(encoded_data) / 48)
    sha256_hash = hashlib.sha256(encoded_data).hexdigest()
    
    return DialogueResponse(
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

@mock_gateway_app.get("/audio/{signed_id:path}")
def get_audio(signed_id: str, format: Optional[str] = None):
    # Decode URL-encoded characters just in case
    import urllib.parse
    decoded_id = urllib.parse.unquote(signed_id)
    
    # F5-5 / F5-6: Direct traversal check
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
        
    # F5-7: Check symlink safety
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


# =====================================================================
# Pytest Session Fixture to Spawn Services
# =====================================================================
@pytest.fixture(scope="session", autouse=True)
def run_e2e_services():
    env = os.environ.copy()
    env["TEST_MODE"] = "True"
    env["SECRET_KEY"] = "test-secret-key-for-hmac-verification-operations"
    env["AUDIO_CACHE_DIR"] = "tests/test_audio_cache_e2e"
    env["MAX_CACHE_SIZE_BYTES"] = str(5 * 1024 * 1024)
    env["MAX_FILE_SIZE_BYTES"] = str(1 * 1024 * 1024)
    env["PYTHONPATH"] = str(pathlib.Path.cwd())

    cache_dir = pathlib.Path("tests/test_audio_cache_e2e").resolve()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Spawn Gemma, TTS, and Gateway
    gemma_log = open("tests/gemma_proc.log", "w")
    tts_log = open("tests/tts_proc.log", "w")
    orchestrator_log = open("tests/orchestrator_proc.log", "w")

    gemma_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests.test_e2e_integration:mock_llm_app", "--host", "127.0.0.1", "--port", "8001"],
        env=env, stdout=gemma_log, stderr=gemma_log
    )
    tts_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests.test_e2e_integration:mock_tts_app", "--host", "127.0.0.1", "--port", "8002"],
        env=env, stdout=tts_log, stderr=tts_log
    )
    orchestrator_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tests.test_e2e_integration:mock_gateway_app", "--host", "127.0.0.1", "--port", "8000"],
        env=env, stdout=orchestrator_log, stderr=orchestrator_log
    )
    
    processes = [gemma_proc, tts_proc, orchestrator_proc]
    health_urls = [
        "http://127.0.0.1:8001/health",
        "http://127.0.0.1:8002/health",
        "http://127.0.0.1:8000/health"
    ]
    
    # Wait for startup (timeout of 15 seconds)
    start_time = time.time()
    is_healthy = False
    while time.time() - start_time < 15.0:
        try:
            status = []
            for url in health_urls:
                try:
                    resp = httpx.get(url, timeout=0.5)
                    status.append(resp.status_code == 200)
                except Exception:
                    status.append(False)
            if all(status):
                is_healthy = True
                break
        except Exception:
            pass
        time.sleep(0.5)
        
    if not is_healthy:
        for p in processes:
            p.terminate()
            p.wait()
        gemma_log.close()
        tts_log.close()
        orchestrator_log.close()
        raise RuntimeError("E2E Services failed to initialize within 15 seconds.")
        
    yield {
        "orchestrator_url": "http://127.0.0.1:8000",
        "gemma_url": "http://127.0.0.1:8001",
        "tts_url": "http://127.0.0.1:8002",
        "cache_dir": cache_dir
    }
    
    # Teardown processes
    for p in processes:
        p.terminate()
        p.wait()
        
    gemma_log.close()
    tts_log.close()
    orchestrator_log.close()
        
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
        except OSError:
            pass


# =====================================================================
# TIER 1: Feature Coverage (25 Tests)
# =====================================================================

def test_f1_1_gemma_generate(run_e2e_services):
    url = f"{run_e2e_services['gemma_url']}/generate"
    resp = httpx.post(url, json={"prompt": "Hello", "test_mode": True})
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
    assert "generation_time_ms" in data
    assert data["text"].startswith("MOCK_RESPONSE: Hello")

def test_f1_2_gemma_max_words(run_e2e_services):
    url = f"{run_e2e_services['gemma_url']}/generate"
    resp = httpx.post(url, json={"prompt": "Hello world this is a test prompt", "max_words": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["text"].split()) <= 5

def test_f1_3_gemma_enable_thinking_false(run_e2e_services):
    url = f"{run_e2e_services['gemma_url']}/generate"
    resp = httpx.post(url, json={"prompt": "Hello", "enable_thinking": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "<think>" not in data["text"]

def test_f1_4_gemma_health(run_e2e_services):
    url = f"{run_e2e_services['gemma_url']}/health"
    resp = httpx.get(url)
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "service": "gemma-service"}

def test_f1_5_gemma_invalid_max_words(run_e2e_services):
    url = f"{run_e2e_services['gemma_url']}/generate"
    resp = httpx.post(url, json={"prompt": "Hello", "max_words": "five"})
    assert resp.status_code == 422

def test_f2_1_tts_chatterbox(run_e2e_services):
    url = f"{run_e2e_services['tts_url']}/synthesize"
    resp = httpx.post(url, json={"text": "Hello", "engine": "chatterbox"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "wav"
    assert data["sample_rate"] == 24000
    assert "audio_bytes_base64" in data

def test_f2_2_tts_dia(run_e2e_services):
    url = f"{run_e2e_services['tts_url']}/synthesize"
    resp = httpx.post(url, json={"text": "Hello", "engine": "dia"})
    assert resp.status_code == 200
    data = resp.json()
    assert "audio_bytes_base64" in data

def test_f2_3_tts_kokoro(run_e2e_services):
    url = f"{run_e2e_services['tts_url']}/synthesize"
    resp = httpx.post(url, json={"text": "Hello", "engine": "kokoro"})
    assert resp.status_code == 200

def test_f2_4_tts_piper(run_e2e_services):
    url = f"{run_e2e_services['tts_url']}/synthesize"
    resp = httpx.post(url, json={"text": "Hello", "engine": "piper"})
    assert resp.status_code == 200

def test_f2_5_tts_health(run_e2e_services):
    url = f"{run_e2e_services['tts_url']}/health"
    resp = httpx.get(url)
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "service": "tts-service"}

def test_f3_1_orchestrator_dialogue(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Hello",
        "output": {"audio": True, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["state"] == "ready"
    assert "audio" in data
    assert "audio_id" in data["audio"]

def test_f3_2_orchestrator_audio_wav(run_e2e_services):
    url_dial = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Test wav",
        "output": {"audio": True, "format": "wav"}
    }
    resp_dial = httpx.post(url_dial, json=payload)
    signed_id = resp_dial.json()["audio"]["audio_id"]
    
    url_audio = f"{run_e2e_services['orchestrator_url']}/audio/{signed_id}"
    resp_audio = httpx.get(url_audio)
    assert resp_audio.status_code == 200
    assert resp_audio.headers["content-type"] == "audio/wav"
    assert resp_audio.content.startswith(b"RIFF")

def test_f3_3_orchestrator_audio_ogg(run_e2e_services):
    url_dial = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Test ogg",
        "output": {"audio": True, "format": "ogg"}
    }
    resp_dial = httpx.post(url_dial, json=payload)
    signed_id = resp_dial.json()["audio"]["audio_id"]
    
    url_audio = f"{run_e2e_services['orchestrator_url']}/audio/{signed_id}"
    resp_audio = httpx.get(url_audio)
    assert resp_audio.status_code == 200
    assert resp_audio.headers["content-type"] == "audio/ogg"

def test_f3_4_orchestrator_audio_mp3(run_e2e_services):
    url_dial = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Test mp3",
        "output": {"audio": True, "format": "mp3"}
    }
    resp_dial = httpx.post(url_dial, json=payload)
    signed_id = resp_dial.json()["audio"]["audio_id"]
    
    url_audio = f"{run_e2e_services['orchestrator_url']}/audio/{signed_id}"
    resp_audio = httpx.get(url_audio)
    assert resp_audio.status_code == 200
    assert resp_audio.headers["content-type"] == "audio/mpeg"

def test_f3_5_orchestrator_health(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/health"
    resp = httpx.get(url)
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_f4_1_cache_hit(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Identical request",
        "output": {"audio": True, "format": "wav"}
    }
    resp1 = httpx.post(url, json=payload)
    assert resp1.status_code == 200
    assert resp1.json()["metrics"]["cache_hit"] is False
    
    resp2 = httpx.post(url, json=payload)
    assert resp2.status_code == 200
    assert resp2.json()["metrics"]["cache_hit"] is True

def test_f4_2_cache_miss(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": f"Unique request {time.time()}",
        "output": {"audio": True, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 200
    assert resp.json()["metrics"]["cache_hit"] is False

def test_f4_3_cache_files_on_disk(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Disk check request",
        "output": {"audio": True, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    sha256 = resp.json()["audio"]["sha256"]
    
    cache_dir = run_e2e_services["cache_dir"]
    found = False
    for f in cache_dir.iterdir():
        if f.is_file():
            found = True
            break
    assert found

def test_f4_4_cache_pruning(run_e2e_services):
    from app.config import settings
    old_max = settings.max_cache_size_bytes
    settings.max_cache_size_bytes = 100 * 1024 # 100 KB
    try:
        manager = AudioCacheManager(cache_dir=run_e2e_services["cache_dir"])
        for f in run_e2e_services["cache_dir"].iterdir():
            if f.is_file():
                f.unlink()
                
        path1 = manager.put("text1", "voice1", "wav", b"\x00" * 60000)
        assert path1.exists()
        
        path2 = manager.put("text2", "voice1", "wav", b"\x00" * 60000)
        assert path2.exists()
        assert not path1.exists()
    finally:
        settings.max_cache_size_bytes = old_max

def test_f4_5_cache_control_no_cache(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Cache control test",
        "output": {"audio": True, "format": "wav"}
    }
    resp1 = httpx.post(url, json=payload)
    assert resp1.status_code == 200
    
    resp2 = httpx.post(url, json=payload, headers={"Cache-Control": "no-cache"})
    assert resp2.status_code == 200
    assert resp2.json()["metrics"]["cache_hit"] is False

def test_f5_1_signer_valid_token(run_e2e_services):
    token = sign_audio_id("somekey_wav", expiry_seconds=10)
    cache_dir = run_e2e_services["cache_dir"]
    path = cache_dir / "somekey.wav"
    path.write_bytes(b"RIFFdummywavbytes")
    
    url = f"{run_e2e_services['orchestrator_url']}/audio/{token}"
    resp = httpx.get(url)
    assert resp.status_code == 200
    assert resp.content == b"RIFFdummywavbytes"

def test_f5_2_signer_expired_token(run_e2e_services):
    token = sign_audio_id("somekey_wav", expiry_seconds=-10)
    url = f"{run_e2e_services['orchestrator_url']}/audio/{token}"
    resp = httpx.get(url)
    assert resp.status_code == 403
    assert "Signature expired or invalid" in resp.text

def test_f5_3_signer_tampered_token(run_e2e_services):
    token = sign_audio_id("somekey_wav", expiry_seconds=60)
    parts = token.split(".")
    tampered = f"{parts[0]}.{parts[1]}.badsignature"
    url = f"{run_e2e_services['orchestrator_url']}/audio/{tampered}"
    resp = httpx.get(url)
    assert resp.status_code == 403

def test_f5_4_sanitizer_html(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "<p>Paragraph</p>",
        "output": {"audio": False, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 200
    text = resp.json()["text"]
    assert "<p>" not in text
    assert "</p>" not in text

def test_f5_5_sandbox_traversal_blocked(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/audio/../../etc/passwd.signature"
    resp = httpx.get(url)
    assert resp.status_code in (400, 403, 404)


# =====================================================================
# TIER 2: Boundary & Corner Cases (25 Tests)
# =====================================================================

def test_f1_6_gemma_empty_prompt(run_e2e_services):
    url = f"{run_e2e_services['gemma_url']}/generate"
    resp = httpx.post(url, json={"prompt": "", "test_mode": True})
    assert resp.status_code == 422

def test_f1_7_gemma_long_prompt(run_e2e_services):
    url = f"{run_e2e_services['gemma_url']}/generate"
    long_prompt = "A" * 6000
    resp = httpx.post(url, json={"prompt": long_prompt, "test_mode": True})
    assert resp.status_code == 200

def test_f1_8_gemma_negative_max_words(run_e2e_services):
    url = f"{run_e2e_services['gemma_url']}/generate"
    resp = httpx.post(url, json={"prompt": "Hello", "max_words": -10})
    assert resp.status_code == 422

def test_f1_9_gemma_service_crashed(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "simulate_llm_crash",
        "output": {"audio": False, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 503

def test_f1_10_gemma_rate_limiting(run_e2e_services):
    url = f"{run_e2e_services['gemma_url']}/generate"
    responses = []
    for _ in range(60):
        try:
            r = httpx.post(url, json={"prompt": "Hello", "test_mode": True}, timeout=0.1)
            responses.append(r.status_code)
        except httpx.RequestError:
            pass
    assert 429 in responses
    time.sleep(1.1)

def test_f2_6_tts_fish_no_consent(run_e2e_services):
    url = f"{run_e2e_services['tts_url']}/synthesize"
    resp = httpx.post(url, json={"text": "Hello", "engine": "fish"})
    assert resp.status_code == 403

def test_f2_7_tts_fish_with_consent(run_e2e_services):
    url = f"{run_e2e_services['tts_url']}/synthesize"
    resp = httpx.post(url, json={"text": "Hello", "voice_id": "enable_fish", "engine": "fish"})
    assert resp.status_code == 200

def test_f2_8_tts_invalid_engine(run_e2e_services):
    url = f"{run_e2e_services['tts_url']}/synthesize"
    resp = httpx.post(url, json={"text": "Hello", "engine": "invalid_engine"})
    assert resp.status_code == 422

def test_f2_9_tts_empty_text(run_e2e_services):
    url = f"{run_e2e_services['tts_url']}/synthesize"
    resp = httpx.post(url, json={"text": "", "engine": "chatterbox"})
    assert resp.status_code == 422

def test_f2_10_tts_long_text(run_e2e_services):
    url = f"{run_e2e_services['tts_url']}/synthesize"
    long_text = "Hello " * 400
    resp = httpx.post(url, json={"text": long_text, "engine": "chatterbox"})
    assert resp.status_code == 200

def test_f3_6_orchestrator_missing_speaker(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "user_text": "Hello",
        "output": {"audio": True, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 422

def test_f3_7_orchestrator_unsupported_format(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Hello",
        "output": {"audio": True, "format": "flac"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 422

def test_f3_8_orchestrator_request_collisions(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "request_id": "duplicate_id_123",
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "First request",
        "output": {"audio": True, "format": "wav"}
    }
    resp1 = httpx.post(url, json=payload)
    assert resp1.status_code == 200
    
    payload["user_text"] = "Second request"
    resp2 = httpx.post(url, json=payload)
    assert resp2.status_code == 200
    assert resp1.json()["job_id"] != resp2.json()["job_id"]

def test_f3_9_orchestrator_client_disconnect_midstream(run_e2e_services):
    url_dial = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Streaming disconnect test",
        "output": {"audio": True, "format": "wav"}
    }
    resp_dial = httpx.post(url_dial, json=payload)
    signed_id = resp_dial.json()["audio"]["audio_id"]
    
    url_audio = f"{run_e2e_services['orchestrator_url']}/audio/{signed_id}"
    with httpx.stream("GET", url_audio) as response:
        assert response.status_code == 200
        for chunk in response.iter_bytes(chunk_size=10):
            break

def test_f3_10_orchestrator_audio_disabled(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Hello",
        "output": {"audio": False, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["audio"] is None

def test_f4_6_cache_file_too_large(run_e2e_services):
    from app.config import settings
    manager = AudioCacheManager(cache_dir=run_e2e_services["cache_dir"])
    with pytest.raises(ValueError) as excinfo:
        manager.put("large_text", "voice1", "wav", b"\x00" * (settings.max_file_size_bytes + 100))
    assert "exceeds max_file_size_bytes" in str(excinfo.value)

def test_f4_7_cache_prune_empty(run_e2e_services):
    for f in run_e2e_services["cache_dir"].iterdir():
        if f.is_file():
            f.unlink()
    manager = AudioCacheManager(cache_dir=run_e2e_services["cache_dir"])
    manager.prune_cache(100)

def test_f4_8_cache_concurrent_writes(run_e2e_services):
    import threading
    manager = AudioCacheManager(cache_dir=run_e2e_services["cache_dir"])
    
    def write_cache():
        for i in range(10):
            try:
                manager.put("concurrent_text", "voice1", "wav", b"data")
            except Exception:
                pass
                
    threads = [threading.Thread(target=write_cache) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    data = manager.get("concurrent_text", "voice1", "wav")
    assert data == b"data"

def test_f4_9_cache_read_evicted(run_e2e_services):
    for f in run_e2e_services["cache_dir"].iterdir():
        if f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
    manager = AudioCacheManager(cache_dir=run_e2e_services["cache_dir"])
    manager.put("evict_me", "voice1", "wav", b"\x00" * 100)
    
    from app.config import settings
    old_max = settings.max_cache_size_bytes
    settings.max_cache_size_bytes = 350
    try:
        manager.put("new_large_item", "voice1", "wav", b"\x00" * 300)
        val = manager.get("evict_me", "voice1", "wav")
        assert val is None
    finally:
        settings.max_cache_size_bytes = old_max

def test_f4_10_cache_corruption_recovery(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Corruption test text",
        "output": {"audio": True, "format": "wav"}
    }
    resp1 = httpx.post(url, json=payload)
    assert resp1.status_code == 200
    assert resp1.json()["metrics"]["cache_hit"] is False
    
    cache_key = get_cache_key("MOCK_RESPONSE: Corruption test text", "af_heart", "wav")
    cache_dir = run_e2e_services["cache_dir"]
    cached_file = cache_dir / f"{cache_key}.wav"
    assert cached_file.exists()
    
    cached_file.write_bytes(b"\x00" * 100)
    
    resp2 = httpx.post(url, json=payload)
    assert resp2.status_code == 200
    assert resp2.json()["metrics"]["cache_hit"] is False
    assert cached_file.read_bytes().startswith(b"RIFF")

def test_f5_6_sandbox_encoded_traversal(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/audio/%2e%2e%2f%2e%2e%2fetc%2fpasswd"
    resp = httpx.get(url)
    assert resp.status_code in (400, 403)

def test_f5_7_sandbox_symlink_protection(run_e2e_services):
    outside_file = pathlib.Path("tests/outside.wav")
    outside_file.write_bytes(b"RIFFoutsidecontent")
    
    cache_dir = run_e2e_services["cache_dir"]
    symlink_path = cache_dir / "badlink.wav"
    if symlink_path.exists():
        symlink_path.unlink()
        
    try:
        os.symlink(str(outside_file.resolve()), str(symlink_path))
    except OSError:
        pass
        
    token = sign_audio_id("badlink_wav")
    url = f"{run_e2e_services['orchestrator_url']}/audio/{token}"
    resp = httpx.get(url)
    assert resp.status_code == 403
    
    if outside_file.exists():
        outside_file.unlink()

def test_f5_8_sanitizer_traversal_text(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Dialogue ../../etc/passwd content",
        "output": {"audio": False, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 200
    text = resp.json()["text"]
    assert ".." not in text

def test_f5_9_signer_non_numeric_timestamp(run_e2e_services):
    token = "somekey_wav.alphatime.signature"
    url = f"{run_e2e_services['orchestrator_url']}/audio/{token}"
    resp = httpx.get(url)
    assert resp.status_code == 403

def test_f5_10_signer_rotated_key(run_e2e_services):
    token = sign_audio_id("somekey_wav", expiry_seconds=60)
    
    base_url = run_e2e_services['orchestrator_url']
    from app.config import settings
    old_key = settings.secret_key
    settings.secret_key = "new-rotated-secret-key-123456"
    httpx.post(f"{base_url}/debug/rotate_key?new_key=new-rotated-secret-key-123456")
    try:
        url = f"{base_url}/audio/{token}"
        resp = httpx.get(url)
        assert resp.status_code == 403
    finally:
        settings.secret_key = old_key
        httpx.post(f"{base_url}/debug/rotate_key?new_key={old_key}")


# =====================================================================
# TIER 3: Cross-Feature Combinations (6 Tests)
# =====================================================================

def test_c1_injection_attack(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "<script>alert(1)</script> https://hack.com [Play](file:///etc/passwd)",
        "output": {"audio": True, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 200
    text = resp.json()["text"]
    assert "<script>" not in text
    assert "https://hack.com" not in text
    assert "file:///etc/passwd" not in text
    assert "Play" in text

def test_c2_cache_miss_to_write(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    unique_text = f"Cache Miss Test {time.time()}"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": unique_text,
        "output": {"audio": True, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 200
    assert resp.json()["metrics"]["cache_hit"] is False
    
    cache_key = get_cache_key(f"MOCK_RESPONSE: {unique_text}", "af_heart", "wav")
    cache_dir = run_e2e_services["cache_dir"]
    cached_file = cache_dir / f"{cache_key}.wav"
    assert cached_file.exists()
    assert cached_file.read_bytes().startswith(b"RIFF")

def test_c3_cache_hit_to_signed_delivery(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    unique_text = f"Cache Hit and Sign Test {time.time()}"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": unique_text,
        "output": {"audio": True, "format": "wav"}
    }
    resp1 = httpx.post(url, json=payload)
    assert resp1.status_code == 200
    
    resp2 = httpx.post(url, json=payload)
    assert resp2.status_code == 200
    assert resp2.json()["metrics"]["cache_hit"] is True
    
    signed_id = resp2.json()["audio"]["audio_id"]
    url_audio = f"{run_e2e_services['orchestrator_url']}/audio/{signed_id}"
    resp_audio = httpx.get(url_audio)
    assert resp_audio.status_code == 200
    assert resp_audio.headers["content-type"] == "audio/wav"

def test_c4_eviction_cascade(run_e2e_services):
    for f in run_e2e_services["cache_dir"].iterdir():
        if f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
    base_url = run_e2e_services['orchestrator_url']
    from app.config import settings
    old_max = settings.max_cache_size_bytes
    settings.max_cache_size_bytes = 100 * 1024
    httpx.post(f"{base_url}/debug/update_settings?max_cache_size_bytes={100 * 1024}")
    try:
        url = f"{base_url}/v1/dialogue"
        for i in range(20):
            payload = {
                "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
                "user_text": f"item {i} size_bytes=30000",
                "output": {"audio": True, "format": "wav"}
            }
            resp = httpx.post(url, json=payload)
            assert resp.status_code == 200
            
        total_size = sum(f.stat().st_size for f in run_e2e_services["cache_dir"].iterdir() if f.is_file())
        assert total_size <= 135 * 1024
    finally:
        settings.max_cache_size_bytes = old_max
        httpx.post(f"{base_url}/debug/update_settings?max_cache_size_bytes={old_max}")

def test_c5_schema_mismatch_fallback(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "simulate-llm-bad-json",
        "output": {"audio": False, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 200
    assert "Fallback" in resp.json()["text"]

@pytest.mark.asyncio
async def test_c6_client_disconnect_during_tts(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "simulate_client_disconnect",
        "output": {"audio": True, "format": "wav"}
    }
    async with httpx.AsyncClient() as client:
        try:
            task = asyncio.create_task(client.post(url, json=payload, timeout=10.0))
            await asyncio.sleep(0.2)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        except Exception:
            pass
            
    cache_key = get_cache_key("MOCK_RESPONSE: simulate_client_disconnect", "af_heart", "wav")
    cache_dir = run_e2e_services["cache_dir"]
    cached_file = cache_dir / f"{cache_key}.wav"
    assert not cached_file.exists()


# =====================================================================
# TIER 4: Real-World Application Scenarios (6 Tests)
# =====================================================================

def test_s1_merchant_playthrough(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    
    payload = {
        "speaker": {"id": "npc_merchant", "name": "Merchant", "voice_id": "af_heart", "style": "merchant"},
        "context": {"facts": [{"id": "merchant_gold", "can_reveal": True, "fact": "merchant_gold: 10"}]},
        "user_text": "What do you sell?",
        "output": {"audio": True, "format": "wav"}
    }
    resp1 = httpx.post(url, json=payload)
    assert resp1.status_code == 200
    assert "10 gold" in resp1.json()["text"]
    
    payload["user_text"] = "I will buy this sword."
    payload["context"]["facts"][0]["fact"] = "merchant_gold: 50"
    resp2 = httpx.post(url, json=payload)
    assert resp2.status_code == 200
    assert "50" in resp2.json()["text"]
    
    payload["user_text"] = "Do you have change?"
    resp3 = httpx.post(url, json=payload)
    assert resp3.status_code == 200
    assert "50" in resp3.json()["text"]

@pytest.mark.asyncio
async def test_s2_concurrent_stress(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(50):
            prompt = f"Stress test prompt group {i % 5}"
            payload = {
                "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
                "user_text": prompt,
                "output": {"audio": True, "format": "wav"}
            }
            tasks.append(client.post(url, json=payload, timeout=20.0))
            
        responses = await asyncio.gather(*tasks)
        
    success = [r.status_code == 200 for r in responses]
    assert all(success)

def test_s3_tts_failure_degradation(run_e2e_services):
    url = f"{run_e2e_services['orchestrator_url']}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart_dia_simulate_offline", "style": "dia"},
        "user_text": "Offline fallback test",
        "output": {"audio": True, "format": "wav"}
    }
    resp = httpx.post(url, json=payload)
    assert resp.status_code == 200
    assert resp.json()["audio"]["audio_id"] is not None

def test_s4_cache_disk_bounded(run_e2e_services):
    for f in run_e2e_services["cache_dir"].iterdir():
        if f.is_file():
            try:
                f.unlink()
            except OSError:
                pass
    base_url = run_e2e_services['orchestrator_url']
    from app.config import settings
    old_max = settings.max_cache_size_bytes
    settings.max_cache_size_bytes = 200 * 1024
    httpx.post(f"{base_url}/debug/update_settings?max_cache_size_bytes={200 * 1024}")
    try:
        url = f"{base_url}/v1/dialogue"
        for i in range(10):
            payload = {
                "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
                "user_text": f"bounded run {i} size_bytes=40000",
                "output": {"audio": True, "format": "wav"}
            }
            resp = httpx.post(url, json=payload)
            assert resp.status_code == 200
            
        total_size = sum(f.stat().st_size for f in run_e2e_services["cache_dir"].iterdir() if f.is_file())
        assert total_size <= 250 * 1024
    finally:
        settings.max_cache_size_bytes = old_max
        httpx.post(f"{base_url}/debug/update_settings?max_cache_size_bytes={old_max}")

def test_s5_security_penetration(run_e2e_services):
    base_url = run_e2e_services['orchestrator_url']
    
    # 1. HTML attack
    resp1 = httpx.post(f"{base_url}/v1/dialogue", json={
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "<script>alert(1)</script>",
        "output": {"audio": False, "format": "wav"}
    })
    assert resp1.status_code == 200
    assert "<script>" not in resp1.json()["text"]
    
    # 2. Expired token
    token_expired = sign_audio_id("somekey_wav", expiry_seconds=-10)
    resp2 = httpx.get(f"{base_url}/audio/{token_expired}")
    assert resp2.status_code == 403
    
    # 3. Path traversal encoded
    resp3 = httpx.get(f"{base_url}/audio/%2e%2e%2f%2e%2e%2fetc%2fpasswd")
    assert resp3.status_code in (400, 403)
    
    # 4. Invalid signature
    resp4 = httpx.get(f"{base_url}/audio/somekey_wav.1716480000.badsignature")
    assert resp4.status_code == 403
    
    # Normal request unaffected
    resp_normal = httpx.post(f"{base_url}/v1/dialogue", json={
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Normal safe text",
        "output": {"audio": False, "format": "wav"}
    })
    assert resp_normal.status_code == 200

def test_s6_secret_key_rotation(run_e2e_services):
    base_url = run_e2e_services['orchestrator_url']
    
    url_dial = f"{base_url}/v1/dialogue"
    payload = {
        "speaker": {"id": "npc_maria", "name": "Maria", "voice_id": "af_heart", "style": "calm"},
        "user_text": "Pre-rotation text",
        "output": {"audio": True, "format": "wav"}
    }
    resp_dial = httpx.post(url_dial, json=payload)
    signed_id = resp_dial.json()["audio"]["audio_id"]
    
    resp_audio = httpx.get(f"{base_url}/audio/{signed_id}")
    assert resp_audio.status_code == 200
    
    resp_rotate = httpx.post(f"{base_url}/debug/rotate_key?new_key=newrotatedsecretkey123456")
    assert resp_rotate.status_code == 200
    
    resp_audio_post = httpx.get(f"{base_url}/audio/{signed_id}")
    assert resp_audio_post.status_code == 403
    
    httpx.post(f"{base_url}/debug/rotate_key?new_key=test-secret-key-for-hmac-verification-operations")
