# GemmaTTS — Local Voice Agent Stack

Unified single-process Gemma 4 E4B-it LLM and speech synthesis FastAPI application.

## Architecture

```text
User/application request
  → Unified API Server (FastAPI, port 8000)
    → Text generation via Gemma 4 E4B-it (in-process/unified)
    → Text safety/normalization
    → Speech Synthesis (in-process/unified, selectable engine)
    → Audio pipeline (normalize, trim, limit)
    → Audio cache (hash-addressed, HMAC-signed URLs)
  → Output
```

## TTS Engines & Status

| Engine | Model | Quality | Speed | Verification State | Use Case / Notes |
|--------|-------|---------|-------|-------------------|------------------|
| **Chatterbox** | Resemble AI 0.5B | ★★★★★ | Medium | **Default / Verified** | Voice cloning, emotion control |
| **Kokoro** | 82M ONNX | ★★★★ | Fast | **Verified** | Many voices, low latency |
| **Piper** | Lessac Medium | ★★★ | Fastest | **Verified** | Always works, low resource |
| **Dia** | Nari Labs 1.6B | ★★★★★ | Slow | **Experimental** | Multi-speaker dialogue ([S1]/[S2]) |
| **F5-TTS** | SWivid 1.2B | ★★★★ | Medium | **Experimental** | Voice cloning. Disabled by default due to non-commercial license (enable with `ENABLE_F5_TTS=true`) |

## Requirements

- **GPU**: NVIDIA GPU with ≥16GB VRAM (RTX 5090 recommended for real-engine inference)
- **Python**: 3.11+
- **CUDA**: 12.x
- **OS**: Ubuntu/WSL2

## Quick Start

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model Weights
```bash
export HF_TOKEN="your_huggingface_token"
python scripts/download_models.py
```

### 4. Run the Unified Service
```bash
export MODE=dev
export DEBUG_ENABLED=true
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## API Endpoints

The single-process server exposes all endpoints unified under port 8000:

- `POST /v1/dialogue` — Generate text + speech synthesis (unified pipeline)
- `POST /v1/tts` — Speech synthesis with selectable engine and cache lookup
- `POST /synthesize` — Direct in-process speech synthesis
- `POST /synthesize/pcm` — Raw PCM s16le audio stream endpoint
- `POST /synthesize/export` — File download export (WAV/OGG/MP3)
- `GET /audio/{signed_id}` — Retrieve cached audio (HMAC-signed URL verification)
- `POST /generate` — Text generation via Gemma 4 E4B-it
- `GET /health` — Service health checks
- `GET /voices` — List available voice registry metadata

## Configuration Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODE` | `dev` \| `test` \| `real` | `dev` | Execution mode. `real` enables real GPU inference and enforces security policies. |
| `SECRET_KEY` | `str` | *generated* | Key used for signing audio URLs. **Required** in `MODE=real`. |
| `DEBUG_ENABLED` | `bool` | `false` | Enables debug router/endpoints if set to `true`. |
| `ENABLE_F5_TTS` | `bool` | `false` | Set to `true` to enable F5-TTS provider (non-commercial license consent). |
| `KOKORO_PROVIDER_MODE` | `official` \| `legacy_manual_embedding` | `legacy_manual_embedding` | Mode of Kokoro ONNX voice embedding loading. |
| `AUTH_MODE` | `none` \| `token` \| `hmac` | `none` | API authentication mode. |
| `API_TOKEN` | `str` | `None` | Authentication bearer token if `AUTH_MODE=token`. |

## Testing

```bash
# Run all unit/contract mock tests
pytest --ignore=tests/smoke/ -v

# Run real-engine smoke tests under local GPU/WSL environment
pytest tests/smoke/ -v
```

## Benchmarks

Benchmark scripts are provided to measure inference latency and resource footprint under local hardware:

```bash
# Benchmark Gemma 4 E4B-it generation
python benchmarks/benchmark_gemma_5090.py

# Benchmark speech synthesis latency/VRAM
python benchmarks/benchmark_tts_5090.py
```
