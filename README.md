# GemmaTTS — Local Voice Agent Stack

Multi-model TTS orchestration service powered by Gemma 4 E4B-it for text generation and multiple high-quality TTS engines for voice synthesis.

## Architecture

```text
User/application request
  → Orchestrator (FastAPI, port 8000)
  → Gemma 4 E4B-it text generation (port 8001)
  → Text safety/normalization
  → TTS synthesis (port 8002, selectable engine)
  → Audio pipeline (normalize, trim, limit)
  → Audio cache (hash-addressed, HMAC-signed URLs)
  → Output
```

## TTS Engines

| Engine | Model | Quality | Speed | Use Case |
|--------|-------|---------|-------|----------|
| **Chatterbox** | Resemble AI 0.5B | ★★★★★ | Medium | Default — voice cloning, emotion control |
| **Dia** | Nari Labs 1.6B | ★★★★★ | Slow | Multi-speaker dialogue with [S1]/[S2] tags |
| **Kokoro** | 82M ONNX | ★★★★ | Fast | Fast fallback — many voices, low latency |
| **F5-TTS** | SWivid 1.2B | ★★★★ | Medium | Voice cloning via reference audio |
| **Piper** | Lessac Medium | ★★★ | Fastest | Emergency fallback — always works |

## Requirements

- **GPU**: NVIDIA GPU with ≥16GB VRAM (RTX 5090 recommended)
- **Python**: 3.11+
- **CUDA**: 12.x
- **OS**: Ubuntu/WSL2

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model weights (requires HF token for Gemma)
export HF_TOKEN="your_huggingface_token"
python scripts/download_models.py

# 4. Start services
uvicorn app.services.orchestrator_api:app --port 8000 &
uvicorn app.services.gemma_service:app --port 8001 &
uvicorn app.services.tts_service:app --port 8002 &
```

## API Endpoints

### Orchestrator (port 8000)
- `POST /v1/dialogue` — Generate text + synthesize speech
- `GET /audio/{signed_id}` — Retrieve cached audio (HMAC-signed)

### Gemma Service (port 8001)
- `POST /generate` — Text generation with Gemma 4 E4B-it

### TTS Service (port 8002)
- `POST /synthesize` — Speech synthesis with selectable engine
- `GET /health` — Health check

## Model Downloads

```bash
# All models (~25GB total)
python scripts/download_models.py

# Skip Gemma (for TTS-only testing)
python scripts/download_models.py --skip-gemma

# Test models only (lightweight, for CI)
python scripts/download_models.py --test-only
```

**Note:** Chatterbox weights are auto-downloaded by the `chatterbox-tts` pip package on first use.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_tts_service.py -v
python -m pytest tests/test_audio_quality.py -v
```

## Project Structure

```text
app/
  config.py              Settings (env-based)
  audio/
    pipeline.py          Audio normalization/trimming/limiting
    cache.py             Hash-addressed audio cache
    encoder.py           Format conversion (WAV/OGG/MP3)
    signer.py            HMAC-SHA256 URL signing
  safety/
    text_sanitizer.py    Input sanitization
    output_validator.py  Output validation
  services/
    orchestrator_api.py  Central orchestrator (port 8000)
    gemma_service.py     Gemma 4 text generation (port 8001)
    tts_service.py       TTS routing service (port 8002)
    tts/
      chatterbox_worker.py   Resemble AI Chatterbox
      dia_worker.py          Nari Labs Dia 1.6B
      f5_tts_worker.py       F5-TTS
      fish_worker.py         Fish Audio (consent-gated)
      kokoro_worker.py       Kokoro 82M ONNX
      piper_worker.py        Piper
models/                  Downloaded model weights (gitignored)
scripts/
  download_models.py     Model download & verification
tests/                   Test suites
```

## Key References

- [Gemma 4 E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [Dia 1.6B](https://github.com/nari-labs/dia)
- [Kokoro ONNX](https://github.com/thewh1teagle/kokoro-onnx)
- [Piper TTS](https://github.com/OHF-Voice/piper1-gpl)
- [F5-TTS](https://github.com/SWivid/F5-TTS)
