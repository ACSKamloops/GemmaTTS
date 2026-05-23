# Architecture

## Service Topology

```
┌──────────────────────────────────────────────────────┐
│                Orchestrator API (:8000)                │
│   POST /v1/dialogue  │  GET /audio/{signed_id}        │
└─────────┬───────────────────────────┬────────────────┘
          │                           │
          ▼                           ▼
┌─────────────────┐    ┌──────────────────────────────┐
│ Gemma Service    │    │ TTS Service (:8002)           │
│ (:8001)          │    │                              │
│                  │    │  ┌─────────────────────────┐ │
│ google/gemma-4-  │    │  │ Engine Router            │ │
│ E4B-it (BF16)   │    │  │                         │ │
│                  │    │  │  chatterbox → Chatterbox │ │
│ Transformers     │    │  │  dia        → Dia 1.6B  │ │
│ AutoModelFor-    │    │  │  kokoro     → Kokoro 82M│ │
│ CausalLM        │    │  │  f5_tts     → F5-TTS    │ │
│                  │    │  │  piper      → Piper     │ │
│                  │    │  │  fish       → Fish (opt)│ │
│                  │    │  └─────────────────────────┘ │
└─────────────────┘    │                              │
                       │  ┌─────────────────────────┐ │
                       │  │ Audio Pipeline           │ │
                       │  │  1. Resample → 24kHz     │ │
                       │  │  2. DC offset removal    │ │
                       │  │  3. Silence trim         │ │
                       │  │  4. LUFS normalization   │ │
                       │  │  5. Clipping prevention  │ │
                       │  └─────────────────────────┘ │
                       └──────────────────────────────┘
```

## Model Details

### LLM: Gemma 4 E4B-it
- **Source**: `google/gemma-4-E4B-it` (HuggingFace, gated)
- **Size**: ~16GB (safetensors)
- **Precision**: BF16 on CUDA
- **Runtime**: HuggingFace Transformers (`AutoModelForCausalLM`)
- **Thinking**: Disabled (`enable_thinking=False`) for low-latency dialogue

### TTS: Chatterbox (Default)
- **Source**: `chatterbox-tts` pip package (Resemble AI)
- **API**: `ChatterboxTTS.from_pretrained(device="cuda")` → `model.generate(text)`
- **Output**: 24kHz WAV
- **Features**: Zero-shot voice cloning, emotion control, 23+ languages
- **Weights**: Auto-downloaded by pip package

### TTS: Dia 1.6B (Dialogue)
- **Source**: `nari-labs/Dia-1.6B-0626` (HuggingFace)
- **API**: `DiaForConditionalGeneration` + `AutoProcessor` (transformers)
- **Output**: 44100Hz WAV
- **Features**: Multi-speaker dialogue with `[S1]`/`[S2]` tags, non-verbal cues
- **Requires**: `descript-audio-codec` for audio decoding

### TTS: Kokoro 82M (Fast Fallback)
- **Source**: `onnx-community/Kokoro-82M-ONNX` (HuggingFace)
- **API**: `kokoro-onnx` pip package → `Kokoro.create(text, voice)`
- **Output**: 24kHz WAV
- **Features**: 60+ voice embeddings, voice blending, fast ONNX inference

### TTS: Piper (Emergency Fallback)
- **Source**: `rhasspy/piper-voices` (HuggingFace)
- **API**: `piper-tts` pip package → `PiperVoice.load()` → `synthesize()`
- **Output**: 22050Hz WAV (resampled to 24kHz by pipeline)

## Audio Pipeline

All TTS output passes through `app/audio/pipeline.py`:
1. **Resample** to 24kHz target via librosa
2. **DC offset removal** — subtract mean to prevent transient bias
3. **Silence trim** — amplitude threshold with safety margin
4. **LUFS normalization** — EBU R128 to -23 LUFS via pyloudnorm
5. **Clipping prevention** — soft-knee limiter + peak scaling

## Security

- **Input sanitization**: HTML, URLs, dot-traversal patterns stripped
- **Path traversal protection**: Absolute path validation, symlink rejection
- **HMAC-SHA256 signed URLs**: Audio files served with expiring signatures (5 min default)
- **Rate limiting**: 40 req/sec per service
