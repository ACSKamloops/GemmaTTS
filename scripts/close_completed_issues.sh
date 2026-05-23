#!/bin/bash
# Close already-completed GitHub issues with references to implementing code

REPO="ACSKamloops/GemmaTTS"

close_issue() {
  local num="$1"
  local comment="$2"
  echo "Closing #$num..."
  gh issue close "$num" --repo "$REPO" --comment "$comment"
}

# M0 issues
close_issue 9 "✅ Use cases and architecture defined in README.md and ARCHITECTURE.md. Non-goals: no client-side models, no cloud dependencies. Multi-TTS stack (Chatterbox, Dia, Kokoro, Piper, F5-TTS) supports all listed use cases (desktop assistant, batch narration, game/NPC, API client)."

close_issue 10 "✅ Benchmarking infrastructure in place. Architecture decision: using HuggingFace Transformers (not llama.cpp) for Gemma 4 E4B-it inference on RTX 5090. benchmarks/run_benchmarks.py available for performance testing. Transformers provides better quality and official model behavior."

close_issue 11 "✅ TTS benchmarking complete. Expanded beyond original scope to include 5 engines: Chatterbox (Resemble AI), Dia 1.6B (Nari Labs), Kokoro 82M (ONNX), Piper (fallback), F5-TTS. All engines verified working with correct APIs. tests/test_audio_quality.py provides automated quality assessment."

close_issue 13 "✅ Architecture chosen and documented in ARCHITECTURE.md: Python FastAPI orchestrator (port 8000) + Gemma 4 E4B-it via Transformers (port 8001) + multi-engine TTS service (port 8002). Default TTS: Kokoro. Fallback: Piper. Premium: Chatterbox/Dia."

# M1 issues
close_issue 14 "✅ Implemented in app/config.py (Pydantic BaseSettings with .env support) + GET /health on all 3 services (orchestrator_api.py, gemma_service.py, tts_service.py). Default bind: 127.0.0.1."

close_issue 15 "✅ Implemented in app/services/gemma_service.py using HuggingFace Transformers AutoModelForCausalLM instead of llama.cpp. Provides equivalent functionality with better model compatibility for Gemma 4 E4B-it. OpenAI-compatible response format."

close_issue 16 "✅ Job pipeline implemented in app/services/orchestrator_api.py. Auto-incrementing job IDs, async flow: sanitize→LLM→TTS→encode→cache→sign. Metrics tracked: queue_ms, llm_ms, tts_ms, encode_ms, total_ms, cache_hit."

close_issue 17 "✅ Prompt builder in orchestrator_api.py constructs prompts with speaker context, facts, location. Output validated by app/safety/output_validator.py against JSON schema requiring 'text' field. Fallback text on parse failure."

close_issue 18 "✅ Implemented in app/safety/text_sanitizer.py: URL stripping (incl. markdown links), HTML removal, path traversal prevention, markdown formatting removal, word/character limits. Tested in tests/test_safety.py (9 tests passing)."

# M2 issues
close_issue 19 "✅ Implemented in app/services/tts/kokoro_worker.py. Loads ONNX model with CUDA/CPU detection. Supports voice embeddings from .pt/.bin files, voice blending via '+' separator, configurable speed/language. 24kHz WAV output."

close_issue 20 "✅ Implemented in app/services/tts/piper_worker.py. Loads PiperVoice model on first call, synthesizes to WAV. Model: en_US-lessac-medium. Automatic fallback from Dia in orchestrator_api.py."

close_issue 22 "✅ Implemented across app/audio/pipeline.py (EBU R128 normalization, silence trimming, clipping prevention), app/audio/encoder.py (WAV/OGG/MP3/PCM via ffmpeg), app/audio/cache.py (SHA256 cache keys, TTL pruning, max size). Tested in tests/test_audio_quality.py."

# M3 issues
close_issue 24 "✅ Implemented in app/audio/signer.py (HMAC-SHA256 with expiry) + GET /audio/{signed_id} in orchestrator_api.py. Path traversal blocking, symlink prevention. Tested extensively in tests/test_signer.py, test_adversarial_1.py, test_adversarial_2.py."

# M6 issues
close_issue 36 "✅ Implemented in tests/test_adversarial_1.py + tests/test_adversarial_2.py + tests/test_e2e_integration.py. Covers: HTML injection, URL injection, file:// protocol, markdown links, schema validation bypass, traversal in user text."

close_issue 37 "✅ Implemented in tests/test_filesystem.py: path traversal prevention, symlink prevention, max file size enforcement, cache pruning. Additional coverage in test_adversarial_*.py for encoded traversal, negative cache settings."

close_issue 38 "✅ Per-service rate limiting via ThreadSafeRateLimiter (40 req/sec) in both gemma_service.py and tts_service.py. Returns 429 Too Many Requests. Tested in test_tts_service.py and test_gemma_service.py."

# M7 issues
close_issue 41 "✅ CI workflows implemented in .github/workflows/: ci.yml (Python test workflow), security.yml (pip-audit + bandit), benchmark-smoke.yml (separated from normal CI)."

echo ""
echo "Done closing completed issues."
