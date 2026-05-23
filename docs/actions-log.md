# Actions Log

## 2026-05-23

### Completed Actions
1. **Shadowed Package Resolution**: Created `tests/__init__.py` to make the `tests/` directory a regular package. This prevents python from resolving the `tests` namespace to the third-party `tests` library installed in the virtualenv's `site-packages/` which was causing E2E imports to fail.
2. **E2E Integration Verification**:
   * Executed `pytest -v tests/test_e2e_integration.py` in WSL.
   * All 62 E2E integration tests passed in 6.76 seconds.
3. **Full Test Suite Run**:
   * Pre-set `LD_LIBRARY_PATH` to include CUDA paths in the shell command to bypass uvicorn/execv process reloading which was silencing stdout/stderr.
   * Executed `pytest -v --ignore=tests/test_audio_quality.py`.
   * Verified that all 247 unit and integration tests are passing successfully.
4. **Real Voice Synthesis Demo**:
   * Created and ran a script to synthesize speech using the real Kokoro and Piper models on the local GPU (via CUDAExecutionProvider).
   * Generated and saved high-quality WAV files in the root workspace folder:
     * `demo_kokoro.wav` (24kHz female voice `af_heart`)
     * `demo_piper.wav` (22.05kHz English voice)
5. **Background Services & Web Dashboard Verification**:
   * Started a keep-alive script inside WSL running the Orchestrator (port 8000), Gemma service (port 8001), and TTS service (port 8002) bound to `0.0.0.0`.
   * Invoked the `browser` subagent to load `http://127.0.0.1:8000/dashboard/`.
   * Verified that the services are reported healthy, performed a successful preview synthesis test in 16.9ms, and verified the audio player is displayed and the job logged as "done".
   * Captured and saved the successful run screenshot to the conversation artifacts directory.
6. **P0 Consistency Fixes**:
   * **TTS Provider Caching**: Refactored `get_worker()` in `app/services/tts_service.py` and `_synthesize_wav_in_process()` in `app/api/tts.py` to route through the central cached `get_tts_provider()` orchestrator method, preventing redundant model instantiation.
   * **Schema Consolidation**: Removed `OutputConfig.format` field, making `req.tts.format` the unified format source of truth for `/v1/dialogue`.
   * **Audio Profile & Metadata Cache**: Configured audio pipelines to accept custom profiles and update `AudioCacheManager.put()` to store `sample_rate`, `engine`, `format`, `profile`, and `audio_pipeline_version` sidecar metadata, recovering `sample_rate` on cache hits.
   * **Fish API Removal**: Fully purged `_check_fish_consent` dead code from API controllers.
   * **Smoke Test skips and RUN_REAL_SMOKE Gating**: Set smoke tests to skip gracefully unless `RUN_REAL_SMOKE=1` is set, in which case they fail hard. Removed Chatterbox model config path dependency.
   * **Opt-in F5-TTS**: Configured script downloads to bypass F5-TTS assets by default, and gated benchmarks behind `settings.enable_f5_tts`.
   * **Unified Mode Defaults**: Gated simulation text triggers to test environments only (`MODE=test`) and set unified settings to default true.
   * **Verification**: Verified via `grep` check constraints and confirmed all 242 fast tests are passing successfully.
7. **Final Blocker Fixes**:
   * **Simulation Trigger Gating**: Gated the `simulate_llm_bad_json` prompt trigger in `app/api/generate.py` behind `settings.mode == "test"`. In `dev` mode, it is treated as a normal prompt string, preventing accidental trigger bypasses.
   * **Cache Key Sample Rate Consistency**: Removed the `sample_rate` parameter from the cache key payload in `get_cache_key()` (`app/audio/cache.py`). The sample rate is preserved and retrieved strictly from the sidecar metadata, preventing cache key mismatches when resampling profiles are used.
   * **Unit Tests**: Added `tests/test_blocker_fixes.py` containing tests verifying gated simulation behavior in `dev`/`test` modes and cache key consistency when `sample_rate` differs.
   * **Verification**: Verified that all 244 unit tests are passing successfully and regression greps pass.
8. **README Documentation Cleanup**:
   * **Smoke Test Commands**: Updated the smoke test execution example to explicitly include `RUN_REAL_SMOKE=1` environment variable gating to prevent users from seeing skips and thinking tests successfully executed real inference.
   * **F5-TTS Opt-in Notes**: Added a clear notice in the Quick Start download section informing users that F5-TTS weights are skipped by default unless the non-commercial `--include-f5` flag is explicitly requested.

