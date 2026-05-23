# Benchmark Kokoro, Piper, and optional KittenTTS

## Summary
Measure TTS quality, CPU/RAM/GPU use, generation latency, audio duration, and cache hit behavior.

## Acceptance criteria
- [ ] Kokoro benchmark includes cold start, warm start, 10-word, 40-word, and 120-word lines.
- [ ] Piper benchmark runs through persistent server mode, not per-line CLI model loading.
- [ ] Optional KittenTTS spike is marked experimental and not a release blocker.

## References
- Kokoro model card: https://huggingface.co/hexgrad/Kokoro-82M
- Piper CLI docs: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md
