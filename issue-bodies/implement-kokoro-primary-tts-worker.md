# Implement Kokoro primary TTS worker

## Summary
Load Kokoro once and synthesize speech through a persistent worker.

## Acceptance criteria
- [ ] Kokoro loads once at service startup or first use.
- [ ] Supports voice_id, speed, and language config.
- [ ] Writes WAV or raw audio array to the audio pipeline.
- [ ] Cold/warm latency metrics are recorded.

## References
- Kokoro KPipeline usage: https://huggingface.co/hexgrad/Kokoro-82M
