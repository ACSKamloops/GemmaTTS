# Implement PCM streaming endpoint

## Summary
Expose low-latency PCM streaming for desktop assistants and live clients.

## Acceptance criteria
- [ ] Streaming endpoint returns PCM chunks.
- [ ] Client can start playback before the whole line is encoded.
- [ ] Fallback to file-based output remains available.
- [ ] Metrics include time_to_first_audio_chunk.
