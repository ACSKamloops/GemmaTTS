# Implement audio normalization, encoding, and cache

## Summary
Normalize generated audio, encode to target formats, hash, cache, and return metadata.

## Acceptance criteria
- [ ] Cache key includes model version, engine, voice, speed, normalized text, sample rate, codec, and encoder settings.
- [ ] Supports WAV and OGG; optional MP3/Opus behind config.
- [ ] Audio metadata includes sha256, bytes, duration_ms, sample_rate, format.
- [ ] Cache uses TTL and max disk size.
