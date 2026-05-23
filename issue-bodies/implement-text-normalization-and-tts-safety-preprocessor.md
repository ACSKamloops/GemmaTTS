# Implement text normalization and TTS safety preprocessor

## Summary
Normalize text before synthesis: max length, forbidden characters, URL stripping, pronunciation hints, and whitespace cleanup.

## Acceptance criteria
- [ ] Max characters and max words enforced.
- [ ] URLs/markdown/control characters removed or replaced.
- [ ] Configurable replacement dictionary exists.
- [ ] TTS input is logged safely without secrets or private context.
