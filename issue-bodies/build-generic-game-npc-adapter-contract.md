# Build generic game/NPC adapter contract

## Summary
Define a game-agnostic contract for NPC dialogue systems without binding to Arma.

## Acceptance criteria
- [ ] Input contract includes actor_id, speaker profile, listener context, max_words, and interaction_type.
- [ ] Output contract includes text, audio_id, duration, emotion/mood, and metadata.
- [ ] No engine-specific networking or file assumptions are embedded.
