# Implement sentence chunking for long text

## Summary
For long lines, split text into sentence chunks so the first audio chunk can be produced before the full passage is synthesized.

## Acceptance criteria
- [ ] Sentence boundaries are detected safely.
- [ ] Chunks preserve order and include chunk_index.
- [ ] API can return partial audio events.
- [ ] Normal short requests still use whole-line generation.
