# Add prompt-injection regression tests

## Summary
Add tests for direct prompt injection, hidden instruction extraction, URL injection, and format-breaking attacks.

## Acceptance criteria
- [ ] Tests prove player/user text cannot override system rules.
- [ ] Output schema validation rejects malformed responses.
- [ ] URLs and markdown are blocked or sanitized before TTS.
