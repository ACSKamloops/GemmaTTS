# Implement Piper persistent fallback worker

## Summary
Add Piper as a fallback through persistent server/process mode rather than per-line CLI invocation.

## Acceptance criteria
- [ ] Piper fallback can be enabled/disabled in config.
- [ ] Fallback triggers on Kokoro unavailable, timeout, or explicit voice selection.
- [ ] Piper model path and voice metadata are configurable.
- [ ] Per-line CLI model reload is not used.

## References
- Piper CLI docs recommending web server for repeated use: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md
