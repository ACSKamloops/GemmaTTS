# Implement prompt builder with schema-constrained outputs

## Summary
Build prompt templates that separate system instructions from user text and force a small JSON output contract.

## Acceptance criteria
- [ ] User text is treated as data, not instruction.
- [ ] Output schema includes `text`, `mood`, `language`, and `safety_notes` or equivalent.
- [ ] Output is validated before TTS.
- [ ] Invalid model output is retried or rejected.
