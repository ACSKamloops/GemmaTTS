# Implement API auth modes

## Summary
Support local-open mode, token mode, and signed audio download mode.

## Acceptance criteria
- [ ] Local-only mode binds to 127.0.0.1 by default.
- [ ] Token mode protects mutation endpoints.
- [ ] Audio endpoint accepts signed IDs only.
- [ ] CORS is disabled by default unless configured.
