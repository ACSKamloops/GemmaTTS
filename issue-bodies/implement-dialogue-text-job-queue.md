# Implement dialogue/text job queue

## Summary
Create durable or in-memory job records for LLM/TTS requests.

## Acceptance criteria
- [ ] Jobs have states: queued, generating_text, synthesizing, encoding, ready, failed, cancelled.
- [ ] Every job has request_id, timestamps, metrics, and error field.
- [ ] Concurrency limits are separately configurable for LLM, TTS, and encoding.
