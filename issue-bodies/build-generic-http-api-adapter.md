# Build generic HTTP API adapter

## Summary
Expose stable REST endpoints for external projects.

## Acceptance criteria
- [ ] `POST /v1/dialogue` creates a full LLM+TTS job.
- [ ] `POST /v1/tts` synthesizes explicit text.
- [ ] `GET /v1/jobs/{id}` returns job status.
- [ ] `GET /v1/voices` returns voice registry.
