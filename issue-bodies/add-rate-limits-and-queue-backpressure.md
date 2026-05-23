# Add rate limits and queue backpressure

## Summary
Prevent runaway generation and audio-cache DoS.

## Acceptance criteria
- [ ] Per-client and global limits exist.
- [ ] Queue max length is enforced.
- [ ] Old jobs can be cancelled.
- [ ] 429/503 responses are documented.
