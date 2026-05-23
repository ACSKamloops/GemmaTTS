# Implement signed audio asset endpoint

## Summary
Expose generated audio through signed short-lived IDs without arbitrary file paths or arbitrary URLs.

## Acceptance criteria
- [ ] `GET /audio/{signed_id}` validates signature and expiry.
- [ ] No raw filesystem path is accepted.
- [ ] Content-Type and Content-Length are set.
- [ ] Invalid, expired, oversized, or missing assets return safe errors.
