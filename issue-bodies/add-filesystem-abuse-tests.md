# Add filesystem abuse tests

## Summary
Protect the audio cache and export paths from traversal and symlink writes.

## Acceptance criteria
- [ ] Path traversal attempts fail.
- [ ] Symlink writes outside approved root fail.
- [ ] Max file size and max cache size are enforced.
- [ ] Export names are sanitized.
