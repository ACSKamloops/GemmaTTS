# Scaffold Python service with typed config and health endpoints

## Summary
Create the base service with environment config, structured logging, health checks, and OpenAPI docs.

## Acceptance criteria
- [ ] `GET /health` returns model/service status.
- [ ] `GET /version` returns build and config metadata.
- [ ] Service starts without loading large models when configured in dry-run mode.
- [ ] Config supports local-only and LAN-serving modes.
