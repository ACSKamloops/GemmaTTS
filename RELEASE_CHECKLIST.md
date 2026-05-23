# GemmaTTS — Release Checklist

Use this checklist before every tagged release to ensure quality and consistency.

---

## Pre-Release

### 1. Version Bump

- [ ] Update version string in `app/__init__.py` (if present) or `pyproject.toml`
- [ ] Update version in `README.md` header/badges (if applicable)
- [ ] Update `SETUP.md` if any setup steps have changed
- [ ] Update `ARCHITECTURE.md` if the service topology or model list has changed
- [ ] Ensure `requirements.txt` pins are correct and up-to-date
- [ ] Run `pip freeze > requirements-lock.txt` to capture exact versions (optional but recommended)

### 2. Code Quality

- [ ] All linting passes: `ruff check app/ tests/`
- [ ] No `print()` statements in production code (use `logging` module)
- [ ] Type hints present on all public functions
- [ ] No hardcoded secrets, tokens, or credentials in source
- [ ] `.env.example` is up-to-date with all required environment variables

### 3. Test Requirements

- [ ] **Unit tests pass**: `python -m pytest tests/ -v`
- [ ] **Async tests pass**: `python -m pytest tests/ -v -k async`
- [ ] **Smoke test passes**: `python scripts/smoke_test.py`
- [ ] **All TTS engines tested** (at least one synthesis per engine):
  - [ ] Chatterbox
  - [ ] Dia 1.6B
  - [ ] Kokoro 82M
  - [ ] Piper (fallback)
  - [ ] F5-TTS (if available)
- [ ] **Audio pipeline verified**: Output audio meets quality thresholds
  - [ ] Sample rate: 24 kHz
  - [ ] LUFS: -23 ±1 dB
  - [ ] No clipping artifacts
- [ ] **Security tests**:
  - [ ] Input sanitization blocks HTML/URLs/traversal
  - [ ] HMAC-signed URLs expire correctly
  - [ ] Path traversal protection verified

### 4. Model Compatibility Checks

- [ ] **Gemma 4 E4B-it** loads and generates text correctly
  - [ ] Model ID: `google/gemma-4-E4B-it`
  - [ ] BF16 precision on CUDA
  - [ ] `enable_thinking=False` confirmed
  - [ ] Generation produces coherent output within safety bounds
- [ ] **Chatterbox** auto-downloads weights and synthesizes
- [ ] **Dia 1.6B** loads via transformers, `[S1]`/`[S2]` tags work
- [ ] **Kokoro 82M** ONNX model loads, voice selection works
- [ ] **Piper** loads fallback voice, produces audio
- [ ] **Model download script** runs cleanly:
  ```bash
  python scripts/download_models.py --test-only
  ```
- [ ] All model versions match those documented in `ARCHITECTURE.md`

### 5. Docker Validation

- [ ] `docker compose build` succeeds with no errors
- [ ] `docker compose up -d` starts all three services
- [ ] All health checks pass:
  ```bash
  curl http://localhost:8000/health
  curl http://localhost:8001/health
  curl http://localhost:8002/health
  ```
- [ ] GPU passthrough works inside containers (`nvidia-smi` in container)
- [ ] Model volume mount is functional (models persist across restarts)
- [ ] `docker compose down && docker compose up -d` — services recover cleanly
- [ ] Image size is reasonable (document in release notes)

### 6. Documentation Review

- [ ] `README.md` — Quick Start instructions are accurate
- [ ] `SETUP.md` — All three setup paths verified (Ubuntu, WSL2, Docker)
- [ ] `ARCHITECTURE.md` — Model details, ports, and pipeline steps are current
- [ ] API endpoints documented with request/response examples
- [ ] Environment variables table is complete and accurate
- [ ] Troubleshooting section covers known issues
- [ ] CHANGELOG / release notes drafted

### 7. Performance Baseline

- [ ] Record benchmarks for this release:
  - [ ] Gemma text generation latency (tokens/sec)
  - [ ] TTS synthesis latency per engine (seconds for 100 words)
  - [ ] End-to-end `/v1/dialogue` latency
  - [ ] Peak VRAM usage per engine combination
- [ ] Compare against previous release (note regressions)
- [ ] Document results in `benchmarks/` or release notes

---

## Release Process

### 8. Final Checks

- [ ] `main` / `release` branch is clean — no uncommitted changes
- [ ] All CI checks pass (if CI is configured)
- [ ] `requirements.txt` is committed and tested
- [ ] `.gitignore` excludes `models/`, `.env`, caches, and scratch files

### 9. Git Tagging

```bash
# Ensure you're on the release branch
git checkout main
git pull origin main

# Create annotated tag
git tag -a v0.X.0 -m "Release v0.X.0 — brief description"

# Push tag
git push origin v0.X.0
```

### 10. Release Artifacts

- [ ] Create GitHub Release from the tag
- [ ] Write release notes including:
  - Summary of changes
  - New features
  - Bug fixes
  - Breaking changes (if any)
  - Model version compatibility
  - Known issues
  - Upgrade instructions (if applicable)
- [ ] Attach any release artifacts (pre-built Docker images, etc.)

### 11. Post-Release

- [ ] Verify the tagged release is accessible
- [ ] Test a fresh clone + setup from the release tag:
  ```bash
  git clone --branch v0.X.0 https://github.com/YOUR_ORG/gemma4tts.git
  cd gemma4tts
  # Follow SETUP.md from scratch
  ```
- [ ] Update any deployment environments
- [ ] Announce the release (if applicable)
- [ ] Bump version in `main` to next development version (e.g., `v0.X+1.0-dev`)

---

## Hotfix Process

For critical fixes that need immediate release:

1. Branch from the release tag: `git checkout -b hotfix/v0.X.1 v0.X.0`
2. Apply the fix, run tests (steps 2–4 above, abbreviated)
3. Tag: `git tag -a v0.X.1 -m "Hotfix: brief description"`
4. Push: `git push origin hotfix/v0.X.1 v0.X.1`
5. Create GitHub Release
6. Cherry-pick the fix into `main`
