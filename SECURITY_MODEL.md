# GemmaTTS Security Model

> **Version**: 1.0  
> **Last updated**: 2026-05-23  
> **Scope**: All services — Orchestrator API (`:8000`), Gemma LLM Service (`:8001`), TTS Service (`:8002`)

---

## Table of Contents

1. [Threat Model](#1-threat-model)
2. [Input Validation](#2-input-validation)
3. [Output Validation](#3-output-validation)
4. [Filesystem Security](#4-filesystem-security)
5. [Audio Asset Security](#5-audio-asset-security)
6. [Rate Limiting](#6-rate-limiting)
7. [Network Security](#7-network-security)
8. [Privacy Considerations](#8-privacy-considerations)
9. [Configuration Security](#9-configuration-security)
10. [Known Limitations](#10-known-limitations)
11. [Test Coverage Summary](#11-test-coverage-summary)

---

## 1. Threat Model

GemmaTTS is a local-first, multi-service TTS pipeline designed for game dialogue. The threat model assumes:

### What We Protect Against

| Threat Category          | Attack Vector                            | Severity | Mitigation Layer                     |
|--------------------------|------------------------------------------|----------|--------------------------------------|
| **Prompt Injection**     | Malicious URLs, HTML, or markdown in user text | High | Text sanitizer (`text_sanitizer.py`) |
| **Path Traversal**       | `../../etc/passwd` in cache keys or format params | Critical | Cache path validation (`cache.py`)   |
| **Symlink Jailbreak**    | Symlinks in cache dir pointing to sensitive files | Critical | `is_safe_path()` + symlink rejection |
| **Unauthorized File Access** | Forged or expired audio download URLs    | High     | HMAC-SHA256 signed URLs (`signer.py`) |
| **LLM Output Abuse**     | Unstructured or malformed LLM JSON responses | Medium | Schema-constrained validation (`output_validator.py`) |
| **Denial of Service**    | Request flooding across any service       | High     | Thread-safe sliding-window rate limiter |
| **Resource Exhaustion**  | Oversized audio files filling disk        | Medium   | Per-file and total cache size caps    |
| **Network Exposure**     | Remote access to internal services        | High     | Default bind to `127.0.0.1`          |
| **Data Exfiltration**    | Text/audio sent to cloud APIs             | Medium   | All models run locally; no cloud deps |
| **Timing Attacks**       | Signature comparison leaking valid tokens | Low      | `hmac.compare_digest()` constant-time comparison |

### Trust Boundaries

```
┌────────────────────────────────────────────────────────────────┐
│  UNTRUSTED: User input (text, speaker config, output format)  │
├────────────────────────────────────────────────────────────────┤
│  SEMI-TRUSTED: LLM output (may contain malformed JSON,       │
│                unstructured text, <think> blocks)              │
├────────────────────────────────────────────────────────────────┤
│  TRUSTED: Internal service-to-service calls (localhost only), │
│           filesystem (cache directory), configuration          │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Input Validation

**Module**: [`app/safety/text_sanitizer.py`](app/safety/text_sanitizer.py)

All user-supplied text passes through `sanitize_text()` before reaching the LLM or any downstream service. The sanitizer applies the following transformations in order:

### 2.1 URL Stripping

```
Step 1: Markdown links → Preserve link text, strip URL
        [Google](https://google.com) → Google

Step 2: Raw URLs → Complete removal
        Visit http://evil.com → Visit
```

- **Markdown link regex**: `\[(.*?)\]\((?:https?|ftp|file)://[^\s()\[\]{}]+\)` — extracts readable text from `[text](url)` patterns
- **Raw URL regex**: `(?:https?|ftp|file)://[^\s()\[\]{}]+` — catches bare `http://`, `https://`, `ftp://`, and `file://` URLs
- Covers single-character hostnames (e.g., `http://a`) without leaving dangling parentheses

### 2.2 HTML Tag Removal

```
<script>alert('xss')</script> → alert('xss')
<b>bold</b>                   → bold
```

- Regex: `<[^>]*>` — strips all angle-bracket delimited tags

### 2.3 Path Traversal Removal

```
../../etc/passwd → etc/passwd
path/../file     → path/file
```

- Regex: `\.\.+[/\\]` — detects `../`, `..\\`, and multi-dot sequences like `..../`
- Applied **before** markdown character stripping to prevent backslash-based evasion

### 2.4 Markdown Character Stripping

```
**bold** → bold
`code`   → code
# Header → Header
```

- Characters removed: `* _ \` # ~ [ ] \`

### 2.5 Length Enforcement

| Limit | Default | Config Key |
|-------|---------|------------|
| Max characters | 1,000 | `settings.max_text_chars` |
| Max words | 150 | `settings.max_text_words` |

Character truncation is applied first, then word truncation. Whitespace is normalized to single spaces before limits are checked.

### 2.6 Gemma Service Prompt Limits

The Gemma service (`gemma_service.py`) applies an additional input length gate:

- Prompts **≤ 5,000 characters**: passed through unmodified
- Prompts **> 5,000 characters**: hard-truncated to 1,000 characters (`MAX_FALLBACK_CHARS`)

Pydantic validators additionally enforce:
- `prompt` cannot be empty or whitespace-only
- `max_words` must be a non-negative integer (booleans and floats rejected)

---

## 3. Output Validation

**Module**: [`app/safety/output_validator.py`](app/safety/output_validator.py)

LLM output is inherently unpredictable. The output validator constrains responses to a known schema:

### 3.1 JSON Extraction

The validator handles common LLM output quirks:

1. **Markdown-wrapped JSON**: Extracts JSON from `` ```json {...} ``` `` blocks
2. **Conversational chatter**: Falls back to finding the first `{` and last `}` to isolate JSON from surrounding prose
3. **Strict parsing**: Uses `json.loads()` — no permissive/partial JSON parsing

### 3.2 Schema Enforcement

```python
class DialogueResponseSchema(BaseModel):
    text: str
```

- Validated via Pydantic `BaseModel`
- Response must contain at minimum a `text` field of type `str`
- Extra fields are silently accepted (forward-compatible)
- Invalid JSON or missing `text` field returns `None`, triggering fallback dialogue

### 3.3 Fallback Behavior

When validation fails, the orchestrator uses a hardcoded fallback string (`"Fallback dialogue text due to schema mismatch."`) rather than propagating raw LLM output to TTS. This prevents:
- Injected control sequences reaching the audio pipeline
- Arbitrary text being synthesized as speech

### 3.4 Think Block Stripping

Gemma's `<think>...</think>` reasoning blocks are stripped from output when `enable_thinking=False` (the default):

```python
# Closed tags
re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
# Unclosed tags (truncated thinking)
re.sub(r'<think>.*', '', text, flags=re.DOTALL)
```

---

## 4. Filesystem Security

**Module**: [`app/audio/cache.py`](app/audio/cache.py)

The audio cache is the primary filesystem attack surface. Multiple layers of defense prevent unauthorized file access:

### 4.1 Path Traversal Prevention

Three-stage defense in `get_file_path()`:

```
Stage 1: Reject inputs containing /, \, or .. characters
         → Raises PermissionError immediately

Stage 2: Sanitize key and format to alphanumeric + hyphen/underscore
         safe_key = [c for c in key if c.isalnum() or c in ("-", "_")]
         safe_format = [c for c in format if c.isalnum()]

Stage 3: Resolve the final path and verify it's inside the cache directory
         → is_safe_path(resolved_path, cache_dir)
```

### 4.2 Safe Path Validation

```python
def is_safe_path(path: Path, base_dir: Path) -> bool:
```

- Resolves **both** the target path and the base directory to absolute paths (following symlinks)
- Verifies that `base_dir` is a parent of the resolved path
- Handles both existing files (uses `path.resolve()`) and non-existent files (uses `os.path.abspath()`)
- Returns `False` on any `ValueError` or `OSError` (fail-closed)

### 4.3 Symlink Rejection

Cache reads perform an explicit symlink check:

```python
if path.is_symlink():
    real_path = path.resolve()
    if not is_safe_path(real_path, self.cache_dir):
        raise PermissionError("Symlink targets outside cache directory.")
```

The audio download endpoint (`GET /audio/{signed_id}`) applies the same check before serving any file.

### 4.4 Cache Size Enforcement

| Limit | Default | Config Key |
|-------|---------|------------|
| Max per-file size | 5 MB | `settings.max_file_size_bytes` |
| Max total cache size | 50 MB | `settings.max_cache_size_bytes` |

- Files exceeding `max_file_size_bytes` are rejected with `ValueError` before any write
- Files exceeding `max_cache_size_bytes` are rejected before any write
- LRU pruning removes oldest files when total cache size would be exceeded by new entries
- Cache pruning skips symlinks and `.json` metadata files
- Negative `max_cache_size_bytes` values cause `put()` to raise `ValueError`

### 4.5 Audio Download Endpoint Hardening

The `GET /audio/{signed_id}` endpoint (orchestrator) applies additional protections:

1. **URL decoding**: `urllib.parse.unquote()` before any checks (prevents `%2e%2e%2f` bypass)
2. **Direct traversal rejection**: Rejects any input containing `..`, `/`, or `\` characters
3. **Signature verification**: HMAC-SHA256 (see §5) before any filesystem access
4. **Symlink re-check**: Even after signature verification, symlinks to paths outside the cache are rejected

---

## 5. Audio Asset Security

**Module**: [`app/audio/signer.py`](app/audio/signer.py)

Audio files are never served directly. Every audio download requires a cryptographically signed, time-limited token.

### 5.1 Token Format

```
{audio_id}.{expiry_timestamp}.{hmac_sha256_signature}
```

Example:
```
abc123_ogg.1748012345.a1b2c3d4e5f6...
```

### 5.2 Signing Process

```python
message = f"{audio_id}.{expiry_time}"
signature = hmac.new(
    secret_key.encode("utf-8"),
    message.encode("utf-8"),
    hashlib.sha256
).hexdigest()
```

- **Algorithm**: HMAC-SHA256
- **Key**: `settings.secret_key` — 64-character hex string by default (256 bits of entropy)
- **Expiry**: Configurable via `settings.signed_url_expiry_seconds` (default: 300 seconds / 5 minutes)

### 5.3 Verification

`verify_signed_audio_id()` performs:

1. **Format validation**: Token must split into exactly 3 parts using `rsplit(".", 2)` — supports audio IDs containing dots
2. **Expiry check**: `time.time() > expiry_time` → reject expired tokens
3. **Signature comparison**: Uses `hmac.compare_digest()` for **constant-time** comparison, preventing timing side-channel attacks
4. **Fail-closed**: Returns `None` on any error (parse failure, expired, invalid signature)

### 5.4 Key Rotation

The debug endpoint `POST /debug/rotate_key` allows runtime key rotation with enforcement:

- **Minimum key length**: 32 characters (keys shorter than 32 chars are rejected with HTTP 400)
- **Empty keys rejected**: Empty strings return HTTP 400
- Existing signed tokens become invalid immediately after rotation (by design)

---

## 6. Rate Limiting

**Module**: Implemented in both [`gemma_service.py`](app/services/gemma_service.py) and [`tts_service.py`](app/services/tts_service.py)

### 6.1 Implementation

```python
class ThreadSafeRateLimiter:
    def __init__(self, limit: int = 40, window_seconds: float = 1.0):
```

- **Algorithm**: Sliding-window counter using a `collections.deque`
- **Thread safety**: `threading.Lock` protects all timestamp mutations
- **Window**: 1.0 second sliding window
- **Limit**: 40 requests per second per service

### 6.2 Enforcement Points

| Service | Endpoint | Limit | Response on Exceed |
|---------|----------|-------|--------------------|
| Gemma LLM | `POST /generate` | 40 req/s | HTTP 429 `Too Many Requests` |
| TTS | `POST /synthesize` | 40 req/s | HTTP 429 `Too Many Requests` |

The Orchestrator API does not have its own rate limiter but is inherently gated by the downstream service limits.

### 6.3 Behavior Under Load

- Expired timestamps are lazily pruned on each `is_allowed()` call
- The deque grows to at most `limit` entries before rejecting
- No queueing — excess requests are immediately rejected (fail-fast)

---

## 7. Network Security

### 7.1 Default Binding

```python
# app/config.py
host: str = "127.0.0.1"
port: int = 8000
```

All three services default to binding on `127.0.0.1` (localhost only). This means:

- **No remote access** without explicit reconfiguration
- Service-to-service calls use `http://127.0.0.1:{port}` hardcoded URLs
- The Orchestrator calls Gemma at `127.0.0.1:8001` and TTS at `127.0.0.1:8002`

### 7.2 Inter-Service Communication

```
Orchestrator (:8000)  →  Gemma (:8001)   via HTTP POST (localhost)
Orchestrator (:8000)  →  TTS   (:8002)   via HTTP POST (localhost)
```

- All inter-service communication is **unencrypted HTTP** over loopback
- This is acceptable because traffic never leaves the host machine
- Timeout of **5 seconds** on all inter-service HTTP calls prevents hung connections

### 7.3 No TLS

TLS is not implemented. This is intentional for a local-only deployment. If the system were exposed to a network, a reverse proxy (nginx, Caddy) should terminate TLS in front of the Orchestrator.

---

## 8. Privacy Considerations

### 8.1 Local-Only Processing

GemmaTTS is designed for **complete data locality**:

| Component | Cloud Dependency? | Notes |
|-----------|-------------------|-------|
| Gemma LLM | ❌ None | Model weights downloaded once from HuggingFace, then run locally |
| Chatterbox TTS | ❌ None | Weights auto-downloaded by pip package, inference is local |
| Dia TTS | ❌ None | Local transformer inference |
| Kokoro TTS | ❌ None | ONNX inference, fully offline |
| Piper TTS | ❌ None | Local ONNX/C++ inference |
| Fish Audio TTS | ⚠️ Optional | Requires explicit opt-in via `ENABLE_FISH_AUDIO=true` env var |
| Audio caching | ❌ None | Stored on local filesystem only |

### 8.2 Fish Audio Consent Gate

Fish Audio is the only engine that may interact with external services. It is protected by an explicit consent check:

```python
if req.engine == "fish":
    enable_fish = os.environ.get("ENABLE_FISH_AUDIO", "false").lower() == "true"
    if not enable_fish:
        raise HTTPException(status_code=403, detail="Fish Audio engine requires explicit consent.")
```

### 8.3 No Telemetry

- No analytics, crash reporting, or usage telemetry is collected
- No data is transmitted to any remote server during normal operation
- Model weights are the only external downloads (one-time, at setup)

### 8.4 Cache Privacy

- Audio cache files are stored as SHA-256 hashes of `text:voice_id:format` — the original text is **not** stored in filenames
- Cache directory defaults to `public/data/audio_cache/` with configurable location
- Cache contents should be treated as sensitive (they contain synthesized audio of user-provided text)

---

## 9. Configuration Security

**Module**: [`app/config.py`](app/config.py)

### 9.1 Secret Key Management

```python
secret_key: str = Field(
    default_factory=lambda: os.getenv("SECRET_KEY", secrets.token_hex(32))
)
```

- **Production**: Set `SECRET_KEY` environment variable or add to `.env` file
- **Development**: Auto-generated 64-character hex string (256 bits) via `secrets.token_hex(32)`
- **Caution**: Auto-generated keys change on each restart, invalidating all signed URLs. Set a stable key for persistent deployments.

### 9.2 Configuration Sources

Settings are loaded via Pydantic's `BaseSettings` with `SettingsConfigDict`:

1. **`.env` file** (lowest priority)
2. **Environment variables** (highest priority, overrides `.env`)

### 9.3 Sensitive Defaults

| Setting | Default | Security Rationale |
|---------|---------|-------------------|
| `host` | `127.0.0.1` | Prevents remote access |
| `signed_url_expiry_seconds` | `300` (5 min) | Limits replay window |
| `max_text_chars` | `1000` | Prevents prompt bombing |
| `max_text_words` | `150` | Bounds LLM output length |
| `max_cache_size_bytes` | `50 MB` | Prevents disk exhaustion |
| `max_file_size_bytes` | `5 MB` | Caps individual file size |

---

## 10. Known Limitations

### 10.1 Input Validation Gaps

| Issue | Description | Risk |
|-------|-------------|------|
| **Trailing traversal** | `path/..` (without trailing slash) is **not** stripped by `TRAVERSAL_REGEX` | Low — cache key sanitization and `is_safe_path()` prevent exploitation |
| **Empty LLM text** | `validate_llm_json` accepts `{"text": ""}` and `{"text": "   "}` | Low — may cause TTS to synthesize silence or error |
| **Unicode normalization** | No Unicode normalization (e.g., homoglyph attacks, RTL overrides) | Low — local-only system reduces attack surface |

### 10.2 Architectural Limitations

| Issue | Description | Risk |
|-------|-------------|------|
| **No authentication** | No API keys, OAuth, or user authentication on any endpoint | Medium — mitigated by localhost binding |
| **No TLS** | All traffic is plaintext HTTP | Low — localhost-only traffic |
| **Debug endpoints in production** | `/debug/rotate_key` and `/debug/update_settings` are always exposed | Medium — should be disabled or protected in production |
| **No request signing** | Inter-service calls are not authenticated | Low — localhost-only, all services on same host |
| **Mutable settings at runtime** | `settings` object can be mutated by debug endpoints and tests | Medium — no persistence, resets on restart |

### 10.3 Denial of Service Surface

| Issue | Description | Risk |
|-------|-------------|------|
| **Per-service rate limiting** | Rate limits are per-service, not per-client IP | Medium — a single abusive client can exhaust the rate limit for all users |
| **LLM inference latency** | A crafted prompt near the 5,000 character limit can cause long inference times | Low — `max_new_tokens` is bounded |
| **No request size limits** | FastAPI default body size applies; no custom per-endpoint caps | Low — Pydantic validation rejects most malformed payloads |

---

## 11. Test Coverage Summary

### 11.1 Test Files

| Test File | Focus Area | Key Tests |
|-----------|------------|-----------|
| [`test_safety.py`](tests/test_safety.py) | Text sanitizer + output validator | URL stripping, HTML removal, markdown stripping, JSON schema validation |
| [`test_signer.py`](tests/test_signer.py) | HMAC signing + verification | Sign/verify round-trip, expiry, tampered tokens |
| [`test_filesystem.py`](tests/test_filesystem.py) | Cache path safety | Path traversal, symlink rejection, `is_safe_path()`, file size limits, cache pruning |
| [`test_adversarial_1.py`](tests/test_adversarial_1.py) | Adversarial edge cases (batch 1) | Dotted audio IDs, boundary inputs, empty keys, negative settings, truncation limits |
| [`test_adversarial_2.py`](tests/test_adversarial_2.py) | Adversarial edge cases (batch 2) | Sanitizer bypass, token split DoS, cache size bypass, key rotation, duration metadata, validation error propagation |
| [`test_gemma_service.py`](tests/test_gemma_service.py) | Gemma LLM service | Prompt validation, rate limiting, test mode, think-block stripping |
| [`test_tts_service.py`](tests/test_tts_service.py) | TTS service | Engine routing, Fish Audio consent, dummy WAV generation |
| [`test_orchestrator_api.py`](tests/test_orchestrator_api.py) | Orchestrator gateway | End-to-end dialogue flow, cache hits, format validation, signed URL serving |
| [`test_e2e_integration.py`](tests/test_e2e_integration.py) | Full pipeline integration | Multi-service flow, fallback behavior, client disconnect |
| [`test_audio_quality.py`](tests/test_audio_quality.py) | Audio pipeline | WAV encoding, format conversion, quality metrics |

### 11.2 Security-Specific Test Cases

#### Path Traversal & Filesystem Abuse
- `test_path_traversal_prevention` — `../../etc/passwd` in cache keys → `PermissionError`
- `test_symlink_prevention` — symlinks to external files → `PermissionError`
- `test_is_safe_path` — resolved path verification against base directory
- `test_max_file_size` — oversized data rejection
- `test_cache_pruning` — LRU eviction under size pressure

#### Signed URL Security
- `test_signer_with_dots` — audio IDs containing dots (e.g., `my.audio.file`) parse correctly via `rsplit(".", 2)`
- `test_signer_boundary_inputs` — empty tokens, too few/many parts, non-integer timestamps, huge timestamps
- `test_token_split_dos` — 100,000-dot payload does not crash the verifier
- `test_empty_key_rotation_and_forgery` — empty and short keys rejected at `/debug/rotate_key`

#### Input Sanitization
- `test_sanitizer_single_char_host_url` — URLs with 1-char hostnames (`http://a`) fully stripped
- `test_link_sanitizer_bypass` — markdown links with short hostnames do not leave artifacts
- `test_sanitizer_trailing_traversal` — trailing `..` without slash is documented but not exploitable

#### LLM Output Validation
- `test_output_validator_empty_fields` — empty-string and whitespace-only `text` fields accepted (known limitation)
- `test_swallowed_validation_errors` — 422 errors from Gemma correctly propagated through orchestrator

#### Rate Limiting & DoS
- `test_gemma_truncation_limits` — prompts > 5,000 chars truncated to 1,000
- `test_gemma_huge_max_words` — extremely large `max_words` values handled without overflow
- `test_cache_negative_settings` — negative `max_cache_size_bytes` raises `ValueError`
- `test_cache_prune_incoming_greater_than_max` — files exceeding cache limit rejected

#### Consent & Access Control
- `test_piper_dependency_discrepancy` — missing engine dependencies return 503 (not 500)
- Fish Audio consent gate tested in `test_tts_service.py`

### 11.3 Test Configuration

The test suite (`conftest.py`) overrides production settings with safe test values:

```python
settings.audio_cache_dir = Path("tests/test_audio_cache").resolve()
settings.max_cache_size_bytes = 1024 * 1024       # 1 MB
settings.max_file_size_bytes = 100 * 1024          # 100 KB
settings.secret_key = "test-secret-key-for-hmac-verification-operations"
```

The test cache directory is created before tests and removed after the session completes.

---

## Appendix: Security Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         User / Game Client          │
                    │     (untrusted input boundary)      │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │      Orchestrator API (:8000)        │
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │   Text Sanitizer             │    │
                    │  │   • URL stripping            │    │
                    │  │   • HTML removal             │    │
                    │  │   • Traversal prevention     │    │
                    │  │   • Markdown stripping       │    │
                    │  │   • Length enforcement        │    │
                    │  └──────────────┬──────────────┘    │
                    │                 │                    │
                    │  ┌──────────────▼──────────────┐    │
                    │  │   Audio Cache Manager        │    │
                    │  │   • Path traversal defense   │    │
                    │  │   • Symlink rejection        │    │
                    │  │   • Size enforcement         │    │
                    │  │   • SHA-256 cache keys       │    │
                    │  └──────────────┬──────────────┘    │
                    │                 │                    │
                    │  ┌──────────────▼──────────────┐    │
                    │  │   HMAC Signer                │    │
                    │  │   • SHA-256 signed tokens    │    │
                    │  │   • Time-limited expiry      │    │
                    │  │   • Constant-time verify     │    │
                    │  └─────────────────────────────┘    │
                    └──────┬──────────────────┬───────────┘
                           │                  │
              ┌────────────▼───┐    ┌────────▼──────────┐
              │ Gemma LLM      │    │ TTS Service       │
              │ (:8001)        │    │ (:8002)           │
              │                │    │                   │
              │ • Rate limiter │    │ • Rate limiter    │
              │ • Prompt size  │    │ • Engine consent  │
              │   limits       │    │   (Fish Audio)    │
              │ • Pydantic     │    │ • Pydantic        │
              │   validation   │    │   validation      │
              │ • Think-block  │    │ • Lazy worker     │
              │   stripping    │    │   loading         │
              └────────────────┘    └───────────────────┘
```
