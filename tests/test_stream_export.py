"""Tests for the /synthesize/stream and /synthesize/export endpoints."""
import io
import struct
import wave
import pytest
from fastapi.testclient import TestClient
from app.services.tts_service import app, rate_limiter

client = TestClient(app)


# ------------------------------------------------------------------ helpers
def _clear_rate_limiter():
    rate_limiter.timestamps.clear()


# ================================================================== STREAM
class TestSynthesizeStream:
    """POST /synthesize/stream – PCM streaming endpoint."""

    def setup_method(self):
        _clear_rate_limiter()

    # --- happy path ---
    @pytest.mark.parametrize("engine", ["chatterbox", "kokoro", "piper"])
    def test_stream_returns_pcm_audio(self, engine):
        payload = {
            "text": "Hello streaming world",
            "engine": engine,
            "test_mode": True,
        }
        resp = client.post("/synthesize/stream", json=payload)
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/pcm"
        assert resp.headers["X-Sample-Rate"] == "24000"
        assert resp.headers["X-Channels"] == "1"
        assert resp.headers["X-Bit-Depth"] == "16"
        assert "X-Time-To-First-Chunk-Ms" in resp.headers

        # Body must be non-empty raw PCM (even number of bytes for s16le)
        pcm = resp.content
        assert len(pcm) > 0
        assert len(pcm) % 2 == 0

    def test_stream_pcm_matches_wav_payload(self):
        """PCM body should equal the raw frames from the generated WAV."""
        payload = {
            "text": "PCM fidelity check duration_sec=0.1",
            "engine": "chatterbox",
            "test_mode": True,
        }
        resp = client.post("/synthesize/stream", json=payload)
        assert resp.status_code == 200
        pcm_body = resp.content

        # Also synthesise the same text via the regular endpoint and
        # extract PCM from the WAV for comparison.
        import base64
        resp2 = client.post("/synthesize", json=payload)
        wav_bytes = base64.b64decode(resp2.json()["audio_bytes_base64"])
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            expected_pcm = wf.readframes(wf.getnframes())

        assert pcm_body == expected_pcm

    def test_stream_time_to_first_chunk_reasonable(self):
        payload = {
            "text": "Timing test",
            "engine": "chatterbox",
            "test_mode": True,
        }
        resp = client.post("/synthesize/stream", json=payload)
        ttfc = float(resp.headers["X-Time-To-First-Chunk-Ms"])
        # In test mode, synthesis is near-instant; just verify it's positive
        assert ttfc >= 0.0

    # --- validation ---
    def test_stream_empty_text_rejected(self):
        resp = client.post(
            "/synthesize/stream",
            json={"text": "", "engine": "chatterbox", "test_mode": True},
        )
        assert resp.status_code == 422

    def test_stream_invalid_engine_rejected(self):
        resp = client.post(
            "/synthesize/stream",
            json={"text": "Hello", "engine": "nonexistent", "test_mode": True},
        )
        assert resp.status_code == 422

    # --- consent / offline ---
    def test_stream_fish_consent_blocked(self, monkeypatch):
        monkeypatch.delenv("ENABLE_FISH_AUDIO", raising=False)
        payload = {
            "text": "Hello",
            "voice_id": "default",
            "engine": "fish",
            "test_mode": True,
        }
        resp = client.post("/synthesize/stream", json=payload)
        assert resp.status_code == 403

    def test_stream_dia_offline(self):
        payload = {
            "text": "Hello",
            "voice_id": "simulate_offline_mode",
            "engine": "dia",
            "test_mode": True,
        }
        resp = client.post("/synthesize/stream", json=payload)
        assert resp.status_code == 503


# ================================================================== EXPORT
class TestSynthesizeExport:
    """POST /synthesize/export – file download endpoint."""

    def setup_method(self):
        _clear_rate_limiter()

    # --- happy path (WAV – no ffmpeg needed) ---
    def test_export_wav_download(self):
        payload = {
            "text": "Hello export",
            "engine": "chatterbox",
            "format": "wav",
            "filename": "my-clip",
            "test_mode": True,
        }
        resp = client.post("/synthesize/export", json=payload)
        assert resp.status_code == 200

        # Content-Disposition
        cd = resp.headers["Content-Disposition"]
        assert 'attachment' in cd
        assert 'my-clip.wav' in cd

        # Metadata sidecar headers
        assert resp.headers["X-Audio-Format"] == "wav"
        assert resp.headers["X-Sample-Rate"] == "24000"
        assert resp.headers["X-Channels"] == "1"
        assert resp.headers["X-Bit-Depth"] == "16"
        assert float(resp.headers["X-Audio-Duration-Ms"]) > 0

        # Body is valid WAV
        with wave.open(io.BytesIO(resp.content), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000

    def test_export_default_filename(self):
        payload = {
            "text": "Hello",
            "engine": "chatterbox",
            "format": "wav",
            "test_mode": True,
        }
        resp = client.post("/synthesize/export", json=payload)
        assert resp.status_code == 200
        assert "tts_output.wav" in resp.headers["Content-Disposition"]

    # --- filename sanitisation ---
    @pytest.mark.parametrize(
        "raw,expected_base",
        [
            ("normal_name", "normal_name"),
            ("../../../etc/passwd", "________etc_passwd"),
            ("some file.wav", "some_file_wav"),
            ("a/b\\c", "c"),                 # path component stripped
            ("hello world", "hello_world"),
            ("good-name_123", "good-name_123"),
        ],
    )
    def test_export_filename_sanitised(self, raw, expected_base):
        payload = {
            "text": "Hello",
            "engine": "chatterbox",
            "format": "wav",
            "filename": raw,
            "test_mode": True,
        }
        resp = client.post("/synthesize/export", json=payload)
        assert resp.status_code == 200
        cd = resp.headers["Content-Disposition"]
        assert f"{expected_base}.wav" in cd

    # --- validation ---
    def test_export_empty_text_rejected(self):
        resp = client.post(
            "/synthesize/export",
            json={"text": "", "engine": "chatterbox", "format": "wav", "test_mode": True},
        )
        assert resp.status_code == 422

    def test_export_unsupported_format_rejected(self):
        resp = client.post(
            "/synthesize/export",
            json={"text": "Hello", "engine": "chatterbox", "format": "flac", "test_mode": True},
        )
        assert resp.status_code == 422

    # --- consent / offline ---
    def test_export_fish_consent_blocked(self, monkeypatch):
        monkeypatch.delenv("ENABLE_FISH_AUDIO", raising=False)
        payload = {
            "text": "Hello",
            "voice_id": "default",
            "engine": "fish",
            "format": "wav",
            "test_mode": True,
        }
        resp = client.post("/synthesize/export", json=payload)
        assert resp.status_code == 403

    def test_export_dia_offline(self):
        payload = {
            "text": "Hello",
            "voice_id": "simulate_offline_mode",
            "engine": "dia",
            "format": "wav",
            "test_mode": True,
        }
        resp = client.post("/synthesize/export", json=payload)
        assert resp.status_code == 503


# ================================================================== EXISTING
class TestExistingEndpointsStillWork:
    """Smoke-tests to confirm the original endpoints are not broken."""

    def setup_method(self):
        _clear_rate_limiter()

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_synthesize_original(self):
        payload = {
            "text": "Original endpoint check size_bytes=500",
            "engine": "chatterbox",
            "test_mode": True,
        }
        resp = client.post("/synthesize", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["format"] == "wav"
        assert data["sample_rate"] == 24000
        import base64
        assert len(base64.b64decode(data["audio_bytes_base64"])) == 500
