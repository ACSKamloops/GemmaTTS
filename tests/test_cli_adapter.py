"""Tests for app.adapters.cli module."""

import json
import pytest
from unittest.mock import patch, MagicMock
import base64
import io
import wave
import struct

from app.adapters.cli import build_parser, main, _read_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav_b64(duration: float = 0.1, sample_rate: int = 24000) -> str:
    """Return a base64-encoded minimal WAV for testing."""
    num_samples = int(duration * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        samples = struct.pack(f"<{num_samples}h", *([1000] * num_samples))
        w.writeframes(samples)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_generate_command(self):
        parser = build_parser()
        args = parser.parse_args(["generate", "-t", "hello"])
        assert args.command == "generate"
        assert args.text == "hello"
        assert args.max_words == 150

    def test_speak_command_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["speak", "-t", "hello world"])
        assert args.command == "speak"
        assert args.text == "hello world"
        assert args.voice == "default"
        assert args.engine is None
        assert args.output is None

    def test_speak_command_full_options(self):
        parser = build_parser()
        args = parser.parse_args([
            "speak", "-t", "hello", "-v", "narrator",
            "-e", "piper", "-o", "out.wav",
        ])
        assert args.voice == "narrator"
        assert args.engine == "piper"
        assert args.output == "out.wav"

    def test_dialogue_command(self):
        parser = build_parser()
        args = parser.parse_args([
            "dialogue", "-t", "greet the hero",
            "-v", "bard", "-f", "ogg", "-o", "scene.ogg",
        ])
        assert args.command == "dialogue"
        assert args.format == "ogg"
        assert args.voice == "bard"

    def test_voices_command(self):
        parser = build_parser()
        args = parser.parse_args(["voices"])
        assert args.command == "voices"

    def test_health_command(self):
        parser = build_parser()
        args = parser.parse_args(["health"])
        assert args.command == "health"

    def test_global_host_option(self):
        parser = build_parser()
        args = parser.parse_args(["--host", "0.0.0.0", "health"])
        assert args.host == "0.0.0.0"

    def test_no_test_mode_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--no-test-mode", "health"])
        assert args.test_mode is False

    def test_missing_command_errors(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# _read_text tests
# ---------------------------------------------------------------------------


class TestReadText:
    def test_returns_plain_string(self):
        assert _read_text("hello world") == "hello world"

    def test_reads_stdin_on_dash(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("piped input\n"))
        assert _read_text("-") == "piped input"

    def test_exits_on_empty_stdin(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
        with pytest.raises(SystemExit):
            _read_text("-")


# ---------------------------------------------------------------------------
# Command function tests (mock httpx)
# ---------------------------------------------------------------------------


class TestCmdGenerate:
    @patch("app.adapters.cli.httpx.Client")
    def test_generate_success(self, mock_client_cls, capsys):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "text": "Generated text",
            "generation_time_ms": 42.0,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_resp)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        main(["generate", "-t", "test prompt"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["text"] == "Generated text"


class TestCmdSpeak:
    @patch("app.adapters.cli.httpx.Client")
    def test_speak_no_output(self, mock_client_cls, capsys):
        wav_b64 = _make_wav_b64()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "audio_bytes_base64": wav_b64,
            "format": "wav",
            "sample_rate": 24000,
            "synthesis_time_ms": 10.0,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_resp)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        main(["speak", "-t", "hello"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "hint" in data
        assert data["format"] == "wav"

    @patch("app.adapters.cli.httpx.Client")
    def test_speak_with_output(self, mock_client_cls, capsys, tmp_path):
        wav_b64 = _make_wav_b64()
        out_file = tmp_path / "output.wav"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "audio_bytes_base64": wav_b64,
            "format": "wav",
            "sample_rate": 24000,
            "synthesis_time_ms": 10.0,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock(post=MagicMock(return_value=mock_resp)))
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        main(["speak", "-t", "hello", "-o", str(out_file)])
        assert out_file.exists()
        assert out_file.stat().st_size > 0


class TestCmdVoices:
    def test_voices_output(self, capsys):
        main(["voices"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "engines" in data
        assert len(data["engines"]) == 6
        engine_names = {e["engine"] for e in data["engines"]}
        assert "kokoro" in engine_names
        assert "piper" in engine_names


class TestCmdHealth:
    @patch("app.adapters.cli.httpx.Client")
    def test_health_all_unreachable(self, mock_client_cls, capsys):
        # Simulate connect errors
        mock_client_instance = MagicMock()
        mock_client_instance.get.side_effect = Exception("connect error")
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client_instance)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        main(["health"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "orchestrator" in data
        assert "gemma" in data
        assert "tts" in data
        # All should report some error/status
        for svc in data.values():
            assert "status" in svc
