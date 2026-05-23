"""
GemmaTTS CLI adapter.

Provides a command-line interface to interact with the running GemmaTTS
services (Gemma LLM on port 8001, TTS on port 8002, Orchestrator on 8000).

Usage examples:
    python -m app.adapters.cli generate -t "Tell me about dragons"
    python -m app.adapters.cli speak -t "Hello world" -e kokoro
    python -m app.adapters.cli dialogue -t "Greet the hero" -v narrator -o out.wav
    echo "Some text" | python -m app.adapters.cli speak --text -
    python -m app.adapters.cli voices
    python -m app.adapters.cli health
"""

import argparse
import base64
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import httpx

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gemmatts-cli")

# ---------------------------------------------------------------------------
# Service URL helpers
# ---------------------------------------------------------------------------

DEFAULT_LLM_PORT = 8001
DEFAULT_TTS_PORT = 8002
DEFAULT_ORCHESTRATOR_PORT = 8000


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


# ---------------------------------------------------------------------------
# Individual command implementations
# ---------------------------------------------------------------------------


def _read_text(raw: str) -> str:
    """Return *raw* as-is, unless it is ``'-'`` in which case read stdin."""
    if raw == "-":
        text = sys.stdin.read()
        if not text.strip():
            logger.error("No text received on stdin")
            sys.exit(1)
        return text.strip()
    return raw


def cmd_generate(args: argparse.Namespace) -> None:
    """Call the Gemma LLM ``/generate`` endpoint and print the result."""
    text = _read_text(args.text)
    url = f"{_base_url(args.host, args.llm_port)}/generate"

    payload = {
        "prompt": text,
        "max_words": args.max_words,
        "enable_thinking": False,
        "test_mode": args.test_mode,
    }

    logger.info("POST %s  (test_mode=%s)", url, args.test_mode)

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        _print_json(data)
    except httpx.HTTPStatusError as exc:
        _handle_http_error(exc)
    except httpx.ConnectError:
        logger.error("Cannot connect to Gemma service at %s", url)
        sys.exit(1)


def cmd_speak(args: argparse.Namespace) -> None:
    """Call the TTS ``/synthesize`` endpoint and optionally save audio."""
    text = _read_text(args.text)
    engine = args.engine or settings.default_tts_engine
    url = f"{_base_url(args.host, args.tts_port)}/synthesize"

    payload = {
        "text": text,
        "voice_id": args.voice,
        "engine": engine,
        "test_mode": args.test_mode,
    }

    logger.info(
        "POST %s  engine=%s voice=%s (test_mode=%s)",
        url, engine, args.voice, args.test_mode,
    )

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        _handle_http_error(exc)
        return
    except httpx.ConnectError:
        logger.error("Cannot connect to TTS service at %s", url)
        sys.exit(1)
        return

    audio_b64 = data.get("audio_bytes_base64")
    if not audio_b64:
        logger.error("Response did not contain audio_bytes_base64")
        sys.exit(1)

    audio_bytes = base64.b64decode(audio_b64)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_bytes)
        logger.info("Audio written to %s (%d bytes)", out_path, len(audio_bytes))
        summary = {
            "format": data.get("format", "wav"),
            "sample_rate": data.get("sample_rate"),
            "synthesis_time_ms": data.get("synthesis_time_ms"),
            "output_file": str(out_path),
            "size_bytes": len(audio_bytes),
        }
        _print_json(summary)
    else:
        # No output path – dump metadata only (raw audio to stdout is rarely
        # useful; pipe to a file with -o instead).
        summary = {
            "format": data.get("format", "wav"),
            "sample_rate": data.get("sample_rate"),
            "synthesis_time_ms": data.get("synthesis_time_ms"),
            "size_bytes": len(audio_bytes),
            "hint": "Use -o <path> to save the audio file.",
        }
        _print_json(summary)


def cmd_dialogue(args: argparse.Namespace) -> None:
    """Call the Orchestrator ``/v1/dialogue`` endpoint (full pipeline)."""
    text = _read_text(args.text)
    engine = args.engine or settings.default_tts_engine
    url = f"{_base_url(args.host, args.orch_port)}/v1/dialogue"

    payload = {
        "speaker": {
            "id": args.voice,
            "name": args.voice,
            "voice_id": args.voice,
            "style": engine,  # orchestrator infers engine from style
        },
        "user_text": text,
        "max_words": args.max_words,
        "output": {
            "audio": True,
            "format": args.format,
        },
        "test_mode": args.test_mode,
    }

    logger.info(
        "POST %s  voice=%s format=%s (test_mode=%s)",
        url, args.voice, args.format, args.test_mode,
    )

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        _handle_http_error(exc)
        return
    except httpx.ConnectError:
        logger.error("Cannot connect to Orchestrator at %s", url)
        sys.exit(1)
        return

    # If audio was produced and we have an audio_id, fetch the file.
    audio_meta = data.get("audio")
    if audio_meta and args.output:
        audio_id = audio_meta.get("audio_id", "")
        audio_url = f"{_base_url(args.host, args.orch_port)}/audio/{audio_id}"
        logger.info("GET %s", audio_url)

        try:
            with httpx.Client(timeout=30.0) as client:
                audio_resp = client.get(audio_url)
            audio_resp.raise_for_status()
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(audio_resp.content)
            logger.info(
                "Audio written to %s (%d bytes)", out_path, len(audio_resp.content),
            )
            data["output_file"] = str(out_path)
        except httpx.HTTPStatusError as exc:
            logger.warning("Could not fetch audio file: %s", exc)

    _print_json(data)


def cmd_voices(args: argparse.Namespace) -> None:
    """List available voices / engines."""
    # There is no dedicated voices endpoint; we return static metadata about
    # the supported engines and their default voice_id.
    engines = [
        {"engine": "kokoro", "default_voice": "default", "description": "Kokoro TTS engine"},
        {"engine": "piper", "default_voice": "default", "description": "Piper TTS engine"},
        {"engine": "chatterbox", "default_voice": "default", "description": "Chatterbox TTS engine"},
        {"engine": "dia", "default_voice": "default", "description": "Dia TTS engine"},
        {"engine": "fish", "default_voice": "default", "description": "Fish Audio TTS (requires consent)"},
        {"engine": "f5_tts", "default_voice": "default", "description": "F5-TTS engine"},
    ]
    _print_json({"engines": engines, "default_engine": settings.default_tts_engine})


def cmd_health(args: argparse.Namespace) -> None:
    """Check health of all services."""
    services = {
        "orchestrator": f"{_base_url(args.host, args.orch_port)}/health",
        "gemma": f"{_base_url(args.host, args.llm_port)}/health",
        "tts": f"{_base_url(args.host, args.tts_port)}/health",
    }
    results = {}
    for name, url in services.items():
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url)
            resp.raise_for_status()
            results[name] = resp.json()
        except httpx.ConnectError:
            results[name] = {"status": "unreachable"}
        except httpx.HTTPStatusError as exc:
            results[name] = {"status": "error", "code": exc.response.status_code}
        except Exception as exc:
            results[name] = {"status": "error", "detail": str(exc)}

    _print_json(results)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _print_json(data: object) -> None:
    """Pretty-print *data* as JSON to stdout."""
    print(json.dumps(data, indent=2, default=str))


def _handle_http_error(exc: httpx.HTTPStatusError) -> None:
    """Log an HTTP error and exit."""
    try:
        detail = exc.response.json()
    except Exception:
        detail = exc.response.text
    logger.error(
        "HTTP %d from %s: %s",
        exc.response.status_code,
        exc.request.url,
        detail,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemmatts",
        description="GemmaTTS command-line interface",
    )

    # ---- Global options ----
    parser.add_argument(
        "--host", default=settings.host,
        help="API host (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-port", type=int, default=DEFAULT_LLM_PORT,
        help="Gemma LLM service port (default: %(default)s)",
    )
    parser.add_argument(
        "--tts-port", type=int, default=DEFAULT_TTS_PORT,
        help="TTS service port (default: %(default)s)",
    )
    parser.add_argument(
        "--orch-port", type=int, default=DEFAULT_ORCHESTRATOR_PORT,
        help="Orchestrator gateway port (default: %(default)s)",
    )
    parser.add_argument(
        "--no-test-mode", dest="test_mode", action="store_false",
        help="Disable test mode (run real inference)",
    )
    parser.set_defaults(test_mode=True)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- generate ----
    gen_p = subparsers.add_parser("generate", help="Generate text from a prompt via Gemma LLM")
    gen_p.add_argument("-t", "--text", required=True, help="Input prompt (use '-' for stdin)")
    gen_p.add_argument("--max-words", type=int, default=150, help="Max words in response")
    gen_p.set_defaults(func=cmd_generate)

    # ---- speak ----
    speak_p = subparsers.add_parser("speak", help="Synthesize text to audio via TTS")
    speak_p.add_argument("-t", "--text", required=True, help="Text to speak (use '-' for stdin)")
    speak_p.add_argument("-v", "--voice", default="default", help="Voice ID")
    speak_p.add_argument(
        "-e", "--engine", default=None,
        choices=["kokoro", "piper", "chatterbox", "dia", "fish", "f5_tts"],
        help="TTS engine (default: from config)",
    )
    speak_p.add_argument("-o", "--output", default=None, help="Output audio file path")
    speak_p.set_defaults(func=cmd_speak)

    # ---- dialogue ----
    dlg_p = subparsers.add_parser(
        "dialogue", help="Full pipeline: prompt → LLM → TTS → audio file",
    )
    dlg_p.add_argument("-t", "--text", required=True, help="User text / prompt (use '-' for stdin)")
    dlg_p.add_argument("-v", "--voice", default="default", help="Voice / speaker ID")
    dlg_p.add_argument(
        "-e", "--engine", default=None,
        choices=["kokoro", "piper", "chatterbox", "dia", "fish", "f5_tts"],
        help="TTS engine (default: from config)",
    )
    dlg_p.add_argument("-o", "--output", default=None, help="Output audio file path")
    dlg_p.add_argument(
        "-f", "--format", default="wav",
        choices=["wav", "ogg", "mp3"],
        help="Audio format (default: %(default)s)",
    )
    dlg_p.add_argument("--max-words", type=int, default=150, help="Max words in LLM response")
    dlg_p.set_defaults(func=cmd_dialogue)

    # ---- voices ----
    voices_p = subparsers.add_parser("voices", help="List available TTS engines and voices")
    voices_p.set_defaults(func=cmd_voices)

    # ---- health ----
    health_p = subparsers.add_parser("health", help="Check service health")
    health_p.set_defaults(func=cmd_health)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
