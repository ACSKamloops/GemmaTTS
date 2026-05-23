"""
Desktop assistant adapter for GemmaTTS.

A lightweight stdin → orchestrator → stdout/speaker loop that can serve as
the backbone for a local desktop assistant or kiosk demo.

Usage
-----
::

    # Text-only (no audio playback):
    python -m app.adapters.desktop

    # With custom orchestrator URL and history depth:
    python -m app.adapters.desktop --url http://localhost:8000 --history 10

Features
--------
* Reads lines from stdin (or a pipe/file redirect).
* Sends each line to the orchestrator ``POST /v1/dialogue``.
* Prints the generated text response to stdout.
* Optionally plays back audio via a pluggable callback.
* Keeps a sliding window of conversation history (default: last 5 turns)
  and prepends it to each prompt so the LLM has context.
* Graceful shutdown on Ctrl-C / EOF.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single user↔assistant exchange."""
    user: str
    assistant: str


@dataclass
class ConversationHistory:
    """Fixed-size sliding window of turns."""
    max_turns: int = 5
    turns: List[Turn] = field(default_factory=list)

    def add(self, user: str, assistant: str) -> None:
        self.turns.append(Turn(user=user, assistant=assistant))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def format_context(self) -> str:
        """Render past turns into a text block for the LLM prompt."""
        if not self.turns:
            return ""
        lines: list[str] = ["Previous conversation:"]
        for t in self.turns:
            lines.append(f"  User: {t.user}")
            lines.append(f"  Assistant: {t.assistant}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

# Type alias for an optional audio-playback callback.
# Signature: (wav_bytes: bytes, sample_rate: int) -> None
AudioPlaybackFn = Callable[[bytes, int], None]


class DesktopAssistantAdapter:
    """Interactive REPL that bridges stdin to the GemmaTTS orchestrator.

    Parameters
    ----------
    orchestrator_url : str
        Base URL of the orchestrator gateway.
    voice_id : str
        Voice preset forwarded as ``speaker.voice_id``.
    audio_format : str
        Desired audio format (``wav``, ``ogg``, ``mp3``).
    request_audio : bool
        Whether to request audio synthesis from the orchestrator.
    max_history : int
        Number of past turns to include as conversational context.
    timeout : float
        HTTP timeout in seconds per orchestrator call.
    playback_fn : AudioPlaybackFn | None
        Optional callback invoked with raw audio bytes and sample rate.
        If ``None``, audio playback is silently skipped.
    tts_style : str | None
        Optional TTS engine hint (``dia``, ``kokoro``, etc.).
    """

    def __init__(
        self,
        orchestrator_url: str = "http://127.0.0.1:8000",
        voice_id: str = "default",
        audio_format: str = "wav",
        request_audio: bool = True,
        max_history: int = 5,
        timeout: float = 30.0,
        playback_fn: Optional[AudioPlaybackFn] = None,
        tts_style: Optional[str] = None,
    ) -> None:
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.voice_id = voice_id
        self.audio_format = audio_format
        self.request_audio = request_audio
        self.timeout = timeout
        self.playback_fn = playback_fn
        self.tts_style = tts_style
        self.history = ConversationHistory(max_turns=max_history)
        self._shutdown = False

    # ---- internal ----------------------------------------------------------

    def _build_prompt(self, user_text: str) -> str:
        """Combine history context with the latest user input."""
        ctx = self.history.format_context()
        if ctx:
            return f"{ctx}\n\nUser: {user_text}"
        return user_text

    async def _call_orchestrator(self, prompt: str) -> dict:
        """POST to ``/v1/dialogue`` and return the JSON body."""
        payload = {
            "request_id": str(uuid.uuid4()),
            "speaker": {
                "id": "desktop-user",
                "name": "Desktop Assistant",
                "voice_id": self.voice_id,
                "style": self.tts_style,
            },
            "user_text": prompt,
            "max_words": 150,
            "output": {
                "audio": self.request_audio,
                "format": self.audio_format,
            },
            "test_mode": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.orchestrator_url}/v1/dialogue",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    async def _fetch_audio(self, audio_id: str) -> bytes:
        """Download the audio file via the signed URL."""
        url = f"{self.orchestrator_url}/audio/{audio_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

    # ---- public API --------------------------------------------------------

    async def process_input(self, user_text: str) -> str:
        """Send a single user utterance and return the assistant's text reply.

        Side-effects: plays audio (if configured), updates history.
        """
        prompt = self._build_prompt(user_text)
        data = await self._call_orchestrator(prompt)

        reply_text = data.get("text", "")

        # Audio playback (best-effort)
        audio_meta = data.get("audio")
        if audio_meta and self.playback_fn:
            audio_id = audio_meta.get("audio_id")
            sample_rate = audio_meta.get("sample_rate", 24000)
            if audio_id:
                try:
                    wav_bytes = await self._fetch_audio(audio_id)
                    self.playback_fn(wav_bytes, sample_rate)
                except Exception:
                    logger.warning(
                        "Audio playback failed for audio_id=%s",
                        audio_id,
                        exc_info=True,
                    )

        self.history.add(user=user_text, assistant=reply_text)
        return reply_text

    async def run_loop(self) -> None:
        """Main interactive REPL.  Reads stdin line-by-line until EOF or
        Ctrl-C."""
        print("GemmaTTS Desktop Assistant")
        print("Type your message and press Enter.  Ctrl-C or EOF to quit.\n")

        loop = asyncio.get_running_loop()

        # Register SIGINT for graceful shutdown (Unix only)
        if sys.platform != "win32":
            loop.add_signal_handler(signal.SIGINT, self._request_shutdown)

        try:
            while not self._shutdown:
                # Read from stdin asynchronously so we don't block the loop
                try:
                    line = await loop.run_in_executor(
                        None, self._read_line,
                    )
                except EOFError:
                    break

                if line is None or self._shutdown:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    reply = await self.process_input(line)
                    print(f"\nAssistant: {reply}\n")
                except httpx.HTTPStatusError as exc:
                    logger.error(
                        "Orchestrator error: %s %s",
                        exc.response.status_code,
                        exc.response.text,
                    )
                    print(
                        f"\n[error] Orchestrator returned "
                        f"{exc.response.status_code}\n"
                    )
                except httpx.TimeoutException:
                    logger.error("Orchestrator request timed out")
                    print("\n[error] Request timed out\n")
                except Exception:
                    logger.exception("Unexpected error during processing")
                    print("\n[error] Something went wrong — see logs\n")

        except KeyboardInterrupt:
            pass
        finally:
            print("\nGoodbye.")

    # ---- helpers -----------------------------------------------------------

    def _request_shutdown(self) -> None:
        self._shutdown = True

    @staticmethod
    def _read_line() -> Optional[str]:
        """Blocking stdin readline (runs in executor)."""
        try:
            sys.stdout.write("You: ")
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:  # EOF
                raise EOFError
            return line
        except KeyboardInterrupt:
            raise EOFError


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GemmaTTS Desktop Assistant (stdin → orchestrator → stdout)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Orchestrator base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--voice",
        default="default",
        help="Voice ID sent to the orchestrator",
    )
    parser.add_argument(
        "--format",
        default="wav",
        choices=["wav", "ogg", "mp3"],
        help="Audio output format (default: wav)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio synthesis (text-only mode)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of past turns to keep as context (default: 5)",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="TTS engine style hint (e.g. dia, kokoro)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Entry-point for ``python -m app.adapters.desktop``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args(argv)

    adapter = DesktopAssistantAdapter(
        orchestrator_url=args.url,
        voice_id=args.voice,
        audio_format=args.format,
        request_audio=not args.no_audio,
        max_history=args.history,
        tts_style=args.style,
    )

    try:
        asyncio.run(adapter.run_loop())
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
