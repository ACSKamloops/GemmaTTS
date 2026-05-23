"""
Local speaker playback adapter for GemmaTTS.

Plays WAV audio bytes through the system's default audio output device.
Supports blocking (wait for completion) and non-blocking (background) modes.

Primary backend: ``sounddevice`` (cross-platform, low-latency).
Fallback backend: CLI tools — ``aplay`` (Linux/WSL) or ``afplay`` (macOS).
The fallback is selected automatically when sounddevice is unavailable or
when the ``GEMMA_PLAYBACK_BACKEND`` env var is set to ``"cli"``.
"""

from __future__ import annotations

import io
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from enum import Enum, auto
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger("playback-adapter")


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

class PlaybackBackend(Enum):
    """Available playback backends."""
    SOUNDDEVICE = auto()
    CLI = auto()


def _detect_backend() -> PlaybackBackend:
    """Pick the best available playback backend.

    Respects the ``GEMMA_PLAYBACK_BACKEND`` environment variable
    (values: ``"sounddevice"`` | ``"cli"``).  Otherwise falls back
    automatically when sounddevice cannot be imported.
    """
    env_override = os.environ.get("GEMMA_PLAYBACK_BACKEND", "").lower().strip()
    if env_override == "cli":
        logger.info("Playback backend forced to CLI via GEMMA_PLAYBACK_BACKEND")
        return PlaybackBackend.CLI
    if env_override == "sounddevice":
        return PlaybackBackend.SOUNDDEVICE

    try:
        import sounddevice  # noqa: F401
        return PlaybackBackend.SOUNDDEVICE
    except (ImportError, OSError) as exc:
        logger.warning("sounddevice unavailable (%s); falling back to CLI backend", exc)
        return PlaybackBackend.CLI


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PlaybackError(Exception):
    """Raised when audio playback fails."""


class NoAudioDeviceError(PlaybackError):
    """Raised when no suitable audio output device is found."""


class UnsupportedFormatError(PlaybackError):
    """Raised when the supplied audio bytes cannot be decoded."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes into a NumPy array and sample rate.

    Raises ``UnsupportedFormatError`` when the bytes are not valid audio.
    """
    try:
        data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        return data, samplerate
    except Exception as exc:
        raise UnsupportedFormatError(
            f"Could not decode audio bytes ({len(wav_bytes)} B): {exc}"
        ) from exc


def _resolve_cli_player() -> str:
    """Return the path to a CLI audio player or raise ``NoAudioDeviceError``."""
    system = platform.system()
    if system == "Linux":
        player = shutil.which("aplay")
        if player:
            return player
        # Try paplay (PulseAudio) as secondary option
        player = shutil.which("paplay")
        if player:
            return player
        raise NoAudioDeviceError(
            "No CLI audio player found on Linux. "
            "Install alsa-utils (aplay) or pulseaudio-utils (paplay)."
        )
    if system == "Darwin":
        player = shutil.which("afplay")
        if player:
            return player
        raise NoAudioDeviceError("afplay not found on macOS.")
    raise NoAudioDeviceError(f"No CLI audio player available for {system}.")


# ---------------------------------------------------------------------------
# Playback adapter
# ---------------------------------------------------------------------------

class PlaybackAdapter:
    """Plays WAV audio bytes on the local speaker.

    Parameters
    ----------
    backend : PlaybackBackend | None
        Explicit backend choice.  ``None`` (default) auto-detects.
    device : int | str | None
        ``sounddevice`` output device index or name.  Ignored for CLI.
    """

    def __init__(
        self,
        backend: Optional[PlaybackBackend] = None,
        device: int | str | None = None,
    ) -> None:
        self._backend = backend or _detect_backend()
        self._device = device
        self._lock = threading.Lock()

        # Handles for interrupting playback
        self._sd_event: Optional[threading.Event] = None
        self._cli_proc: Optional[subprocess.Popen] = None
        self._playback_thread: Optional[threading.Thread] = None

        logger.info("PlaybackAdapter initialised (backend=%s)", self._backend.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend(self) -> PlaybackBackend:
        return self._backend

    def play_audio(self, wav_bytes: bytes, blocking: bool = True) -> None:
        """Play WAV audio through the default output device.

        Parameters
        ----------
        wav_bytes : bytes
            Raw WAV file content (including header).
        blocking : bool
            If ``True`` (default), the call blocks until playback finishes
            or ``stop()`` is called.  If ``False``, playback runs in a
            background thread and the call returns immediately.

        Raises
        ------
        UnsupportedFormatError
            The supplied bytes could not be decoded as audio.
        NoAudioDeviceError
            No usable audio output device was found.
        PlaybackError
            A runtime playback error occurred.
        """
        if not wav_bytes:
            raise UnsupportedFormatError("Empty audio bytes supplied.")

        # Decode eagerly so callers get format errors synchronously
        data, samplerate = _decode_wav(wav_bytes)
        logger.info(
            "play_audio: %.2f s, %d Hz, channels=%s, blocking=%s",
            len(data) / samplerate,
            samplerate,
            data.shape[1] if data.ndim > 1 else 1,
            blocking,
        )

        if blocking:
            self._play(data, samplerate, wav_bytes)
        else:
            t = threading.Thread(
                target=self._play,
                args=(data, samplerate, wav_bytes),
                daemon=True,
                name="playback-worker",
            )
            with self._lock:
                self._playback_thread = t
            t.start()

    def stop(self) -> None:
        """Interrupt any in-progress playback immediately."""
        logger.info("stop() requested")
        with self._lock:
            # Signal sounddevice to stop
            if self._sd_event is not None:
                self._sd_event.set()
            # Kill CLI subprocess
            if self._cli_proc is not None:
                try:
                    self._cli_proc.terminate()
                except OSError:
                    pass

    def wait(self, timeout: float | None = None) -> None:
        """Block until background playback finishes.

        Only useful after a non-blocking ``play_audio`` call.
        """
        with self._lock:
            t = self._playback_thread
        if t is not None:
            t.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _play(
        self,
        data: np.ndarray,
        samplerate: int,
        wav_bytes: bytes,
    ) -> None:
        """Route to the correct backend."""
        if self._backend is PlaybackBackend.SOUNDDEVICE:
            self._play_sounddevice(data, samplerate)
        else:
            self._play_cli(wav_bytes)

    # ------------------------------------------------------------------
    # sounddevice backend
    # ------------------------------------------------------------------

    def _play_sounddevice(self, data: np.ndarray, samplerate: int) -> None:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise NoAudioDeviceError(
                f"sounddevice is not available: {exc}"
            ) from exc

        stop_event = threading.Event()
        with self._lock:
            self._sd_event = stop_event

        try:
            sd.play(data, samplerate, device=self._device)
            logger.debug("sounddevice playback started")

            # Wait for playback to finish or stop() to be called
            duration = len(data) / samplerate
            # Poll in 50 ms increments so stop() is responsive
            elapsed = 0.0
            while elapsed < duration + 0.5:
                if stop_event.is_set():
                    sd.stop()
                    logger.info("Playback interrupted by stop()")
                    return
                stop_event.wait(timeout=0.05)
                elapsed += 0.05

            sd.wait()
            logger.info("sounddevice playback complete")
        except sd.PortAudioError as exc:
            raise NoAudioDeviceError(
                f"PortAudio error (no audio device?): {exc}"
            ) from exc
        except Exception as exc:
            raise PlaybackError(f"sounddevice playback failed: {exc}") from exc
        finally:
            with self._lock:
                self._sd_event = None

    # ------------------------------------------------------------------
    # CLI fallback backend
    # ------------------------------------------------------------------

    def _play_cli(self, wav_bytes: bytes) -> None:
        player = _resolve_cli_player()
        logger.debug("CLI playback via %s", player)

        # Write to a temp file (some players don't accept stdin)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            os.write(tmp_fd, wav_bytes)
            os.close(tmp_fd)

            proc = subprocess.Popen(
                [player, tmp_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            with self._lock:
                self._cli_proc = proc

            _, stderr = proc.communicate()
            if proc.returncode not in (0, -15, -9):
                # -15 = SIGTERM (stop()), -9 = SIGKILL — not errors
                raise PlaybackError(
                    f"{player} exited with code {proc.returncode}: "
                    f"{stderr.decode(errors='replace').strip()}"
                )
            logger.info("CLI playback complete (exit=%d)", proc.returncode)
        finally:
            with self._lock:
                self._cli_proc = None
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_adapter: Optional[PlaybackAdapter] = None
_default_lock = threading.Lock()


def _get_default_adapter() -> PlaybackAdapter:
    """Lazily initialise a module-level singleton adapter."""
    global _default_adapter
    with _default_lock:
        if _default_adapter is None:
            _default_adapter = PlaybackAdapter()
        return _default_adapter


def play_audio(wav_bytes: bytes, blocking: bool = True) -> None:
    """Play WAV bytes on the default speaker.

    Convenience wrapper around :class:`PlaybackAdapter`.
    See :meth:`PlaybackAdapter.play_audio` for full docs.
    """
    _get_default_adapter().play_audio(wav_bytes, blocking=blocking)


def stop() -> None:
    """Stop any in-progress playback on the default adapter."""
    global _default_adapter
    with _default_lock:
        if _default_adapter is not None:
            _default_adapter.stop()
