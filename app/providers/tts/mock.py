import io
import re
import struct
import wave
from typing import Optional
from fastapi import HTTPException

from app.providers.tts.base import TTSProvider
from app.config import settings

class MockTTSProvider(TTSProvider):
    """
    Mock TTS provider generating dummy WAV audio.
    Used for unit tests and E2E simulation modes.
    """
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def generate_dummy_wav(self, duration: float = 1.0, size_bytes: Optional[int] = None) -> bytes:
        if size_bytes is not None:
            data_size = max(0, size_bytes - 44)
            num_samples = data_size // 2
        else:
            num_samples = int(duration * self.sample_rate)
            data_size = num_samples * 2
            
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)      # Mono
            wav_file.setsampwidth(2)      # 16-bit
            wav_file.setframerate(self.sample_rate)
            
            chunk_size = 1000
            val_true = struct.pack("<h", 3000)
            val_false = struct.pack("<h", -3000)
            
            samples_written = 0
            while samples_written < num_samples:
                to_write = min(chunk_size, num_samples - samples_written)
                chunk = b"".join(val_true if (i // 120) % 2 == 0 else val_false for i in range(to_write))
                wav_file.writeframes(chunk)
                samples_written += to_write
                
        return buffer.getvalue()

    def synthesize(self, text: str, voice_id: str = "default") -> tuple[bytes, int]:
        # Handle test/simulation triggers ONLY if in test mode
        if settings.mode == "test":
            if voice_id and "simulate_offline" in voice_id:
                raise HTTPException(
                    status_code=503,
                    detail="Dia engine offline"
                )

            size_bytes = None
            if "size_bytes=" in text:
                m = re.search(r"size_bytes=(\d+)", text)
                if m:
                    size_bytes = int(m.group(1))

            duration = 1.0
            if "duration_sec=" in text:
                m = re.search(r"duration_sec=([\d\.]+)", text)
                if m:
                    duration = float(m.group(1))

            wav_bytes = self.generate_dummy_wav(duration=duration, size_bytes=size_bytes)
            return wav_bytes, self.sample_rate

        # Simple static mock response for non-test mode if loaded
        wav_bytes = self.generate_dummy_wav(duration=1.0)
        return wav_bytes, self.sample_rate
