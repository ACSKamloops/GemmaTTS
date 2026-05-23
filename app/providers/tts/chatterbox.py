import io
import logging
import os
import torch
import soundfile as sf
from pathlib import Path

from app.config import settings
from app.audio.cache import is_safe_path
from app.providers.tts.base import TTSProvider

logger = logging.getLogger("chatterbox-provider")

class ChatterboxProvider(TTSProvider):
    """
    Chatterbox TTS Provider. Supports secure zero-shot voice cloning.
    """
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        if self.model is None:
            from chatterbox.tts import ChatterboxTTS

            logger.info(f"Loading ChatterboxTTS on device '{self.device}'...")
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            logger.info("ChatterboxTTS loaded successfully.")

    def synthesize(self, text: str, voice_id: str = "default") -> tuple[bytes, int]:
        self.load()

        generate_kwargs = {"text": text}

        # Safe voice cloning reference loading
        if voice_id and voice_id != "default":
            # Direct path traversal validation
            if ".." in voice_id or "/" in voice_id or "\\" in voice_id:
                raise ValueError("Invalid voice_id: Path traversal or out-of-boundary characters detected.")
            
            # Resolve reference file under settings.voice_ref_dir
            ref_dir = Path(settings.voice_ref_dir)
            ref_path = (ref_dir / voice_id).resolve()
            
            if not is_safe_path(ref_path, ref_dir):
                raise ValueError("Invalid voice_id: Path traversal attempt blocked.")
                
            if ref_path.is_file() and ref_path.suffix == ".wav":
                generate_kwargs["audio_prompt_path"] = str(ref_path)
                logger.info(f"Using safe voice cloning reference: {ref_path}")
            else:
                logger.warning(f"Voice cloning reference not found or invalid: {ref_path}")

        wav_tensor = self.model.generate(**generate_kwargs)
        sample_rate = self.model.sr  # 24000

        # Convert PyTorch tensor to numpy
        waveform = wav_tensor.squeeze().cpu().numpy()

        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), sample_rate
