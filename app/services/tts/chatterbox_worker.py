"""
Chatterbox TTS Worker — Resemble AI Chatterbox
https://github.com/resemble-ai/chatterbox

Uses ChatterboxTTS.from_pretrained() which auto-downloads weights from HuggingFace.
Output: 24 kHz WAV, PyTorch tensor waveform.
"""
import io
import logging
import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger("chatterbox-worker")


class ChatterboxWorker:
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
        """
        Synthesize text to WAV bytes.

        Args:
            text: Text to speak.
            voice_id: If a file path to a .wav reference clip, enables
                      zero-shot voice cloning. Otherwise uses default voice.

        Returns:
            (wav_bytes, sample_rate)
        """
        self.load()

        generate_kwargs = {"text": text}

        # If voice_id points to a .wav file, use it as voice cloning reference
        if voice_id and voice_id != "default" and voice_id.endswith(".wav"):
            import os
            if os.path.isfile(voice_id):
                generate_kwargs["audio_prompt_path"] = voice_id
                logger.info(f"Using voice cloning reference: {voice_id}")

        wav_tensor = self.model.generate(**generate_kwargs)
        sample_rate = self.model.sr  # 24000

        # Convert PyTorch tensor to numpy for soundfile
        # Chatterbox returns shape (1, num_samples)
        waveform = wav_tensor.squeeze().cpu().numpy()

        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), sample_rate
