import os
import io
import numpy as np
import soundfile as sf
from pathlib import Path
from fastapi import HTTPException, status
import logging

logger = logging.getLogger("f5-tts-worker")

class F5TTSWorker:
    def __init__(self, model_dir: str = "models/f5_tts"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.ref_file_path = self.model_dir / "ref.wav"
        self.ref_text = "Hello."

    def ensure_ref_audio(self):
        """
        Creates a short dummy reference WAV file at 24kHz if it doesn't exist,
        enabling self-contained voice synthesis.
        """
        if not self.ref_file_path.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)
            sample_rate = 24000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            # Create a silent/extremely low amplitude 440Hz hum for the reference audio
            data = 0.01 * np.sin(2 * np.pi * 440 * t)
            sf.write(self.ref_file_path, data, sample_rate, format="WAV", subtype="PCM_16")
            logger.info(f"Created default F5-TTS reference audio at {self.ref_file_path}")

    def load(self):
        if self.model is None:
            self.ensure_ref_audio()
            ckpt_path = self.model_dir / "model_1250000.safetensors"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"F5-TTS model checkpoint not found at {ckpt_path}")
            
            try:
                from f5_tts.api import F5TTS
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading F5-TTS on device '{device}' using checkpoint {ckpt_path}...")
                self.model = F5TTS(
                    model="F5TTS_v1_Base",
                    ckpt_file=str(ckpt_path),
                    vocoder_local_path=str(self.model_dir),
                    device=device
                )
                logger.info("F5-TTS loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load F5-TTS: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load F5-TTS model: {str(e)}"
                )

    def synthesize(self, text: str, voice_id: str = "default") -> tuple[bytes, int]:
        self.load()
        
        try:
            # F5-TTS infer method signature:
            # infer(self, ref_file, ref_text, gen_text, ...) -> returns (wav, sr, spec)
            wav, sr, spec = self.model.infer(
                ref_file=str(self.ref_file_path),
                ref_text=self.ref_text,
                gen_text=text
            )
            
            # F5-TTS might return wav as torch.Tensor or numpy array
            import torch
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            elif hasattr(wav, "numpy"):
                wav = wav.numpy()
                
            buffer = io.BytesIO()
            sf.write(buffer, wav, sr, format="WAV", subtype="PCM_16")
            return buffer.getvalue(), sr
        except Exception as e:
            logger.error(f"F5-TTS synthesis failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"F5-TTS synthesis failure: {str(e)}"
            )
