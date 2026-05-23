"""
Dia TTS Worker — Nari Labs Dia 1.6B
https://github.com/nari-labs/dia
https://huggingface.co/nari-labs/Dia-1.6B-0626

Uses HuggingFace transformers (DiaForConditionalGeneration + AutoProcessor).
Requires transformers >= 4.53.0 and descript-audio-codec.
Output: 44100 Hz WAV.
"""
import io
import logging
import numpy as np
import soundfile as sf
import torch
from pathlib import Path

logger = logging.getLogger("dia-worker")


class DiaWorker:
    def __init__(self, model_path: str = "models/dia"):
        self.model_path = Path(model_path)
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 44100  # Dia outputs at 44100 Hz

    def load(self):
        if self.model is None:
            config_path = self.model_path / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Dia model config not found at {config_path}. "
                    f"Run scripts/download_models.py first."
                )

            from transformers import AutoProcessor, DiaForConditionalGeneration

            logger.info(f"Loading Dia 1.6B from {self.model_path} on {self.device}...")

            self.processor = AutoProcessor.from_pretrained(str(self.model_path))
            self.model = DiaForConditionalGeneration.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
            logger.info("Dia 1.6B loaded successfully.")

    def _format_text(self, text: str) -> str:
        """
        Ensure text has [S1] speaker tags for Dia format.
        If no speaker tags present, wrap in [S1] for single-speaker mode.
        """
        if "[S1]" not in text and "[S2]" not in text:
            return f"[S1] {text}"
        return text

    def synthesize(self, text: str, voice_id: str = "default") -> tuple[bytes, int]:
        """
        Synthesize text to WAV bytes using Dia 1.6B.

        Args:
            text: Text to speak. Supports [S1]/[S2] speaker tags and
                  non-verbal cues like (laughs), (sighs), etc.
            voice_id: Currently unused — Dia uses speaker tags for voice control.

        Returns:
            (wav_bytes, sample_rate)
        """
        self.load()

        formatted_text = self._format_text(text)

        inputs = self.processor(
            text=[formatted_text],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=3072,  # ~35 seconds of audio
                guidance_scale=3.0,
                temperature=1.8,
                top_p=0.90,
                top_k=50,
                do_sample=True,
            )

        # Decode tokens to audio waveform
        audio_data = self.processor.batch_decode(outputs)

        # audio_data is a list of numpy arrays, take the first one
        if isinstance(audio_data, list) and len(audio_data) > 0:
            waveform = audio_data[0]
        else:
            waveform = audio_data

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        waveform = np.squeeze(waveform)

        buffer = io.BytesIO()
        sf.write(buffer, waveform, self.sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), self.sample_rate
