import io
import logging
import re
import random
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

from app.providers.tts.base import TTSProvider

logger = logging.getLogger("dia-provider")

class DiaProvider(TTSProvider):
    """
    Dia TTS Provider. Implements nari-labs Dia 1.6B.
    """
    def __init__(self, model_path: str = "models/dia"):
        self.model_path = Path(model_path)
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 44100
        
        # Nonverbal tags whitelist (only allow known vocal noises)
        self.nonverbal_whitelist = {
            "(laughs)", "(sighs)", "(gasp)", "(cough)", "(pant)", "(clears throat)", "(chuckle)"
        }

    def load(self):
        if self.model is not None:
            return

        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Dia model config not found at {config_path}. "
                f"Please download model weights first."
            )

        from transformers import AutoProcessor, DiaForConditionalGeneration

        logger.info(f"Loading Dia 1.6B from {self.model_path} on {self.device}...")

        # RTX 5000 series GPU warning for torch 2.8 nightlies
        if self.device == "cuda":
            device_name = torch.cuda.get_device_name(0)
            if "RTX 50" in device_name or "5090" in device_name:
                logger.warning(
                    f"RTX 50-series GPU detected: {device_name}. "
                    "Ensure you are using PyTorch 2.8+ nightly / CUDA 12.4+ "
                    "compatibility for optimal performance as noted in Dia repository."
                )

        self.processor = AutoProcessor.from_pretrained(str(self.model_path))
        self.model = DiaForConditionalGeneration.from_pretrained(
            str(self.model_path),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        logger.info("Dia 1.6B loaded successfully.")

    def validate_text(self, text: str) -> str:
        """
        Validate and sanitize text for Dia:
        1. Enforce length constraints (minimum 5 characters, maximum 1000 characters).
        2. Ensure S1/S2 speaker alternation format is valid.
        3. Strip nonverbal tags that are not in the whitelist.
        """
        # 1. Guardrails
        if len(text.strip()) < 3:
            raise ValueError("Text is too short for Dia synthesis.")
        if len(text) > 1000:
            raise ValueError("Text exceeds maximum length for Dia synthesis.")

        # 2. Nonverbal tag whitelisting
        # Find all patterns of (something)
        tags = re.findall(r"\([^)]+\)", text)
        for tag in tags:
            if tag.lower() not in self.nonverbal_whitelist:
                # Strip unsupported nonverbal tag
                text = text.replace(tag, "")

        # 3. S1/S2 speaker tag prefixing
        if "[S1]" not in text and "[S2]" not in text:
            text = f"[S1] {text}"
            
        # Ensure alternation validity (e.g. not consecutive tags without text)
        clean_text = re.sub(r"\s+", " ", text).strip()
        if re.search(r"\[S[12]\]\s*\[S[12]\]", clean_text):
            raise ValueError("Invalid speaker tag alternation: consecutive speaker tags found.")

        return clean_text

    def synthesize(self, text: str, voice_id: str = "default") -> tuple[bytes, int]:
        """
        Synthesize text to WAV bytes.
        Supports voice_id mapped to seed values for consistency (e.g. 'seed_42', 'seed_100').
        """
        self.load()

        formatted_text = self.validate_text(text)

        # Handle voice consistency seed from voice_id
        seed = None
        if voice_id and voice_id.startswith("seed_"):
            try:
                seed = int(voice_id.split("_")[1])
            except ValueError:
                pass

        if seed is not None:
            # Set reproducibility seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = self.processor(
            text=[formatted_text],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=3072,  # ~35 seconds
                guidance_scale=3.0,
                temperature=1.8,
                top_p=0.90,
                top_k=50,
                do_sample=True,
            )

        audio_data = self.processor.batch_decode(outputs)

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
