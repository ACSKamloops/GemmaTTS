import os
import io
import logging
import numpy as np
import soundfile as sf
import torch

from app.providers.tts.base import TTSProvider

logger = logging.getLogger("kokoro-provider")

class KokoroProvider(TTSProvider):
    """
    Kokoro 82M ONNX TTS Provider.
    """
    def __init__(
        self,
        model_path: str = "models/kokoro/onnx/model.onnx",
        voices_dir: str = "models/kokoro/voices",
    ):
        self.model_path = model_path
        self.voices_dir = voices_dir
        self.kokoro = None
        self._voice_cache: dict[str, np.ndarray] = {}

    def load(self):
        if self.kokoro is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Kokoro ONNX model not found at {self.model_path}")

        from kokoro_onnx import Kokoro

        dummy_voices_path = os.path.join(self.voices_dir, "_voices_stub.npy")
        if not os.path.exists(dummy_voices_path):
            os.makedirs(self.voices_dir, exist_ok=True)
            np.save(dummy_voices_path, np.zeros((1,), dtype=np.float32))

        # Use CUDA if available
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            logger.info("Kokoro: Using CUDAExecutionProvider")
            # Set ONNX provider env var so kokoro-onnx can pick it up
            os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"
        else:
            logger.info("Kokoro: Using CPUExecutionProvider")

        from app.config import settings

        self.kokoro = Kokoro(self.model_path, voices_path=dummy_voices_path)
        
        if settings.kokoro_provider_mode == "legacy_manual_embedding":
            # Safely wrap internal sess.run to bypass onnxruntime input shape/type restrictions
            original_run = self.kokoro.sess.run
            def patched_run(output_names, input_feed, run_options=None):
                if "speed" in input_feed:
                    input_feed["speed"] = np.array(input_feed["speed"], dtype=np.float32)
                if "style" in input_feed:
                    style = input_feed["style"]
                    if len(style.shape) == 1:
                        input_feed["style"] = np.expand_dims(style, axis=0)
                return original_run(output_names, input_feed, run_options)
            self.kokoro.sess.run = patched_run

        logger.info("Kokoro ONNX model loaded successfully.")

    def _load_voice_embedding(self, voice_id: str) -> np.ndarray:
        if voice_id in self._voice_cache:
            return self._voice_cache[voice_id]

        pt_path = os.path.join(self.voices_dir, f"{voice_id}.pt")
        bin_path = os.path.join(self.voices_dir, f"{voice_id}.bin")

        if os.path.exists(pt_path):
            embedding = torch.load(pt_path, map_location="cpu", weights_only=True)
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.numpy()
            embedding = embedding.astype(np.float32).reshape(-1, 256)
        elif os.path.exists(bin_path):
            embedding = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 256)
        else:
            raise FileNotFoundError(
                f"Voice embedding '{voice_id}' not found at {pt_path} or {bin_path}"
            )

        self._voice_cache[voice_id] = embedding
        return embedding

    def _resolve_voice(self, voice_id: str) -> np.ndarray:
        if "+" in voice_id:
            parts = [p.strip() for p in voice_id.split("+") if p.strip()]
            embeddings = [self._load_voice_embedding(p) for p in parts]
            min_len = min(e.shape[0] for e in embeddings)
            return np.mean([e[:min_len] for e in embeddings], axis=0)
        return self._load_voice_embedding(voice_id)

    def synthesize(self, text: str, voice_id: str = "default") -> tuple[bytes, int]:
        self.load()

        from app.config import settings

        if voice_id == "default":
            voice_id = "af_heart"

        if settings.kokoro_provider_mode == "legacy_manual_embedding":
            embedding = self._resolve_voice(voice_id)
        else:
            embedding = self.kokoro.get_voice(voice_id)

        samples, sample_rate = self.kokoro.create(
            text, voice=embedding, speed=1.0, lang="en-us"
        )

        samples = np.squeeze(samples)

        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue(), sample_rate
