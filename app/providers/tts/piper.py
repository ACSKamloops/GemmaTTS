import os
import io
import wave
from app.providers.tts.base import TTSProvider

class PiperProvider(TTSProvider):
    """
    Piper ONNX TTS Provider.
    """
    def __init__(self, model_path: str = "models/piper/en_US-lessac-medium.onnx"):
        self.model_path = model_path
        self.voice = None

    def load(self):
        if self.voice is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Piper ONNX model not found at {self.model_path}")
            from piper import PiperVoice
            self.voice = PiperVoice.load(self.model_path)

    def synthesize(self, text: str, voice_id: str = "default") -> tuple[bytes, int]:
        self.load()
        
        chunks = list(self.voice.synthesize(text))
        if not chunks:
            raise RuntimeError("Piper synthesis returned no audio chunks")
            
        sample_rate = chunks[0].sample_rate
        sample_width = chunks[0].sample_width
        channels = chunks[0].sample_channels
        
        raw_audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        buffer = io.BytesIO()
        
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio_bytes)
            
        return buffer.getvalue(), sample_rate
