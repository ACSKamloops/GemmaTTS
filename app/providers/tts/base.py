from abc import ABC, abstractmethod

class TTSProvider(ABC):
    """Abstract base class for Text-To-Speech providers."""

    @abstractmethod
    def synthesize(self, text: str, voice_id: str) -> tuple[bytes, int]:
        """
        Synthesize text into WAV audio.
        Returns:
            (wav_bytes, sample_rate)
        """
        pass
