from abc import ABC, abstractmethod
from typing import Optional

class LLMProvider(ABC):
    """Abstract base class for LLM text generation providers."""

    @abstractmethod
    def generate(self, prompt: str, max_words: Optional[int] = None, enable_thinking: bool = False) -> str:
        """
        Generate text response for a given prompt.
        """
        pass
