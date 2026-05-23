import re
from typing import Optional
from app.providers.llm.base import LLMProvider
from app.config import settings

class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider for unit tests and E2E simulation.
    Allows simulating failures, thinking blocks, and pre-baked answers.
    """
    def __init__(self, enable_thinking: bool = False):
        self.enable_thinking = enable_thinking

    def generate(self, prompt: str, max_words: Optional[int] = None, enable_thinking: bool = False) -> str:
        lowered_prompt = prompt.lower()
        
        # Scenario simulation
        if "simulate_llm_bad_json" in prompt:
            # Return raw bad response
            return "not-a-json-string-at-all"
            
        if "sword" in lowered_prompt or "buy" in lowered_prompt:
            reply = "MOCK_RESPONSE: You bought the sword. Merchant gold is now 50."
        elif "change" in lowered_prompt:
            reply = "MOCK_RESPONSE: Yes, I have change. My gold is 50."
        elif "sell" in lowered_prompt:
            reply = "MOCK_RESPONSE: I sell swords and shields. I have 10 gold."
        else:
            reply = f"MOCK_RESPONSE: {prompt}"

        is_thinking_enabled = self.enable_thinking or enable_thinking

        if is_thinking_enabled:
            reply = f"<think>Thinking...</think> {reply}"

        # Clean/truncate reply
        if not is_thinking_enabled:
            reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL)
            reply = re.sub(r'<think>.*', '', reply, flags=re.DOTALL)
        reply = reply.strip()

        max_words_val = max_words or settings.max_text_words
        if max_words_val is not None:
            words = reply.split()
            if len(words) > max_words_val:
                reply = " ".join(words[:max_words_val])

        return reply
