import logging
import threading
from typing import Optional
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from app.config import settings
from app.providers.llm.base import LLMProvider

logger = logging.getLogger("gemma4-transformers-provider")

class Gemma4TransformersProvider(LLMProvider):
    """
    Real inference provider for google/gemma-4-E4B-it model using Hugging Face transformers.
    """
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            if self.model is not None:
                return

            PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
            possible_paths = [
                PROJECT_ROOT / "models" / "gemma",
                PROJECT_ROOT / "models" / "gemma_test",
            ]
            
            selected_path = None
            for path in possible_paths:
                if path.exists() and (path / "config.json").exists():
                    selected_path = path
                    break

            # If load mode is local/hub or no local files, resolve route
            if not selected_path:
                model_id = settings.gemma_model_id
                logger.warning(f"No local gemma models found, loading from hub/id: {model_id}")
                load_target = model_id
            else:
                load_target = str(selected_path)
                
            logger.info(f"Loading Gemma 4 model from: {load_target} on device: {self.device}")

            try:
                self.processor = AutoProcessor.from_pretrained(load_target)
            except Exception as e:
                logger.error(f"Failed to load AutoProcessor: {e}")
                raise e

            try:
                logger.info("Attempting to load model in bfloat16 precision...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    load_target,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="auto" if self.device == "cuda" else None
                )
                if self.device == "cpu":
                    self.model = self.model.to("cpu")
                logger.info("Successfully loaded Gemma 4 model in bfloat16.")
            except Exception as bf_err:
                logger.warning(f"bfloat16 load failed ({bf_err}). Falling back to float32 precision.")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        load_target,
                        low_cpu_mem_usage=True,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    if self.device == "cpu":
                        self.model = self.model.to("cpu")
                    logger.info("Successfully loaded model with float32 precision fallback.")
                except Exception as fatal_err:
                    logger.error(f"Fatal: Failed to load Gemma 4 model: {fatal_err}")
                    raise fatal_err

            self.model.eval()

    def generate(self, prompt: str, max_words: Optional[int] = None, enable_thinking: bool = False) -> str:
        """
        Generate text using the Gemma 4 processor chat template and model.
        """
        if self.model is None or self.processor is None:
            self.load()

        messages = [{"role": "user", "content": prompt}]
        
        # Apply official chat template with thinking settings
        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        inputs = self.processor(text=formatted_prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]
        
        max_words_val = max_words or settings.max_text_words
        max_new_tokens = max(50, max_words_val * 3)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        generated_tokens = outputs[0][input_len:]
        response = self.processor.decode(generated_tokens, skip_special_tokens=False)
        
        # Parse spoken response using the processor API
        parsed_response = self.processor.parse_response(response)
        if isinstance(parsed_response, dict):
            spoken_response = parsed_response.get("content", "")
        elif isinstance(parsed_response, list) and len(parsed_response) > 0 and isinstance(parsed_response[0], dict):
            spoken_response = parsed_response[0].get("content", "")
        else:
            spoken_response = str(parsed_response)
        
        # Trim whitespace
        spoken_response = spoken_response.strip()

        # Word limits enforcement
        if max_words_val is not None:
            words = spoken_response.split()
            if len(words) > max_words_val:
                spoken_response = " ".join(words[:max_words_val])

        return spoken_response
