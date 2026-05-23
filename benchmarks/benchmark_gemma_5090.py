import time
import torch
import psutil
from app.providers.llm.gemma4_transformers import Gemma4TransformersProvider
from app.config import settings

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"CUDA memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    else:
        print("CUDA not available")

def run_benchmark():
    print("=== Gemma 4 E4B-it Benchmark ===")
    print(f"Model ID: {settings.gemma_model_id}")
    print(f"Model Path: {settings.gemma_model_path}")
    print_gpu_memory()

    print("\nLoading provider...")
    t0 = time.time()
    provider = Gemma4TransformersProvider()
    provider.load()
    load_time = time.time() - t0
    print(f"Load time: {load_time:.2f} seconds")
    print_gpu_memory()

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in three sentences.",
        "Write a short backstory for a sci-fi game character named Maria, who is an engineer on a cargo spaceship.",
    ]

    print("\nWarmup run...")
    provider.generate("Hello!", max_words=10)

    print("\nRunning benchmark prompts...")
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: '{prompt}'")
        t_start = time.time()
        output = provider.generate(prompt, max_words=150)
        elapsed = time.time() - t_start
        word_count = len(output.split())
        words_per_sec = word_count / elapsed if elapsed > 0 else 0
        print(f"Output: {output}")
        print(f"Elapsed: {elapsed:.3f} seconds | Word count: {word_count} | ~{words_per_sec:.1f} words/sec")
        print_gpu_memory()

if __name__ == "__main__":
    run_benchmark()
