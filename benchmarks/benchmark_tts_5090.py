import time
import torch
import numpy as np
from app.services.tts_service import get_worker
from app.audio.probe import get_audio_duration_ms
from app.config import settings

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        return f"{allocated:.2f} GB / {reserved:.2f} GB"
    return "N/A"

def run_benchmark():
    print("=== TTS Performance Benchmark ===")
    print("Measuring latency, RTF, and VRAM usage on available engines.")
    
    text = "The quick brown fox jumps over the lazy dog. Voice synthesis is fast and highly efficient."
    engines = ["chatterbox", "kokoro", "piper", "dia", "f5_tts"]
    
    results = []
    
    for engine in engines:
        print(f"\n--- Benchmarking Engine: {engine} ---")
        try:
            from unittest.mock import patch
            with patch.object(settings, "mode", "dev"):
                worker = get_worker(engine)
                
                # Warmup
                print("Warming up...")
                worker.synthesize("Hello warmup.", "default")
                
                # Run actual benchmark
                print("Synthesizing...")
                t0 = time.time()
                wav_bytes, sample_rate = worker.synthesize(text, "default")
                elapsed = time.time() - t0
                
                duration_ms = get_audio_duration_ms(wav_bytes, "wav")
                duration_sec = duration_ms / 1000.0
                rtf = elapsed / duration_sec if duration_sec > 0 else 0
                vram = print_gpu_memory()
                
                print(f"Success! Latency: {elapsed:.3f}s | Audio Duration: {duration_sec:.2f}s | RTF: {rtf:.3f} | VRAM: {vram}")
                results.append({
                    "engine": engine,
                    "latency": f"{elapsed:.3f}s",
                    "duration": f"{duration_sec:.2f}s",
                    "rtf": f"{rtf:.3f}",
                    "vram": vram,
                    "status": "Success"
                })
        except Exception as e:
            print(f"Skipped/Failed {engine}: {e}")
            results.append({
                "engine": engine,
                "latency": "N/A",
                "duration": "N/A",
                "rtf": "N/A",
                "vram": "N/A",
                "status": f"Failed ({type(e).__name__})"
            })
            
    print("\n=== Benchmark Summary Table ===")
    print(f"{'Engine':<15} | {'Latency':<10} | {'Duration':<10} | {'RTF':<8} | {'VRAM':<15} | {'Status':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['engine']:<15} | {r['latency']:<10} | {r['duration']:<10} | {r['rtf']:<8} | {r['vram']:<15} | {r['status']:<15}")

if __name__ == "__main__":
    run_benchmark()
