#!/usr/bin/env python3
"""Quick smoke test to verify all worker imports work."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Python:", sys.version)
print()

results = []

def test_import(name, fn):
    try:
        fn()
        print(f"  ✓ {name}")
        results.append(True)
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        results.append(False)

print("--- Worker imports ---")
test_import("KokoroWorker", lambda: __import__("app.services.tts.kokoro_worker", fromlist=["KokoroWorker"]))
test_import("PiperWorker", lambda: __import__("app.services.tts.piper_worker", fromlist=["PiperWorker"]))
test_import("ChatterboxWorker", lambda: __import__("app.services.tts.chatterbox_worker", fromlist=["ChatterboxWorker"]))
test_import("DiaWorker", lambda: __import__("app.services.tts.dia_worker", fromlist=["DiaWorker"]))
test_import("F5TTSWorker", lambda: __import__("app.services.tts.f5_tts_worker", fromlist=["F5TTSWorker"]))
test_import("AudioPipeline", lambda: __import__("app.audio.pipeline", fromlist=["AudioPipeline"]))
test_import("Config", lambda: __import__("app.config", fromlist=["settings"]))

print("\n--- Library imports ---")
test_import("kokoro_onnx.Kokoro", lambda: __import__("kokoro_onnx", fromlist=["Kokoro"]))
test_import("piper.PiperVoice", lambda: __import__("piper", fromlist=["PiperVoice"]))
test_import("chatterbox.tts.ChatterboxTTS", lambda: __import__("chatterbox.tts", fromlist=["ChatterboxTTS"]))

# transformers import can be slow and may have CUDA init warnings
try:
    import torch
    print(f"\n  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  torch info: {e}")

test_import("transformers.DiaForConditionalGeneration", lambda: __import__("transformers", fromlist=["DiaForConditionalGeneration"]))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
