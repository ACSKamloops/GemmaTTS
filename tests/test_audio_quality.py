import pytest
import numpy as np
import io
import os
import soundfile as sf
from pathlib import Path
import pyloudnorm as pyln
import whisper
import jiwer
import re

# Pre-set CUDA execution provider library path for ORT
os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib/ollama/cuda_v12:" + os.environ.get("LD_LIBRARY_PATH", "")

@pytest.fixture(scope="module")
def whisper_model():
    # Load base model for accurate local GPU execution in test suite
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("base", device=device)

BENCHMARK_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Wait! Is the west warehouse door locked from the inside? Check it immediately.",
    "Under this section, we must maintain silence.",
    "The target is approaching the coordinates now, stay alert.",
    "Voice synthesis technology has improved significantly in recent years.",
    "Please adjust the output volume to a comfortable level.",
    "This is a test of the emergency broadcast system.",
    "Welcome to the simulation, please choose your character.",
    "Verify that all systems are operational and running within standard parameters.",
    "The project successfully passed all verification gates and is ready."
]

def is_engine_available(engine: str) -> bool:
    if engine == "kokoro":
        return Path("models/kokoro/onnx/model.onnx").exists()
    elif engine == "piper":
        return Path("models/piper/en_US-lessac-medium.onnx").exists()
    elif engine == "f5_tts":
        return Path("models/f5_tts/model_1250000.safetensors").exists()
    elif engine == "chatterbox":
        return Path("models/chatterbox/config.json").exists()
    elif engine == "dia":
        return Path("models/dia/config.json").exists()
    elif engine == "fish":
        return Path("models/fish_audio").exists()
    return False

AVAILABLE_ENGINES = [e for e in ["kokoro", "piper", "f5_tts", "chatterbox", "dia", "fish"] if is_engine_available(e)]

def clean_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

@pytest.mark.parametrize("engine", AVAILABLE_ENGINES)
def test_engine_audio_quality(engine, whisper_model):
    from app.services.tts_service import get_worker
    worker = get_worker(engine)
    
    # Run the quality check for all benchmark prompts
    for prompt in BENCHMARK_PROMPTS:
        # 1. Synthesize audio via the worker
        wav_bytes, sr = worker.synthesize(prompt, voice_id="default")
        
        # 2. Post-process through the AudioPipeline (resampling, silence trim, normalization, clipping prevention)
        from app.audio.pipeline import AudioPipeline
        pipeline = AudioPipeline()
        processed_wav_bytes, processed_sr = pipeline.process_wav_bytes(wav_bytes, sr)
        
        # 3. Load processed audio
        audio_data, samplerate = sf.read(io.BytesIO(processed_wav_bytes))
        assert samplerate == 24000, f"Expected sample rate of 24000Hz, got {samplerate}Hz"
        
        # 4. Check Loudness (-23 LUFS integrated ± 1.0)
        meter = pyln.Meter(samplerate)
        if len(audio_data.shape) == 1:
            audio_2d = audio_data[:, np.newaxis]
        else:
            audio_2d = audio_data
        loudness = meter.integrated_loudness(audio_2d)
        assert -24.0 <= loudness <= -22.0, f"Loudness is {loudness} LUFS, expected -23 LUFS ±1.0"
        
        # 5. Check Clipping Prevention (Peak amplitude <= 0.95)
        peak = np.max(np.abs(audio_data))
        assert peak <= 0.95, f"Peak amplitude is {peak}, expected <= 0.95 to prevent clipping"
        
        # 6. Whisper ASR Word Error Rate (WER) Check (< 5%)
        audio_fp32 = audio_data.astype(np.float32)
        result = whisper_model.transcribe(audio_fp32, fp16=False)
        transcription = result["text"].strip()
        
        cleaned_prompt = clean_text(prompt)
        cleaned_transcription = clean_text(transcription)
        
        # Compute Word Error Rate
        wer = jiwer.wer(cleaned_prompt, cleaned_transcription)
        assert wer < 0.05, f"ASR Word Error Rate for engine '{engine}' is {wer:.2%} (expected < 5%). Prompt: '{prompt}' | Transcription: '{transcription}'"
