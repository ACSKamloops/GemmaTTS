#!/usr/bin/env python3
"""
Download model weights for the Gemma 4 + TTS orchestration stack.

Chatterbox: Weights auto-downloaded by chatterbox-tts pip package (no manual download needed).
Dia 1.6B:   Downloaded from nari-labs/Dia-1.6B-0626 via HuggingFace Hub.
Kokoro 82M: Downloaded from onnx-community/Kokoro-82M-ONNX (ONNX model + voice embeddings).
Piper:      Downloaded from rhasspy/piper-voices (ONNX model + config).
Gemma 4:    Downloaded from google/gemma-4-E4B-it (gated, requires HF token).
F5-TTS:     Downloaded from SWivid/F5-TTS + charactr/vocos-mel-24khz.

Usage:
    python scripts/download_models.py                  # Download all models
    python scripts/download_models.py --test-only      # Download only test/lightweight models
    python scripts/download_models.py --skip-gemma     # Skip the large Gemma model
"""
import os
import sys
import argparse
import shutil
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("Error: 'huggingface_hub' package is required. Run 'pip install huggingface_hub' first.")
    sys.exit(1)

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = WORKSPACE_DIR / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Download model weights for Gemma 4 & TTS stack")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"),
                        help="Hugging Face token (needed for gated models like Gemma 4)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help=f"Directory to save weights (default: {DEFAULT_MODELS_DIR})")
    parser.add_argument("--test-only", action="store_true",
                        help="Download only lightweight test models (test gemma, kokoro, piper)")
    parser.add_argument("--skip-gemma", action="store_true", help="Skip downloading Gemma 4")
    parser.add_argument("--skip-dia", action="store_true", help="Skip downloading Dia 1.6B")
    parser.add_argument("--skip-f5", action="store_true", help="Skip downloading F5-TTS")
    return parser.parse_args()


def _get_token(token=None):
    """Resolve HF token from arg, env, or cached login."""
    if token:
        return token
    from huggingface_hub import get_token
    return get_token()


# ---- Gemma ----

def download_gemma(output_dir: Path, token: str) -> bool:
    """Download google/gemma-4-E4B-it (gated, ~16GB)."""
    print("Downloading Gemma 4 E4B-it...")
    token = _get_token(token)
    if not token:
        print("  WARNING: Gated model requires HF token. Set HF_TOKEN or use --token.")
        return False
    try:
        snapshot_download(
            repo_id="google/gemma-4-E4B-it",
            local_dir=output_dir / "gemma",
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "*.bin"],  # prefer safetensors
        )
        print("  ✓ Gemma 4 downloaded.")
        return True
    except Exception as e:
        print(f"  ✗ Gemma 4 download failed: {e}")
        return False


def download_gemma_test(output_dir: Path) -> bool:
    """Download lightweight test model for unit tests."""
    print("Downloading Gemma test fallback...")
    try:
        snapshot_download(
            repo_id="fxmarty/tiny-random-GemmaForCausalLM",
            local_dir=output_dir / "gemma_test",
        )
        print("  ✓ Gemma test model downloaded.")
        return True
    except Exception as e:
        print(f"  ✗ Gemma test download failed: {e}")
        return False


# ---- TTS Models ----

def download_dia(output_dir: Path, token: str) -> bool:
    """Download nari-labs/Dia-1.6B-0626 (~6.5GB)."""
    print("Downloading Dia 1.6B...")
    try:
        snapshot_download(
            repo_id="nari-labs/Dia-1.6B-0626",
            local_dir=output_dir / "dia",
            token=_get_token(token),
        )
        print("  ✓ Dia 1.6B downloaded.")
        return True
    except Exception as e:
        print(f"  ✗ Dia 1.6B download failed: {e}")
        return False


def download_kokoro(output_dir: Path) -> bool:
    """Download Kokoro 82M ONNX model and voice embeddings."""
    print("Downloading Kokoro 82M ONNX...")
    try:
        snapshot_download(
            repo_id="onnx-community/Kokoro-82M-ONNX",
            local_dir=output_dir / "kokoro",
            allow_patterns=["config.json", "onnx/model.onnx", "voices/*.bin", "voices/*.pt"],
        )
        print("  ✓ Kokoro 82M downloaded.")
        return True
    except Exception as e:
        print(f"  ✗ Kokoro download failed: {e}")
        return False


def download_piper(output_dir: Path) -> bool:
    """Download Piper en_US-lessac-medium voice model."""
    print("Downloading Piper voice model...")
    try:
        piper_dir = output_dir / "piper"
        piper_dir.mkdir(parents=True, exist_ok=True)

        files = [
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        ]
        for f in files:
            local_path = hf_hub_download(repo_id="rhasspy/piper-voices", filename=f)
            dest = piper_dir / Path(f).name
            shutil.copy(local_path, dest)

        print("  ✓ Piper voice downloaded.")
        return True
    except Exception as e:
        print(f"  ✗ Piper download failed: {e}")
        return False


def download_f5_tts(output_dir: Path) -> bool:
    """Download F5-TTS model checkpoint and Vocos vocoder."""
    print("Downloading F5-TTS...")
    try:
        f5_dir = output_dir / "f5_tts"
        f5_dir.mkdir(parents=True, exist_ok=True)

        local_model = hf_hub_download(
            repo_id="SWivid/F5-TTS",
            filename="F5TTS_v1_Base/model_1250000.safetensors",
        )
        shutil.copy(local_model, f5_dir / "model_1250000.safetensors")

        local_config = hf_hub_download(repo_id="charactr/vocos-mel-24khz", filename="config.yaml")
        local_vocab = hf_hub_download(repo_id="charactr/vocos-mel-24khz", filename="pytorch_model.bin")
        shutil.copy(local_config, f5_dir / "config.yaml")
        shutil.copy(local_vocab, f5_dir / "pytorch_model.bin")

        print("  ✓ F5-TTS downloaded.")
        return True
    except Exception as e:
        print(f"  ✗ F5-TTS download failed: {e}")
        return False


# ---- Verification ----

def verify_integrity(output_dir: Path, test_only: bool, skipped: set) -> bool:
    """Verify essential files exist for each model."""
    print("\nVerifying model file integrity...")
    ok = True

    checks = {
        "gemma_test": ["config.json"],
        "piper": ["en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"],
        "kokoro": {"onnx_candidates": ["onnx/model.onnx", "model.onnx"], "voice_dir": "voices"},
    }

    if not test_only:
        checks.update({
            "gemma": ["config.json", "tokenizer.json"],
            "dia": ["config.json"],
            "f5_tts": ["model_1250000.safetensors"],
        })

    for model_name, spec in checks.items():
        if model_name in skipped:
            continue

        model_path = output_dir / model_name
        if not model_path.exists():
            print(f"  ✗ {model_name}: directory missing")
            ok = False
            continue

        if model_name == "kokoro":
            # Special kokoro check — needs ONNX model + at least one voice file
            has_onnx = any((model_path / c).exists() for c in spec["onnx_candidates"])
            voice_dir = model_path / spec["voice_dir"]
            has_voices = voice_dir.exists() and any(voice_dir.iterdir())
            if has_onnx and has_voices:
                print(f"  ✓ {model_name}")
            else:
                if not has_onnx:
                    print(f"  ✗ {model_name}: ONNX model missing")
                if not has_voices:
                    print(f"  ✗ {model_name}: voice files missing")
                ok = False
        else:
            missing = [f for f in spec if not (model_path / f).exists()]
            if missing:
                print(f"  ✗ {model_name}: missing {missing}")
                ok = False
            else:
                print(f"  ✓ {model_name}")

    # Chatterbox uses pip package auto-download, no local files to check
    print("  ✓ chatterbox (weights managed by chatterbox-tts pip package)")

    return ok


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    skipped = set()

    # Always download lightweight models
    results["gemma_test"] = download_gemma_test(args.output_dir)
    results["kokoro"] = download_kokoro(args.output_dir)
    results["piper"] = download_piper(args.output_dir)

    if args.test_only:
        skipped.update(["gemma", "dia", "f5_tts"])
    else:
        token = _get_token(args.token)
        if args.skip_gemma:
            skipped.add("gemma")
        else:
            results["gemma"] = download_gemma(args.output_dir, token)
        if args.skip_dia:
            skipped.add("dia")
        else:
            results["dia"] = download_dia(args.output_dir, token)
        if args.skip_f5:
            skipped.add("f5_tts")
        else:
            results["f5_tts"] = download_f5_tts(args.output_dir)

    # Verify
    integrity_ok = verify_integrity(args.output_dir, args.test_only, skipped)
    all_ok = all(results.values()) and integrity_ok

    print()
    if all_ok:
        print("All downloads completed and verified successfully!")
        sys.exit(0)
    else:
        failed = [k for k, v in results.items() if not v]
        if failed:
            print(f"Failed downloads: {failed}")
        if not integrity_ok:
            print("Some verifications failed. Check messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
