#!/bin/bash
set -e
cd /home/astraithious/gemma4tts
source .venv/bin/activate
export PYTHONPATH=/home/astraithious/gemma4tts
export TEST_MODE=true

echo "=== Running core tests ==="
python -m pytest tests/test_safety.py tests/test_signer.py tests/test_filesystem.py -v --tb=short
echo ""
echo "=== Running TTS service tests ==="
python -m pytest tests/test_tts_service.py -v --tb=short
echo ""
echo "=== Running Gemma service tests ==="
python -m pytest tests/test_gemma_service.py -v --tb=short
echo ""
echo "=== All test runs complete ==="
