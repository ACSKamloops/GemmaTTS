#!/bin/bash
cd /home/astraithious/gemma4tts
source .venv/bin/activate
python -m pytest tests/test_cli_adapter.py -v --tb=short 2>&1
echo "EXIT_CODE=$?"
