#!/usr/bin/env python3
"""Run all service tests inline via pytest.main()."""
import os
import sys
import io

os.chdir("/home/astraithious/gemma4tts")
sys.path.insert(0, "/home/astraithious/gemma4tts")
os.environ["TEST_MODE"] = "true"

old_stdout = sys.stdout
old_stderr = sys.stderr
captured_out = io.StringIO()
captured_err = io.StringIO()
sys.stdout = captured_out
sys.stderr = captured_err

import pytest
exit_code = pytest.main([
    "tests/test_safety.py",
    "tests/test_signer.py",
    "tests/test_filesystem.py",
    "tests/test_tts_service.py",
    "tests/test_gemma_service.py",
    "-v", "--tb=short", "--color=no",
    "-p", "no:cacheprovider",
])

sys.stdout = old_stdout
sys.stderr = old_stderr

outpath = "/home/astraithious/gemma4tts/test_output.txt"
stdout_val = captured_out.getvalue()
stderr_val = captured_err.getvalue()

with open(outpath, "w") as f:
    f.write(f"STDOUT ({len(stdout_val)} chars):\n{stdout_val}\n")
    f.write(f"STDERR ({len(stderr_val)} chars):\n{stderr_val}\n")
    f.write(f"EXIT CODE: {exit_code}\n")

print(f"Done rc={exit_code}")
if stdout_val:
    lines = stdout_val.strip().split("\n")
    for line in lines[-25:]:
        print(line)
