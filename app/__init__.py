# App Package Initialization
import os
import sys

# Auto-reload with CUDA library paths in LD_LIBRARY_PATH before loading C extensions
if sys.platform.startswith("linux"):
    required_paths = [
        "/usr/local/lib/ollama/cuda_v12",
        "/usr/local/lib/ollama/mlx_cuda_v13",
        "/home/astraithious/gemma4tts/.venv/lib/python3.12/site-packages/nvidia/cu13/lib",
        "/home/astraithious/gemma4tts/.venv/lib/python3.12/site-packages/nvidia/cufft/lib",
        "/home/astraithious/gemma4tts/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/home/astraithious/gemma4tts/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib"
    ]

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths_in_ld = [p.strip() for p in ld_path.split(":") if p.strip()]

    needs_reload = False
    for path in required_paths:
        if path not in paths_in_ld:
            paths_in_ld.insert(0, path)
            needs_reload = True

    if needs_reload:
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths_in_ld)
        # Avoid infinite loop if somehow execv fails to update environment
        if os.environ.get("__LD_LIBRARY_PATH_RELOADED") != "1":
            os.environ["__LD_LIBRARY_PATH_RELOADED"] = "1"
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:
                # If execv fails, just print and continue
                sys.stderr.write(f"Warning: LD_LIBRARY_PATH reload failed: {e}\n")

