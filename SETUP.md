# GemmaTTS — Setup Guide

Complete setup instructions for the GemmaTTS local voice agent stack.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Hardware Requirements](#hardware-requirements)
- [Model Memory Requirements](#model-memory-requirements)
- [Setup: Ubuntu / Debian (Native)](#setup-ubuntu--debian-native)
- [Setup: Windows + WSL2](#setup-windows--wsl2)
- [Setup: Docker](#setup-docker)
- [Environment Variables Reference](#environment-variables-reference)
- [Verifying the Installation](#verifying-the-installation)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Dependency     | Minimum Version | Notes                                        |
|----------------|-----------------|----------------------------------------------|
| NVIDIA GPU     | —               | ≥16 GB VRAM (RTX 3090/4090/5090 recommended) |
| NVIDIA Driver  | 535+            | `nvidia-smi` to check                        |
| CUDA Toolkit   | 12.x            | Bundled with PyTorch wheel                   |
| Python         | 3.11+           | 3.12 recommended                             |
| pip            | 23.0+           | `pip --version`                               |
| Git            | 2.30+           | For cloning                                  |
| ffmpeg         | 5.0+            | Audio format conversion                      |
| Docker Engine  | 24.0+           | Only for Docker setup                        |
| NVIDIA Container Toolkit | latest | Only for Docker setup                     |
| HuggingFace account | —          | Gated model access for Gemma                 |

---

## Hardware Requirements

| Component | Minimum              | Recommended            |
|-----------|----------------------|------------------------|
| GPU       | 16 GB VRAM (RTX 3090) | 32 GB VRAM (RTX 5090)  |
| RAM       | 32 GB                | 64 GB                  |
| Storage   | 50 GB free           | 100 GB free (SSD/NVMe) |
| CPU       | 8 cores              | 16+ cores              |

---

## Model Memory Requirements

| Model             | Disk Size | VRAM (Inference) | Notes                                     |
|-------------------|-----------|------------------|-------------------------------------------|
| Gemma 4 E4B-it    | ~16 GB    | ~14 GB           | BF16 precision; gated — requires HF token |
| Chatterbox (0.5B) | ~2 GB     | ~3 GB            | Auto-downloaded by pip on first use        |
| Dia 1.6B          | ~6 GB     | ~5 GB            | Requires `descript-audio-codec`            |
| Kokoro 82M (ONNX) | ~350 MB   | ~1 GB            | CPU-capable fallback via ONNX              |
| F5-TTS (1.2B)     | ~5 GB     | ~4 GB            | Voice cloning via reference audio          |
| Piper (Lessac)    | ~65 MB    | ~200 MB          | CPU-only emergency fallback                |
| **Total (all)**   | **~25 GB**| **~22 GB peak**  | Not all engines loaded simultaneously      |

> **Tip:** With `DEFAULT_TTS_ENGINE=kokoro` and Gemma loaded, peak VRAM is ~15 GB — fits comfortably on a 16 GB card. Multi-engine mode loads engines on-demand.

---

## Setup: Ubuntu / Debian (Native)

### 1. Install system dependencies

```bash
# Update package lists
sudo apt-get update && sudo apt-get upgrade -y

# Install essentials
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    software-properties-common

# Install Python 3.12 (if not already available)
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
```

### 2. Verify NVIDIA driver and CUDA

```bash
# Check GPU driver
nvidia-smi

# Expected output: Driver Version ≥ 535, CUDA Version ≥ 12.x
# If missing, install:
# sudo apt-get install -y nvidia-driver-545
```

### 3. Clone the repository

```bash
git clone https://github.com/YOUR_ORG/gemma4tts.git
cd gemma4tts
```

### 4. Create and activate virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate

# Verify
python --version  # Should print Python 3.12.x
```

### 5. Install Python dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 6. Configure environment

```bash
cp .env.example .env  # or create manually

# Edit .env with your values:
cat > .env << 'EOF'
HF_TOKEN=hf_your_huggingface_token_here
SECRET_KEY=your_random_secret_key_here
DEFAULT_TTS_ENGINE=kokoro
GEMMA_MODEL_PATH=models/gemma
EOF
```

### 7. Download model weights

```bash
# Set HuggingFace token (required for gated Gemma model)
export HF_TOKEN="hf_your_token_here"

# Download all models (~25 GB)
python scripts/download_models.py

# Or skip Gemma for TTS-only testing
python scripts/download_models.py --skip-gemma

# Or test models only (lightweight, CI-friendly)
python scripts/download_models.py --test-only
```

### 8. Start services

```bash
# Option A: Start all services in background
uvicorn app.services.orchestrator_api:app --host 0.0.0.0 --port 8000 &
uvicorn app.services.gemma_service:app    --host 0.0.0.0 --port 8001 &
uvicorn app.services.tts_service:app      --host 0.0.0.0 --port 8002 &

# Option B: Start in separate terminals (recommended for development)
# Terminal 1:
uvicorn app.services.gemma_service:app --host 0.0.0.0 --port 8001 --reload
# Terminal 2:
uvicorn app.services.tts_service:app --host 0.0.0.0 --port 8002 --reload
# Terminal 3:
uvicorn app.services.orchestrator_api:app --host 0.0.0.0 --port 8000 --reload
```

### 9. Verify

```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# Smoke test
python scripts/smoke_test.py
```

---

## Setup: Windows + WSL2

### 1. Enable WSL2 and install Ubuntu

```powershell
# In an elevated PowerShell prompt:
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2

# Reboot if prompted, then open the Ubuntu terminal to complete setup
```

### 2. Install NVIDIA GPU driver (Windows side)

Download and install the latest [NVIDIA Game Ready / Studio Driver](https://www.nvidia.com/download/index.aspx) for your GPU on the **Windows** host.

> **Important:** Do NOT install CUDA inside WSL2 separately — the Windows driver provides CUDA support automatically in WSL2.

### 3. Verify GPU passthrough in WSL2

```bash
# Inside WSL2 Ubuntu:
nvidia-smi

# You should see your Windows GPU listed with driver version and CUDA version
```

### 4. Install NVIDIA Container Toolkit (for Docker in WSL2)

```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 5. Follow the Ubuntu / Debian setup

From step 1 in the [Ubuntu / Debian (Native)](#setup-ubuntu--debian-native) section above, follow all remaining steps inside your WSL2 Ubuntu terminal.

### WSL2-specific tips

- **Access services from Windows:** Use `http://localhost:8000` — WSL2 forwards ports to the Windows host automatically.
- **File performance:** Keep the repo inside WSL2's filesystem (`/home/...`), not on `/mnt/c/` — I/O is dramatically faster.
- **Memory limit:** Create `%USERPROFILE%/.wslconfig` if WSL2 consumes too much RAM:
  ```ini
  [wsl2]
  memory=32GB
  processors=8
  ```

---

## Setup: Docker

### 1. Prerequisites

Ensure Docker Engine and NVIDIA Container Toolkit are installed (see [WSL2 step 4](#4-install-nvidia-container-toolkit-for-docker-in-wsl2) or [NVIDIA's install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

### 2. Configure environment

```bash
cd gemma4tts

# Create .env file for Docker Compose
cat > .env << 'EOF'
HF_TOKEN=hf_your_huggingface_token_here
SECRET_KEY=$(openssl rand -hex 32)
DEFAULT_TTS_ENGINE=kokoro
GEMMA_MODEL_ID=google/gemma-4-E4B-it
ORCHESTRATOR_PORT=8000
EOF
```

### 3. Download models (before building)

Models are stored in the `models/` directory which is bind-mounted into all containers:

```bash
# Option A: Download on host, then bind-mount
python scripts/download_models.py

# Option B: Use a one-shot container to download
docker compose run --rm gemma python scripts/download_models.py
```

### 4. Build and start

```bash
# Build the image and start all services
docker compose up --build -d

# Watch logs
docker compose logs -f

# Check service status
docker compose ps
```

### 5. Verify

```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# Test dialogue endpoint
curl -X POST http://localhost:8000/v1/dialogue \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, how are you?", "engine": "kokoro"}'
```

### Docker Compose commands reference

```bash
# Stop all services
docker compose down

# Rebuild after code changes
docker compose up --build -d

# View specific service logs
docker compose logs -f tts

# Scale (if not using GPU pinning)
docker compose up -d --scale tts=2

# Enter a running container
docker compose exec orchestrator bash

# Remove volumes (WARNING: deletes cached audio)
docker compose down -v
```

---

## Environment Variables Reference

| Variable               | Default                     | Description                                                 |
|------------------------|-----------------------------|-------------------------------------------------------------|
| `HF_TOKEN`             | *(required for Gemma)*      | HuggingFace API token for gated model downloads             |
| `SECRET_KEY`           | *(auto-generated)*          | HMAC-SHA256 key for signed audio URLs                       |
| `HOST`                 | `127.0.0.1`                 | Bind address for services (`0.0.0.0` in Docker)             |
| `ORCHESTRATOR_PORT`    | `8000`                      | Orchestrator API port                                       |
| `GEMMA_PORT`           | `8001`                      | Gemma text generation service port                          |
| `TTS_PORT`             | `8002`                      | TTS synthesis service port                                  |
| `GEMMA_MODEL_PATH`     | `models/gemma`              | Path to Gemma model weights directory                       |
| `GEMMA_MODEL_ID`       | `google/gemma-4-E4B-it`     | HuggingFace model ID for Gemma                              |
| `DEFAULT_TTS_ENGINE`   | `kokoro`                    | Default TTS engine: `chatterbox`, `dia`, `kokoro`, `f5_tts`, `piper` |
| `GEMMA_SERVICE_URL`    | `http://localhost:8001`     | Internal URL for Gemma service (set automatically in Docker)|
| `TTS_SERVICE_URL`      | `http://localhost:8002`     | Internal URL for TTS service (set automatically in Docker)  |
| `SIGNED_URL_EXPIRY_SECONDS` | `300`                  | Audio URL expiry time in seconds                            |
| `MAX_CACHE_SIZE_BYTES` | `52428800` (50 MB)          | Maximum audio cache size                                    |
| `MAX_TEXT_CHARS`        | `1000`                      | Maximum input text length (characters)                      |
| `MAX_TEXT_WORDS`        | `150`                       | Maximum input text length (words)                           |

---

## Verifying the Installation

### Quick health check

```bash
# All three services should return {"status": "ok"} or similar
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8001/health | python3 -m json.tool
curl -s http://localhost:8002/health | python3 -m json.tool
```

### Smoke test

```bash
python scripts/smoke_test.py
```

### Run test suite

```bash
python -m pytest tests/ -v
```

---

## Troubleshooting

### CUDA / GPU Issues

**Error: `CUDA out of memory`**
- Check current VRAM usage: `nvidia-smi`
- Close other GPU-intensive applications
- Use a lighter TTS engine: set `DEFAULT_TTS_ENGINE=piper`
- Reduce Gemma batch size or switch to a quantized model

**Error: `RuntimeError: No CUDA GPUs are available`**
- Verify driver: `nvidia-smi` (should show your GPU)
- In WSL2: Ensure the Windows NVIDIA driver is up-to-date
- In Docker: Confirm NVIDIA Container Toolkit is installed:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

**Error: `libcuda.so not found`**
- Missing NVIDIA driver or incorrect `LD_LIBRARY_PATH`
- In WSL2, do NOT install `cuda-toolkit` inside WSL — the driver comes from Windows

### Python / Dependency Issues

**Error: `ModuleNotFoundError: No module named 'torch'`**
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

**Error: `chatterbox-tts` installation fails**
- Ensure `ffmpeg` is installed: `sudo apt-get install ffmpeg`
- Ensure `libsndfile1` is installed: `sudo apt-get install libsndfile1`

**Error: `onnxruntime-gpu` fails to import**
- Ensure CUDA 12.x is available
- Try CPU fallback: `pip install onnxruntime` (instead of `-gpu`)

### Service / Network Issues

**Error: `Connection refused` on port 8001/8002**
- Start services in correct order: Gemma → TTS → Orchestrator
- Check if ports are in use: `ss -tlnp | grep -E '800[012]'`
- In Docker: verify network connectivity: `docker compose exec orchestrator curl http://gemma:8001/health`

**Error: `uvicorn: command not found`**
- Ensure venv is activated
- Install directly: `pip install uvicorn[standard]`

### Model Download Issues

**Error: `401 Unauthorized` downloading Gemma**
- Gemma is a gated model — accept the license at https://huggingface.co/google/gemma-4-E4B-it
- Set your token: `export HF_TOKEN=hf_your_token_here`
- Verify: `huggingface-cli whoami`

**Error: Download stalls or times out**
- Check disk space: `df -h .`
- Retry with resume: `python scripts/download_models.py` (idempotent)
- Use `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads:
  ```bash
  pip install hf-transfer
  HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/download_models.py
  ```

### Docker-Specific Issues

**Error: `docker: Error response from daemon: could not select device driver`**
- NVIDIA Container Toolkit not installed or not configured:
  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

**Error: Containers can't reach each other**
- Verify network: `docker network inspect gemmatts-net`
- Ensure services use container names (e.g., `http://gemma:8001`), not `localhost`

**Error: `Permission denied` on model volume**
- Fix ownership: `sudo chown -R 1000:1000 ./models`
- Or run with: `docker compose run --user root gemma chown -R gemmatts:gemmatts /opt/gemmatts/models`
