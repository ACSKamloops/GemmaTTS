# ==============================================================================
# GemmaTTS — GPU-accelerated multi-service Dockerfile
# Base: NVIDIA CUDA 12.4.1 runtime on Ubuntu 22.04
# ==============================================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies + Python 3.12 (deadsnakes PPA for Ubuntu 22.04)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        git \
        ffmpeg \
        libsndfile1 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3.12-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Create non-root user
# ---------------------------------------------------------------------------
RUN groupadd -r gemmatts && useradd -r -g gemmatts -m -s /bin/bash gemmatts

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /opt/gemmatts

# ---------------------------------------------------------------------------
# Install Python dependencies (cached layer — only rebuilds when reqs change)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copy application code
# ---------------------------------------------------------------------------
COPY app/ ./app/
COPY scripts/ ./scripts/

# ---------------------------------------------------------------------------
# Create runtime directories and set ownership
# ---------------------------------------------------------------------------
RUN mkdir -p /opt/gemmatts/models \
             /opt/gemmatts/public/data/audio_cache \
    && chown -R gemmatts:gemmatts /opt/gemmatts

# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------
# SERVICE_NAME selects which service to launch: orchestrator | gemma | tts
ENV SERVICE_NAME=orchestrator \
    HOST=0.0.0.0 \
    ORCHESTRATOR_PORT=8000 \
    GEMMA_PORT=8001 \
    TTS_PORT=8002 \
    # Model paths (inside the shared volume)
    GEMMA_MODEL_PATH=/opt/gemmatts/models/gemma \
    # HuggingFace token (set at runtime)
    HF_TOKEN="" \
    # Default TTS engine
    DEFAULT_TTS_ENGINE=kokoro \
    # Security
    SECRET_KEY="" \
    # Python
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# Expose all three service ports
# ---------------------------------------------------------------------------
EXPOSE 8000 8001 8002

# ---------------------------------------------------------------------------
# Health check — hits whichever service is running on port 8000 by default
# Overridden in docker-compose per service
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# Switch to non-root user
# ---------------------------------------------------------------------------
USER gemmatts

# ---------------------------------------------------------------------------
# Entrypoint — dynamically selects the service module and port
# ---------------------------------------------------------------------------
COPY --chown=gemmatts:gemmatts <<'ENTRYPOINT_SCRIPT' /opt/gemmatts/entrypoint.sh
#!/usr/bin/env bash
set -euo pipefail

case "${SERVICE_NAME}" in
    orchestrator)
        MODULE="app.services.orchestrator_api:app"
        PORT="${ORCHESTRATOR_PORT}"
        ;;
    gemma)
        MODULE="app.services.gemma_service:app"
        PORT="${GEMMA_PORT}"
        ;;
    tts)
        MODULE="app.services.tts_service:app"
        PORT="${TTS_PORT}"
        ;;
    *)
        echo "ERROR: Unknown SERVICE_NAME '${SERVICE_NAME}'. Use: orchestrator | gemma | tts"
        exit 1
        ;;
esac

echo "Starting ${SERVICE_NAME} service on ${HOST}:${PORT} ..."
exec python -m uvicorn "${MODULE}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level info
ENTRYPOINT_SCRIPT

RUN chmod +x /opt/gemmatts/entrypoint.sh

ENTRYPOINT ["/opt/gemmatts/entrypoint.sh"]
