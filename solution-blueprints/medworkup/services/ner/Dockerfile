# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# ─────────────────────────────────────────────
# Stage 1: dependency build
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed only for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Use the PyTorch CPU-only index so we get the +cpu wheel (~700 MB) instead
# of the default CUDA wheel (~2.5 GB). All other packages resolve from PyPI.
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ─────────────────────────────────────────────
# Stage 2: runtime image
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app
ENV HF_HOME=/models/hf_cache

# curl for health-check probes in docker-compose
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pre-built wheels – no compiler required in final image
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl \
 && rm -rf /wheels

COPY app/ ./app/

# Non-root user; pre-create cache dir so the volume mount is writable by appuser
RUN useradd -m appuser \
 && mkdir -p /models/hf_cache \
 && chown -R appuser:appuser /models/hf_cache
USER appuser

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
