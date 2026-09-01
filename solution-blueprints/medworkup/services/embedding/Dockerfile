# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# ─────────────────────────────────────────────
# Stage 1: dependency build
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

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

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl \
 && rm -rf /wheels

COPY app/ ./app/

RUN useradd -m appuser \
 && mkdir -p /models/hf_cache \
 && chown -R appuser:appuser /models/hf_cache
USER appuser

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
