# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
#
# Build context must be backend/:
#   docker build -f xgb/Dockerfile -t fraud-xgb .

FROM rocm/dev-ubuntu-24.04:7.1.1-complete

# Prefer the base image's own (internally-consistent) ROCm 7.1.1 runtime libs
# over any copies bundled by pip wheels installed below (torch, onnxruntime-rocm).
# Without this, whichever ROCm-linked wheel installs last can silently shadow
# libamdhip64.so / libhsa-runtime64.so with a mismatched build, causing
# "undefined symbol ... version ROCR_1" at import time.
ENV LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY xgb/requirements.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# PyTorch ROCm wheels bundle libhipblas.so.2 and other ROCm runtime libs that
# onnxruntime-rocm depends on. XGBoost doesn't use torch for inference but we
# need these bundled libs for ORT's ROCMExecutionProvider to load.
#
# Install torch BEFORE amd-xgboost (matching train/Dockerfile.xgb-train's
# known-working order) — amd-xgboost must be the last ROCm-linked package
# installed so its matched libs are the ones actually resolved at runtime.
RUN pip install --no-cache-dir --break-system-packages torch \
        --index-url https://download.pytorch.org/whl/rocm7.1

# AMD-optimised XGBoost with ROCm GPU support
RUN pip install --no-cache-dir --break-system-packages amd-xgboost \
        --extra-index-url https://pypi.amd.com/rocm-7.1.1/simple

# ORT ROCm from AMD's own package index (same source as amd-xgboost).
# Falls back to the generic onnxruntime.ai build, then CPU-only, if not available.
RUN pip install --no-cache-dir --break-system-packages onnxruntime-rocm \
        --extra-index-url https://pypi.amd.com/rocm-7.1.1/simple \
    || pip install --no-cache-dir --break-system-packages onnxruntime-rocm \
        --extra-index-url https://download.onnxruntime.ai/ \
    || pip install --no-cache-dir --break-system-packages onnxruntime

COPY config.yaml ./
COPY xgb/featurize_xgb.py ./featurize.py
COPY xgb/xgb_service.py xgb/predictor_xgb.py ./

VOLUME ["/app/artifacts"]
EXPOSE 3001
CMD ["bentoml", "serve", "xgb_service:XGBService", "--host", "0.0.0.0", "--port", "3001"]
