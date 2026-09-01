# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
#
# Build context must be backend/:
#   docker build -f gnn/Dockerfile -t fraud-gnn .

FROM rocm/dev-ubuntu-24.04:7.1.1-complete

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY gnn/requirements.txt ./
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# PyTorch with ROCm support — adjust the index URL if the ROCm version changes
RUN pip install --no-cache-dir --break-system-packages torch \
        --index-url https://download.pytorch.org/whl/rocm7.1

# ORT ROCm from AMD's own package index (same source as amd-xgboost).
# Falls back to the generic onnxruntime.ai build, then CPU-only, if not available.
RUN pip install --no-cache-dir --break-system-packages onnxruntime-rocm \
        --extra-index-url https://pypi.amd.com/rocm-7.1.1/simple \
    || pip install --no-cache-dir --break-system-packages onnxruntime-rocm \
        --extra-index-url https://download.onnxruntime.ai/ \
    || pip install --no-cache-dir --break-system-packages onnxruntime

COPY config.yaml ./
COPY gnn/model.py ./model.py
COPY gnn/featurize_gnn.py ./featurize.py
COPY gnn/gnn_service.py gnn/predictor_gnn.py ./

VOLUME ["/app/artifacts"]
EXPOSE 3000
CMD ["bentoml", "serve", "gnn_service:GNNService", "--host", "0.0.0.0", "--port", "3000"]
