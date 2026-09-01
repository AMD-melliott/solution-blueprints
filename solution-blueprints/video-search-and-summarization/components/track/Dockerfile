# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Reuse the ROCm vLLM image: it already ships matched torch, torchvision and ROCm versions.
FROM vllm/vllm-openai-rocm:v0.22.0

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Install pinned Python dependencies from requirements.txt (torch/torchvision/ROCm
# come from the base image and are intentionally not listed there).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the detector weights into the image so there is no download at runtime.
# Ordered before COPY so that editing app.py does not invalidate this layer.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
COPY app.py .

ENV TRACK_MODEL=yolo11l.pt
EXPOSE 8102
ENTRYPOINT []
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8102"]
