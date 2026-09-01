<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# aimsb-video-search-and-summarization

Helm chart for the Video Search and Summarization solution blueprint: caption, index,
summarize, and answer questions about recorded video on AMD Instinct GPUs, with object
detection and tracking.

See [`docs/README.md`](docs/README.md) for the overview and architecture, and
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for deployment instructions.

## Layout

- `src/` - FastAPI backend and single-page UI, shipped via ConfigMap and run on a stock Python image.
- `components/` - Dockerfiles for the CPU embedding services (BGE-M3, open CLIP) and the GPU
  tracking service (YOLO11 + ByteTrack); built by the CI image pipeline.
- `templates/` - the app Deployment/Service/HTTPRoute/ConfigMap plus the in-chart VLM, embedding,
  and tracking Deployments/Services.
- Subcharts: `aimchart-llm` (LLM) and `aimchart-chromadb` (vector DB).

## Key configuration (`values.yaml`)

- `llm.image` - the LLM AIM image (default `amdenterpriseai/aim-qwen-qwen3-32b`).
- `vlm.image` / `vlm.model` - the vision-language model (default `Qwen/Qwen3-VL-8B-Instruct`).
- `embedText.image` / `embedImage.image` / `track.image` - the CI-built component images.
- `env_vars.CHUNK_SECONDS` / `env_vars.FRAMES_PER_CHUNK` - segmentation granularity.
- `http_route.enabled` - expose the UI via a Gateway API `HTTPRoute`.
