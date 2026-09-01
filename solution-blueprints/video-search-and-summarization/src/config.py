# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Configuration, sourced from the environment."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
VIDEOS_DIR = DATA_DIR / "videos"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
TRACKED_DIR = DATA_DIR / "tracked"
REPORTS_DIR = DATA_DIR / "reports"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "app.db"

# Inference services (OpenAI-compatible vLLM for VLM/LLM/embeddings).
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:30082/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:30081/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
# Embedding services are vLLM servers (aimchart-embedding) exposing an OpenAI-compatible
# /v1/embeddings endpoint. Text uses the plain "input" form; keyframe images use the
# chat-style "messages" form with an image_url (CLIP image encoder).
EMBED_TEXT_URL = os.environ.get("EMBED_TEXT_URL", "http://127.0.0.1:8100")
EMBED_TEXT_MODEL = os.environ.get("EMBED_TEXT_MODEL", "BAAI/bge-m3")
EMBED_IMAGE_URL = os.environ.get("EMBED_IMAGE_URL", "http://127.0.0.1:8101")
EMBED_IMAGE_MODEL = os.environ.get("EMBED_IMAGE_MODEL", "openai/clip-vit-base-patch32")
TRACK_URL = os.environ.get("TRACK_URL", "http://127.0.0.1:8102")
# When set (e.g. by the Helm chart pointing at aimchart-chromadb), use a remote ChromaDB
# over HTTP instead of the embedded on-disk client.
CHROMA_URL = os.environ.get("CHROMA_URL", "")

# Segmentation defaults (overridable at runtime via the settings API and recorded per video).
CHUNK_SECONDS = float(os.environ.get("CHUNK_SECONDS", "10"))
FRAMES_PER_CHUNK = int(os.environ.get("FRAMES_PER_CHUNK", "8"))

# Recursive summarization: number of consecutive segments grouped into one scene.
SCENE_FANIN = int(os.environ.get("SCENE_FANIN", "5"))

# Hybrid retrieval: per-modality candidate counts and fused result count.
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "20"))
HYBRID_TOP_N = int(os.environ.get("HYBRID_TOP_N", "12"))

FRAME_MAX_SIDE = int(os.environ.get("FRAME_MAX_SIDE", "768"))
UI_DIR = Path(os.environ.get("UI_DIR", "/app/ui_dist"))
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "300"))

# Default domain system prompt; overridable at runtime via the settings API.
DEFAULT_DOMAIN_PROMPT = (
    "You are a video analysis assistant. Describe and reason about video content accurately "
    "and specifically: people, vehicles, objects, scene type, on-screen text, actions, and "
    "notable events such as collisions or unusual activity. Always reference timestamps in "
    "seconds. Do not invent details that are not present."
)
