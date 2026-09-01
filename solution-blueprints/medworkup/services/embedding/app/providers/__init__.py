# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os

from .base import EmbeddingProvider
from .hf_provider import HuggingFaceEmbeddingProvider
from .onnx_provider import ONNXEmbeddingProvider

_DEFAULT_EMBEDDING_MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def build_embedding_provider() -> EmbeddingProvider:
    onnx_path = os.getenv("EMBEDDING_ONNX_PATH", "").strip()
    model_id = os.getenv("EMBEDDING_HF_MODEL_ID", _DEFAULT_EMBEDDING_MODEL_ID)

    if onnx_path:
        return ONNXEmbeddingProvider(onnx_path)
    return HuggingFaceEmbeddingProvider(model_id)


__all__ = ["EmbeddingProvider", "build_embedding_provider"]
