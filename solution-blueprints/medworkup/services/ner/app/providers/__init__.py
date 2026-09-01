# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os

from .base import NERProvider
from .hf_provider import HuggingFaceNERProvider
from .onnx_provider import ONNXNERProvider

_DEFAULT_NER_MODEL_ID = "Ihor/gliner-biomed-large-v1.0"


def build_ner_provider() -> NERProvider:
    onnx_path = os.getenv("NER_ONNX_PATH", "").strip()
    model_id = os.getenv("NER_HF_MODEL_ID", _DEFAULT_NER_MODEL_ID)

    if onnx_path:
        return ONNXNERProvider(onnx_path)
    return HuggingFaceNERProvider(model_id)


__all__ = ["NERProvider", "build_ner_provider"]
