# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)

_VALID_DEVICES = frozenset({"cpu", "cuda", "auto"})


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """SapBERT via native HuggingFace Transformers with mean-pooling."""

    def __init__(self, model_id: str) -> None:
        hf_device = os.getenv("HF_DEVICE", "cpu").lower()
        if hf_device not in _VALID_DEVICES:
            raise RuntimeError(f"HF_DEVICE={hf_device!r} is invalid. Must be cpu, cuda, or auto.")

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("torch and transformers are required for HF mode.") from exc

        if hf_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = hf_device

        hf_home = os.getenv("HF_HOME", "/models/hf_cache")
        cache_marker = Path(hf_home) / "hub" / ("models--" + model_id.replace("/", "--"))
        is_cached = cache_marker.exists()

        logger.info("Model loading mode: HUGGINGFACE")
        if is_cached:
            logger.info("Loading cached model: %s", model_id)
        else:
            logger.info("Downloading model from HuggingFace: %s", model_id)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModel.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {model_id!r} from HuggingFace — "
                "check HUGGING_FACE_HUB_TOKEN and network connectivity."
            ) from exc

        self._model = self._model.to(device)
        self._model.eval()
        self._device = device

        logger.info("HuggingFace embedding provider ready (device=%s).", device)

    def embed(self, text: str) -> List[float]:
        import torch

        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)

        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
        attention_mask = encoding["attention_mask"]  # (1, seq_len)
        pooled = self._mean_pool(last_hidden, attention_mask)  # (1, hidden_dim)
        return pooled.cpu().squeeze().tolist()

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        import torch

        mask = attention_mask.unsqueeze(-1).float()  # (1, seq_len, 1)
        sum_hidden = (last_hidden_state * mask).sum(dim=1)  # (1, hidden_dim)
        count = mask.sum(dim=1).clamp(min=1e-9)  # (1, 1)
        return sum_hidden / count  # (1, hidden_dim)
