# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import List

from ..schemas import Entity
from .base import NERProvider

logger = logging.getLogger(__name__)


class HuggingFaceNERProvider(NERProvider):
    """Downloads and caches GLiNER from HuggingFace Hub."""

    def __init__(self, model_id: str) -> None:
        num_threads = int(os.getenv("OMP_NUM_THREADS", "4"))
        try:
            import torch

            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
        except Exception:
            logger.debug(
                "Skipping torch thread configuration; proceeding with default runtime settings.", exc_info=True
            )

        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise ImportError("gliner package is not installed. Add 'gliner' to requirements.txt.") from exc

        hf_home = os.getenv("HF_HOME", "/models/hf_cache")
        cache_marker = Path(hf_home) / "hub" / ("models--" + model_id.replace("/", "--"))
        is_cached = cache_marker.exists()

        logger.info("Model loading mode: HUGGINGFACE")
        if is_cached:
            logger.info("Loading cached model: %s", model_id)
        else:
            logger.info("Downloading model from HuggingFace: %s", model_id)

        try:
            self._model = GLiNER.from_pretrained(model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {model_id!r} from HuggingFace — "
                "check HUGGING_FACE_HUB_TOKEN and network connectivity."
            ) from exc

        self._model.eval()
        logger.info("HuggingFace NER provider ready (threads=%d).", num_threads)

    def predict(self, text: str, labels: List[str], threshold: float) -> List[Entity]:
        try:
            import torch

            ctx = torch.inference_mode()
        except ImportError:
            ctx = nullcontext()

        with ctx:
            raw = self._model.predict_entities(text, labels, threshold=threshold)

        entities = [
            Entity(
                text=span["text"],
                label=span["label"],
                start=span["start"],
                end=span["end"],
            )
            for span in raw
        ]
        entities.sort(key=lambda e: e.start)
        return entities
