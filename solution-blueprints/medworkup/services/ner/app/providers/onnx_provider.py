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


class ONNXNERProvider(NERProvider):
    """Loads GLiNER from a pre-downloaded local artifact directory."""

    def __init__(self, artifact_dir: str) -> None:
        root = Path(artifact_dir)

        if not root.exists():
            raise RuntimeError(
                f"NER_ONNX_PATH={artifact_dir!r} does not exist. "
                "Create the directory or unset NER_ONNX_PATH to use HF mode."
            )

        if not (root / "gliner_config.json").exists():
            raise RuntimeError(
                f"ONNX artifact directory incomplete — gliner_config.json not found in {root}. "
                "Re-run GLiNER.save_pretrained() or unset NER_ONNX_PATH."
            )

        num_threads = int(os.getenv("OMP_NUM_THREADS", "4"))
        try:
            import torch

            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(num_threads)
        except Exception:
            # Thread tuning is best-effort; keep provider initialization non-fatal.
            logger.warning(
                "Failed to configure torch thread settings (OMP_NUM_THREADS=%d). Continuing with defaults.",
                num_threads,
                exc_info=True,
            )

        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise ImportError("gliner package is not installed. Add 'gliner' to requirements.txt.") from exc

        logger.info("Model loading mode: ONNX")
        logger.info("Loading model from %s", root)
        self._model = GLiNER.from_pretrained(str(root), local_files_only=True)
        self._model.eval()
        logger.info("ONNX NER provider ready (threads=%d).", num_threads)

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
