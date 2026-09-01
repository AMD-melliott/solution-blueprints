# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class ONNXEmbeddingProvider(EmbeddingProvider):
    """SapBERT ONNX wrapper with mean-pooling."""

    def __init__(self, artifact_dir: str) -> None:
        root = Path(artifact_dir)

        if not root.exists():
            raise RuntimeError(
                f"EMBEDDING_ONNX_PATH={artifact_dir!r} does not exist. "
                "Create the directory or unset EMBEDDING_ONNX_PATH to use HF mode."
            )

        onnx_file = root / "model.onnx"
        if not onnx_file.exists():
            raise RuntimeError(
                f"ONNX artifact incomplete — model.onnx not found in {root}. "
                "Export the model or unset EMBEDDING_ONNX_PATH."
            )

        logger.info("Model loading mode: ONNX")
        logger.info("Loading model from %s", root)

        self._tokenizer = AutoTokenizer.from_pretrained(str(root))

        opts = ort.SessionOptions()
        threads = int(os.getenv("OMP_NUM_THREADS", "4"))
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(onnx_file),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}

        logger.info("ONNX embedding provider ready. Input names: %s", self._input_names)

    def embed(self, text: str) -> List[float]:
        encoding = self._tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=128,
            padding="max_length",
        )

        feed = {}
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in self._input_names and key in encoding:
                feed[key] = encoding[key].astype(np.int64)

        last_hidden: np.ndarray = self._session.run(None, feed)[0]
        attention_mask: np.ndarray = encoding["attention_mask"][0]
        embedding = self._mean_pool(last_hidden[0], attention_mask)
        return embedding.tolist()

    @staticmethod
    def _mean_pool(
        hidden_states: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        mask = attention_mask[:, np.newaxis].astype(np.float32)
        sum_hidden = (hidden_states * mask).sum(axis=0)
        count = mask.sum().clip(min=1e-9)
        return sum_hidden / count
