# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""BentoML service for the XGBoost fraud-scoring model."""

from __future__ import annotations

import pathlib

import bentoml
import torch  # preloads bundled ROCm libs (libhipblas etc.) needed by onnxruntime-rocm
from predictor_xgb import XGBPredictor
from pydantic import BaseModel

MODEL_NAME = "fraud-xgb"

# config.yaml lives one level up (backend/), resolve at import time
_CFG_PATH = str(pathlib.Path(__file__).resolve().parent / "config.yaml")

_predictor: XGBPredictor | None = None


def _get_predictor() -> XGBPredictor:
    global _predictor
    if _predictor is None:
        _predictor = XGBPredictor(config_path=_CFG_PATH)
    return _predictor


class _TensorInput(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list


class _TensorOutput(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list


class InferRequest(BaseModel):
    inputs: list[_TensorInput]


class InferResponse(BaseModel):
    outputs: list[_TensorOutput]


@bentoml.service(name=MODEL_NAME)
class XGBService:
    def __init__(self):
        self._predictor = _get_predictor()

    @bentoml.api(route=f"/models/{MODEL_NAME}/infer")
    def infer(self, request: InferRequest) -> InferResponse:
        by_name = {inp.name: inp.data for inp in request.inputs}

        transactions: list[dict] = by_name["transactions"]
        embeddings: list[list[float]] = by_name.get("embeddings", [])

        results = self._predictor.score_batch(transactions, gnn_embeddings=embeddings)
        n = len(results)

        # Return [[p_clean, p_fraud], ...] — orchestrator reads index [1] for p_fraud
        probs = [[1.0 - r["fraud_probability"], r["fraud_probability"]] for r in results]

        return InferResponse(
            outputs=[
                _TensorOutput(name="probabilities", shape=[n, 2], datatype="FP32", data=probs),
            ]
        )
