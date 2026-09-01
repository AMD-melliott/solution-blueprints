# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""BentoML service for the GNN fraud-scoring model."""

from __future__ import annotations

import pathlib

import bentoml
from predictor_gnn import GNNPredictor
from pydantic import BaseModel

MODEL_NAME = "fraud-gnn"

# config.yaml lives one level up (backend/), resolve at import time
_CFG_PATH = str(pathlib.Path(__file__).resolve().parent / "config.yaml")

_predictor: GNNPredictor | None = None


def _get_predictor() -> GNNPredictor:
    global _predictor
    if _predictor is None:
        _predictor = GNNPredictor(config_path=_CFG_PATH)
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
class GNNService:
    def __init__(self):
        self._predictor = _get_predictor()

    @bentoml.api(route=f"/models/{MODEL_NAME}/infer")
    def infer(self, request: InferRequest) -> InferResponse:
        by_name = {inp.name: inp.data for inp in request.inputs}

        transactions: list[dict] = by_name["transactions"]

        # Mode comes from the orchestrator as a single-element BYTES input.
        # Fallback: "isolated" (no graph edges).
        mode_data = by_name.get("mode")
        mode: str = str(mode_data[0]) if mode_data else "isolated"

        # Use single-tx score() for n=1 so the rolling graph is updated;
        # use score_batch() for n>1 to benefit from batched inference.
        if len(transactions) == 1:
            results = [self._predictor.score(transactions[0], mode=mode)]
        else:
            results = self._predictor.score_batch(transactions, mode=mode)

        n = len(results)
        probs = [r["fraud_probability"] for r in results]
        embeddings = [r["embedding"] for r in results]
        embed_dim = len(embeddings[0]) if embeddings else 64

        return InferResponse(
            outputs=[
                _TensorOutput(name="fraud_probability", shape=[n], datatype="FP32", data=probs),
                _TensorOutput(name="embedding", shape=[n, embed_dim], datatype="FP32", data=embeddings),
            ]
        )

    @bentoml.api(route=f"/models/{MODEL_NAME}/reset")
    def reset(self) -> dict:
        """Clear the rolling neighbour graph (called when the stream is stopped)."""
        self._predictor.reset_graph()
        return {"status": "ok"}
