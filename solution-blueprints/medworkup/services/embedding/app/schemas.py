# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from typing import List

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, description="Input text to embed")


class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
