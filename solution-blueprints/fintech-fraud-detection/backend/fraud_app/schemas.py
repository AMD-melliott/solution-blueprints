# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from typing import Optional

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    """Raw transaction fields for scoring. Missing fields are imputed server-side."""

    TransactionAmt: float = Field(..., gt=0)
    ProductCD: Optional[str] = None
    card4: Optional[str] = None
    card6: Optional[str] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None

    card1: Optional[int] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card5: Optional[float] = None

    addr1: Optional[float] = None
    addr2: Optional[float] = None

    TransactionDT: Optional[float] = None

    C1: Optional[float] = None
    C2: Optional[float] = None
    C14: Optional[float] = None

    D1: Optional[float] = None
    D4: Optional[float] = None
    D10: Optional[float] = None
    D15: Optional[float] = None

    M4: Optional[str] = None
    M6: Optional[str] = None

    model_config = {"extra": "allow"}


class FraudScore(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description="APPROVE | REVIEW | DECLINE")
    model_version: str
    model_type: str = Field(..., description="xgb_ensemble | gnn_only | graph")
    rules_matched: list[dict] = Field(default_factory=list, description="Rules that fired, if any")


class BatchScoreRequest(BaseModel):
    transactions: list[TransactionIn] = Field(..., min_length=1, max_length=10_000)
    mode: str = Field(
        "xgb_ensemble",
        pattern="^(xgb_ensemble|gnn_only|graph)$",
    )


class BatchScoreResponse(BaseModel):
    results: list[FraudScore]
    n_transactions: int
