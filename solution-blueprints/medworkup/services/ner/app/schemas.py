# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

_DEFAULT_BIOMED_LABELS: List[str] = [
    "disease",
    "symptom",
    "medication",
    "drug",
    "procedure",
    "anatomical_structure",
    "lab_test",
    "organism",
    "protein",
    "gene",
    "chemical_compound",
    "medical_device",
    "clinical_finding",
]


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000, description="Input clinical text")
    labels: Optional[List[str]] = Field(
        default=None,
        description=(
            "Entity type labels to extract. Defaults to standard biomedical set. "
            "GLiNER is zero-shot — any descriptive label string is valid."
        ),
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Span confidence threshold (0–1).",
    )

    def effective_labels(self) -> List[str]:
        return self.labels if self.labels else _DEFAULT_BIOMED_LABELS


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class NERResponse(BaseModel):
    entities: List[Entity]


class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
