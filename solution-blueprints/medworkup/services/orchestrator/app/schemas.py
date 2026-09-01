# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Shared Pydantic schemas used across the pipeline and API boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ── API boundary ──────────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50_000, description="Raw clinical note (SOAP or free text)")


class AnalyzeResponse(BaseModel):
    entities_raw: List["RawEntity"]
    entities_normalized: List["NormalizedEntity"]
    categorized_entities: "CategorizedEntities"
    diagnoses: List["Diagnosis"]
    pipeline_meta: Optional["PipelineMeta"] = None


# ── Step 1 – preprocessing ────────────────────────────────────────────────────


class SectionMap(BaseModel):
    subjective: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None
    unclassified: Optional[str] = None


class NegationSpan(BaseModel):
    text: str
    start: int
    end: int
    trigger: str  # the negation word/phrase detected


class PreprocessedText(BaseModel):
    original: str
    normalized: str
    sentences: List[str]
    section_map: SectionMap
    negations: List[NegationSpan]
    abbreviations_expanded: Dict[str, str]  # abbrev → expansion (what was substituted)


# ── Step 2 – NER ──────────────────────────────────────────────────────────────


class RawEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    source: str  # "gliner-biomed" | "medcat"
    negated: bool = False


# ── Step 3 – aggregation ──────────────────────────────────────────────────────


class AggregatedEntity(BaseModel):
    text: str
    normalized_text: str  # lowercase, stripped – for comparison only
    label: str
    start: int
    end: int
    sources: List[str]  # models that detected this
    negated: bool = False


# ── Step 4 – semantic normalization ──────────────────────────────────────────


class NormalizedEntity(BaseModel):
    canonical: str
    label: str
    variants: List[str]
    mention_count: int
    negated: bool = False
    embedding: Optional[List[float]] = Field(default=None, exclude=True)  # never serialised out


# ── Step 5 – categorization ───────────────────────────────────────────────────


class CategorizedEntities(BaseModel):
    symptoms: List[str] = []
    diseases: List[str] = []
    findings: List[str] = []
    procedures: List[str] = []
    other: List[str] = []


# ── Steps 6-9 – diagnoses ─────────────────────────────────────────────────────


class Diagnosis(BaseModel):
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[str]
    reasoning: Optional[str] = None
    consistency_note: Optional[str] = None


# ── observability ─────────────────────────────────────────────────────────────


class StageTimer(BaseModel):
    stage: str
    duration_ms: float


class PipelineMeta(BaseModel):
    stage_timings: List[StageTimer]
    total_duration_ms: float
    entity_count_raw: int
    entity_count_normalized: int


# ── error ─────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    error: str
    stage: Optional[str] = None
    detail: Optional[Any] = None
