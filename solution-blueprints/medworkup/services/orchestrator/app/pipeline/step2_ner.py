# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Step 2 – NER Pipeline

Strategy: parallel execution of GLiNER-biomed and MedCAT, followed by
span-aware merging with confidence-based conflict resolution.

  ┌─────────────────────┐   ┌──────────────────────┐
  │  GLiNER-biomed      │   │  MedCAT              │
  │  (primary entity    │   │  (concept linking +  │
  │   extraction)       │   │   normalisation)     │
  └────────┬────────────┘   └──────────┬───────────┘
           │                           │
           └──────────┬────────────────┘
                      ▼
               merge_entities()
                      │
                      ▼
             deduplicated RawEntity list
             (negation-tagged)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.clients.ner_client import NERClient
from app.schemas import NegationSpan, PreprocessedText, RawEntity

logger = logging.getLogger(__name__)

_SOURCE_PRIORITY: Dict[str, int] = {
    "medcat": 3,
    "gliner-biomed": 2,
}

_GLINER_CONF_THRESHOLD = 0.35
_IOU_THRESHOLD = 0.5


@dataclass
class _Span:
    start: int
    end: int
    text: str
    label: str
    source: str
    confidence: float = 1.0
    cui: Optional[str] = None
    negated: bool = False
    meta: Dict = field(default_factory=dict)


def _iou(a: _Span, b: _Span) -> float:
    inter_start = max(a.start, b.start)
    inter_end = min(a.end, b.end)
    if inter_end <= inter_start:
        return 0.0
    inter = inter_end - inter_start
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union > 0 else 0.0


def _overlaps(a: _Span, b: _Span) -> bool:
    return _iou(a, b) >= _IOU_THRESHOLD


def _parse_gliner(raw: List[RawEntity]) -> List[_Span]:
    return [
        _Span(
            start=e.start,
            end=e.end,
            text=e.text,
            label=e.label,
            source="gliner-biomed",
            confidence=getattr(e, "confidence", 1.0) or 1.0,
        )
        for e in raw
    ]


def _parse_medcat(raw: List[RawEntity]) -> List[_Span]:
    return [
        _Span(
            start=e.start,
            end=e.end,
            text=e.text,
            label=e.label,
            source="medcat",
            confidence=getattr(e, "confidence", 1.0) or 1.0,
            cui=getattr(e, "cui", None),
            meta=getattr(e, "meta", {}) or {},
        )
        for e in raw
    ]


def _resolve_conflict(a: _Span, b: _Span) -> _Span:
    pri_a = _SOURCE_PRIORITY.get(a.source, 0)
    pri_b = _SOURCE_PRIORITY.get(b.source, 0)

    if pri_a != pri_b:
        winner, loser = (a, b) if pri_a > pri_b else (b, a)
    elif a.confidence != b.confidence:
        winner, loser = (a, b) if a.confidence >= b.confidence else (b, a)
    else:
        winner, loser = (a, b) if (a.end - a.start) >= (b.end - b.start) else (b, a)

    if winner.cui is None and loser.cui is not None:
        winner.cui = loser.cui
    return winner


def _merge_spans(all_spans: List[_Span]) -> List[_Span]:
    accepted: List[_Span] = []
    for candidate in sorted(all_spans, key=lambda s: (s.start, -(s.end - s.start))):
        conflicts = [i for i, a in enumerate(accepted) if _overlaps(a, candidate)]
        if not conflicts:
            accepted.append(candidate)
        else:
            best = candidate
            for i in reversed(conflicts):
                best = _resolve_conflict(best, accepted.pop(i))
            accepted.append(best)
    return sorted(accepted, key=lambda s: s.start)


def _to_raw_entity(span: _Span) -> RawEntity:
    return RawEntity(
        text=span.text,
        label=span.label,
        start=span.start,
        end=span.end,
        source=span.source,
        confidence=span.confidence,
        negated=span.negated,
        cui=span.cui,
    )


def _tag_negations(spans: List[_Span], negations: List[NegationSpan]) -> None:
    for span in spans:
        for neg in negations:
            if not (span.end <= neg.start or span.start >= neg.end):
                span.negated = True
                break


async def extract_entities(
    preprocessed: PreprocessedText,
    ner_client: NERClient,
    medcat_client=None,
) -> List[RawEntity]:
    """
    NER pipeline.

    Runs GLiNER-biomed and MedCAT in parallel. Merges results with
    span-aware deduplication and negation tagging.
    """
    text = preprocessed.normalized

    tasks: List[Tuple[str, asyncio.Task]] = []
    tasks.append(
        (
            "gliner-biomed",
            asyncio.create_task(ner_client.extract(text, "gliner-biomed", threshold=_GLINER_CONF_THRESHOLD)),
        )
    )

    if medcat_client is not None:
        tasks.append(("medcat", asyncio.create_task(medcat_client.process(text))))

    results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

    all_spans: List[_Span] = []
    for (source, _), result in zip(tasks, results):
        if isinstance(result, BaseException):
            logger.error("[NER/%s] failed: %s", source, result)
            continue

        logger.info("[NER/%s] returned %d entities", source, len(result))

        if source == "gliner-biomed":
            all_spans.extend(_parse_gliner(result))
        elif source == "medcat":
            all_spans.extend(_parse_medcat(result))

    if not all_spans:
        logger.warning("[NER] No entities extracted from any source.")
        return []

    merged = _merge_spans(all_spans)
    logger.info("[NER] %d spans after merging (from %d raw)", len(merged), len(all_spans))

    _tag_negations(merged, preprocessed.negations)

    return [_to_raw_entity(s) for s in merged]
