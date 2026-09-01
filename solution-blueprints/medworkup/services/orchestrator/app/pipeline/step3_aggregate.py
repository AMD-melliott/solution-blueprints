# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Step 3 – Entity Aggregation

  1. Sort entities by start offset.
  2. Merge overlapping or adjacent spans from different models.
  3. Deduplicate by normalised text + label.
  4. Preserve source provenance.
"""

from __future__ import annotations

import re
from typing import List

from app.schemas import AggregatedEntity, RawEntity


def _normalize_for_comparison(text: str) -> str:
    """Lowercase + strip punctuation. Used only for dedup, never shown to users."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _spans_overlap(a: RawEntity, b: RawEntity, tolerance: int = 2) -> bool:
    return not (a.end + tolerance < b.start or b.end + tolerance < a.start)


def aggregate_entities(raw: List[RawEntity]) -> List[AggregatedEntity]:
    if not raw:
        return []

    # ── merge overlapping spans ────────────────────────────────────────────────
    sorted_entities = sorted(raw, key=lambda e: (e.start, -(e.end - e.start)))

    merged: List[RawEntity] = []
    for entity in sorted_entities:
        if (
            merged
            and _spans_overlap(merged[-1], entity)
            and merged[-1].label == entity.label
            and _normalize_for_comparison(merged[-1].text) == _normalize_for_comparison(entity.text)
        ):
            prev = merged[-1]
            # Keep the longer span; accumulate sources below
            if (entity.end - entity.start) > (prev.end - prev.start):
                merged[-1] = entity
        else:
            merged.append(entity)

    # ── deduplicate by normalised text + label, merging sources ───────────────
    seen: dict[tuple, AggregatedEntity] = {}

    for entity in merged:
        key = (_normalize_for_comparison(entity.text), entity.label)
        if key in seen:
            agg = seen[key]
            if entity.source not in agg.sources:
                agg.sources.append(entity.source)
            # If any mention is non-negated, mark the cluster as non-negated
            if not entity.negated:
                agg.negated = False
        else:
            seen[key] = AggregatedEntity(
                text=entity.text,
                normalized_text=key[0],
                label=entity.label,
                start=entity.start,
                end=entity.end,
                sources=[entity.source],
                negated=entity.negated,
            )

    return list(seen.values())
