# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Step 9 – Consistency Check

A second, cheaper LLM pass that:
  - verifies each diagnosis against the available clinical evidence
  - adjusts confidence scores for clear contradictions or missed support
  - attaches a brief explanatory note when an adjustment is made
"""

from __future__ import annotations

import logging
from typing import List

from app.clients.llm_client import LLMClient
from app.prompts import consistency as consistency_prompt
from app.schemas import CategorizedEntities, Diagnosis

logger = logging.getLogger(__name__)


async def check_consistency(
    diagnoses: List[Diagnosis],
    categorized: CategorizedEntities,
    client: LLMClient,
) -> List[Diagnosis]:
    if not diagnoses:
        return []

    messages = consistency_prompt.build_messages(
        categorized_entities=categorized.model_dump(),
        diagnoses=[d.model_dump(exclude={"reasoning", "consistency_note"}) for d in diagnoses],
    )

    try:
        adjustments = await client.chat_json(messages, temperature=0.0, max_tokens=1024)
    except ValueError as exc:
        logger.error("[Consistency] LLM parse error – skipping adjustments: %s", exc)
        return diagnoses

    if not isinstance(adjustments, list):
        logger.error("[Consistency] Expected list, got %s – skipping", type(adjustments))
        return diagnoses

    # Build lookup by name (case-insensitive)
    adj_map = {str(item.get("name", "")).strip().lower(): item for item in adjustments if isinstance(item, dict)}

    for diagnosis in diagnoses:
        adj = adj_map.get(diagnosis.name.lower())
        if adj is None:
            continue

        delta = float(adj.get("confidence_adjustment", 0.0))
        note: str | None = adj.get("note") or None

        if abs(delta) > 1e-4:
            old = diagnosis.confidence
            diagnosis.confidence = max(0.0, min(1.0, old + delta))
            logger.info(
                "[Consistency] %s: %.2f → %.2f (%+.2f) | %s",
                diagnosis.name,
                old,
                diagnosis.confidence,
                delta,
                note,
            )

        if note:
            diagnosis.consistency_note = note

    # Re-sort after adjustments
    diagnoses.sort(key=lambda d: d.confidence, reverse=True)
    return diagnoses
