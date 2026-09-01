# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Step 5 – Entity Categorization

Sends canonical entities (non-negated) to the LLM for classification into:
symptoms / diseases / findings / procedures / other.

Negated entities are excluded from LLM input and placed in 'other' automatically.
"""

from __future__ import annotations

import logging
from typing import List

from app.clients.llm_client import LLMClient
from app.prompts import categorization as cat_prompt
from app.schemas import CategorizedEntities, NormalizedEntity

logger = logging.getLogger(__name__)

_VALID_KEYS = {"symptoms", "diseases", "findings", "procedures", "other"}


def _make_empty() -> CategorizedEntities:
    return CategorizedEntities()


async def categorize_entities(
    entities: List[NormalizedEntity],
    client: LLMClient,
) -> CategorizedEntities:
    if not entities:
        return _make_empty()

    positive = [e for e in entities if not e.negated]
    negated_names = [e.canonical for e in entities if e.negated]

    if not positive:
        return CategorizedEntities(other=negated_names)

    entity_names = [e.canonical for e in positive]
    messages = cat_prompt.build_messages(entity_names)

    try:
        raw = await client.chat_json(messages, temperature=0.0, max_tokens=1024)
    except ValueError as exc:
        # LLM returned unparseable JSON – fall back to 'other' for all
        logger.error("[Categorize] LLM parse error: %s", exc)
        return CategorizedEntities(other=entity_names + negated_names)

    # Validate and sanitise the returned structure
    result = _make_empty()
    for key in _VALID_KEYS:
        if key in raw and isinstance(raw[key], list):
            setattr(result, key, [str(v) for v in raw[key]])

    # Any entity the LLM dropped goes into 'other'
    accounted = {item for key in _VALID_KEYS for item in getattr(result, key)}
    dropped = [name for name in entity_names if name not in accounted]
    if dropped:
        logger.warning("[Categorize] LLM dropped %d entities, placing in 'other': %s", len(dropped), dropped)
        result.other.extend(dropped)

    # Append negated names to other
    result.other.extend(negated_names)

    logger.info(
        "[Categorize] symptoms=%d diseases=%d findings=%d procedures=%d other=%d",
        len(result.symptoms),
        len(result.diseases),
        len(result.findings),
        len(result.procedures),
        len(result.other),
    )
    return result
