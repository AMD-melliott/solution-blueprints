# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Step 6 – Candidate Diagnosis Generation

Hybrid approach:
  1. LLM generates candidates from symptoms + findings.
  2. Any diseases already mentioned in the note are guaranteed to be included.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.clients.llm_client import LLMClient
from app.prompts import candidates as cand_prompt
from app.schemas import CategorizedEntities

logger = logging.getLogger(__name__)


async def generate_candidates(
    categorized: CategorizedEntities,
    client: LLMClient,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: [{"name": str, "initial_rationale": str}, ...]
    """
    symptoms = categorized.symptoms
    findings = categorized.findings
    diseases = categorized.diseases

    if not symptoms and not findings and not diseases:
        logger.warning("[Candidates] No clinical entities available – skipping LLM call.")
        return []

    messages = cand_prompt.build_messages(symptoms, findings, diseases)

    try:
        raw = await client.chat_json(messages, temperature=0.2, max_tokens=1024)
    except ValueError as exc:
        logger.error("[Candidates] LLM parse error: %s", exc)
        # Fall back: at least return the explicitly mentioned diseases
        return [{"name": d, "initial_rationale": "Explicitly mentioned in note."} for d in diseases]

    if not isinstance(raw, list):
        logger.error("[Candidates] Expected list, got %s", type(raw))
        return [{"name": d, "initial_rationale": "Explicitly mentioned in note."} for d in diseases]

    # Normalise entries
    candidates: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for item in raw:
        name = str(item.get("name", "")).strip()
        rationale = str(item.get("initial_rationale", "")).strip()
        if name and name not in seen_names:
            candidates.append({"name": name, "initial_rationale": rationale})
            seen_names.add(name)

    # Guarantee any NER-detected diseases appear as candidates
    for d in diseases:
        if d not in seen_names:
            candidates.insert(0, {"name": d, "initial_rationale": "Explicitly mentioned in note."})
            seen_names.add(d)

    logger.info("[Candidates] Generated %d candidates", len(candidates))
    return candidates
