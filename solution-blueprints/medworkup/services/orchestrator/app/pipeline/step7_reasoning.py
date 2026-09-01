# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Step 7 – Constrained LLM Reasoning

Sends the original note + structured entities + candidates to the LLM.
The prompt enforces:
  - step-by-step reasoning
  - evidence grounded exclusively in provided data
  - structured JSON output
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.clients.llm_client import LLMClient
from app.prompts import reasoning as reasoning_prompt
from app.schemas import CategorizedEntities, Diagnosis

logger = logging.getLogger(__name__)


def _parse_diagnoses(raw: Dict[str, Any]) -> List[Diagnosis]:
    """
    Parse the LLM reasoning response into a list of Diagnosis objects.
    Handles missing / malformed fields defensively.
    """
    diagnoses_raw = raw.get("diagnoses", [])
    if not isinstance(diagnoses_raw, list):
        return []

    results: List[Diagnosis] = []
    for item in diagnoses_raw:
        try:
            name = str(item.get("name", "Unknown")).strip()
            confidence = float(item.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            evidence = item.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            evidence = [str(e) for e in evidence if e]

            reasoning_text = str(item.get("reasoning", "")).strip() or None

            results.append(
                Diagnosis(
                    name=name,
                    confidence=confidence,
                    evidence=evidence,
                    reasoning=reasoning_text,
                )
            )
        except Exception as exc:
            logger.warning("[Reasoning] Skipping malformed diagnosis entry %s: %s", item, exc)

    return results


async def reason_over_candidates(
    original_note: str,
    categorized: CategorizedEntities,
    candidates: List[Dict[str, Any]],
    client: LLMClient,
) -> List[Diagnosis]:
    if not candidates:
        logger.warning("[Reasoning] No candidates to reason over.")
        return []

    messages = reasoning_prompt.build_messages(
        original_note=original_note,
        categorized_entities=categorized.model_dump(),
        candidates=candidates,
    )

    try:
        raw = await client.chat_json(messages, temperature=0.1, max_tokens=2048)
    except ValueError as exc:
        logger.error("[Reasoning] LLM parse error: %s", exc)
        # Minimal fallback: return candidates as low-confidence diagnoses
        return [Diagnosis(name=c["name"], confidence=0.1, evidence=[], reasoning=None) for c in candidates]

    diagnoses = _parse_diagnoses(raw)
    logger.info(
        "[Reasoning] %d diagnoses parsed. Top: %s (%.2f)",
        len(diagnoses),
        diagnoses[0].name if diagnoses else "—",
        diagnoses[0].confidence if diagnoses else 0.0,
    )
    return diagnoses
