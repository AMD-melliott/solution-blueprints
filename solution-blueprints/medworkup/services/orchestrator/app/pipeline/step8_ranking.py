# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Step 8 – Ranking & Structuring

  1. Sort diagnoses by confidence descending.
  2. Clamp confidence to [0, 1].
  3. Filter out diagnoses below the configured minimum confidence threshold.
"""

from __future__ import annotations

import logging
from typing import List

from app.config import settings
from app.schemas import Diagnosis

logger = logging.getLogger(__name__)


def rank_diagnoses(diagnoses: List[Diagnosis]) -> List[Diagnosis]:
    if not diagnoses:
        return []

    # Clamp
    for d in diagnoses:
        d.confidence = max(0.0, min(1.0, d.confidence))

    # Filter
    min_conf = settings.diagnosis_min_confidence
    filtered = [d for d in diagnoses if d.confidence >= min_conf]

    if len(filtered) < len(diagnoses):
        dropped = len(diagnoses) - len(filtered)
        logger.info("[Ranking] Filtered %d low-confidence diagnoses (threshold=%.2f)", dropped, min_conf)

    # Sort
    ranked = sorted(filtered, key=lambda d: d.confidence, reverse=True)

    logger.info(
        "[Ranking] Final ranking: %s",
        [(d.name, round(d.confidence, 2)) for d in ranked],
    )
    return ranked
