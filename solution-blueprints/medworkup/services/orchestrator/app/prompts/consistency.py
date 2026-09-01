# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Prompt: consistency check – second LLM pass to detect contradictions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

SYSTEM = """\
You are a clinical quality-assurance reviewer. Your job is to flag inconsistencies
between the proposed diagnoses and the clinical evidence.
Output valid JSON only.
"""


def build_messages(
    categorized_entities: Dict[str, List[str]],
    diagnoses: List[Dict[str, Any]],
) -> List[dict]:
    user = f"""\
## Clinical Entities
{json.dumps(categorized_entities, indent=2, ensure_ascii=False)}

## Proposed Diagnoses
{json.dumps(diagnoses, indent=2, ensure_ascii=False)}

For each diagnosis, determine:
1. Is it consistent with the provided symptoms and findings?
2. Is any confidence score significantly over- or under-estimated given the evidence?

Return ONLY a JSON array (one entry per diagnosis, same order):
[
  {{
    "name": "...",
    "consistent": true,
    "confidence_adjustment": 0.0,
    "note": "optional one-sentence explanation if adjustment made"
  }},
  ...
]

Rules:
- confidence_adjustment is additive: +0.1 means increase by 0.1, -0.2 means decrease.
- Only adjust if there is a clear contradiction or strong supporting evidence missed.
- If consistent and confidence is reasonable, set adjustment to 0.0 and note to null.
- Do NOT rewrite diagnoses or evidence — only flag and adjust.
"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
