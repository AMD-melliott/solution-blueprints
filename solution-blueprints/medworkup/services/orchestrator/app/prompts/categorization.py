# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Prompt: classify a list of clinical entities into categories.
"""

from __future__ import annotations

import json
from typing import List

SYSTEM = """\
You are a clinical NLP assistant specialized in medical entity classification.
Your output must be valid JSON and nothing else — no prose, no markdown fences.
"""


def build_messages(entities: List[str]) -> List[dict]:
    entity_list = json.dumps(entities, ensure_ascii=False)
    user = f"""\
Classify each of the following clinical entities into exactly one category.

Categories:
- symptoms     : patient-reported complaints or sensations
- diseases     : named conditions or diagnoses
- findings     : objective clinical observations, lab/imaging results
- procedures   : tests, interventions, medications administered
- other        : anything that does not fit above

Entities:
{entity_list}

Return ONLY a JSON object with this exact structure:
{{
  "symptoms":   ["entity", ...],
  "diseases":   ["entity", ...],
  "findings":   ["entity", ...],
  "procedures": ["entity", ...],
  "other":      ["entity", ...]
}}

Rules:
- Every input entity must appear in exactly one list.
- Do not add, rename, or omit entities.
- Do not include explanations outside the JSON.
"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
