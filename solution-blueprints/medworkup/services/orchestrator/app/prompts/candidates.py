# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Prompt: generate candidate differential diagnoses from categorized entities.
"""

from __future__ import annotations

import json
from typing import List

SYSTEM = """\
You are an experienced clinical physician. Your task is to generate a differential
diagnosis list from structured clinical findings.
Output must be valid JSON only — no prose, no markdown.
"""


def build_messages(
    symptoms: List[str],
    findings: List[str],
    diseases_mentioned: List[str],
) -> List[dict]:
    user = f"""\
Given the following clinical information, generate a differential diagnosis list.

Symptoms (patient-reported):
{json.dumps(symptoms, ensure_ascii=False)}

Objective findings:
{json.dumps(findings, ensure_ascii=False)}

Diseases explicitly mentioned in the note:
{json.dumps(diseases_mentioned, ensure_ascii=False)}

Generate 3–8 candidate diagnoses that could explain the clinical picture.
Prioritise conditions consistent with ALL provided information.

Return ONLY a JSON array:
[
  {{"name": "Diagnosis Name", "initial_rationale": "one sentence"}},
  ...
]

Rules:
- Do not include diagnoses unsupported by the provided data.
- If a disease is already mentioned, include it as a candidate.
- Order roughly by clinical plausibility (most likely first).
"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
