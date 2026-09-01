# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Prompt: constrained step-by-step LLM reasoning over candidates.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

SYSTEM = """\
You are a senior clinician performing a rigorous differential diagnosis review.
You MUST reason step-by-step and base ALL conclusions strictly on the provided data.
Do NOT introduce information not present in the clinical note or entity list.
Output must be valid JSON only.
"""


def build_messages(
    original_note: str,
    categorized_entities: Dict[str, List[str]],
    candidates: List[Dict[str, Any]],
) -> List[dict]:
    user = f"""\
## Clinical Note
{original_note}

## Extracted & Categorized Entities
{json.dumps(categorized_entities, indent=2, ensure_ascii=False)}

## Candidate Diagnoses to Evaluate
{json.dumps(candidates, indent=2, ensure_ascii=False)}

---

Perform the following steps IN ORDER and include them in your output:

1. **Summary**: Briefly summarise the key clinical findings from the note and entities.
2. **Evaluation**: For each candidate diagnosis, evaluate supporting and opposing evidence
   using ONLY information present above.
3. **Likelihood**: Assign a likelihood score (0.0–1.0) to each candidate.
4. **Reasoning**: For the top 3 candidates, provide one concise explanatory sentence.

Return ONLY a JSON object with this exact structure:
{{
  "summary": "...",
  "diagnoses": [
    {{
      "name": "...",
      "confidence": 0.0,
      "evidence": ["supporting fact from note", ...],
      "reasoning": "one sentence"
    }}
  ]
}}

Constraints:
- confidence values must sum to ≤ 1.0 (they are independent likelihoods, not a distribution)
- evidence strings must be direct references to provided data, not new clinical claims
- include ALL candidates in the output, even low-confidence ones
- do not fabricate symptoms or findings not present in the note
"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
