# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Step 4 – Semantic Normalization

  1. Embed each aggregated entity.
  2. Greedy cosine-similarity clustering.
  3. Pick the most-mentioned / longest variant as the canonical form.

Design note: greedy single-linkage clustering is simple and deterministic.
TODO: Replace with SNOMED/UMLS lookup in a future phase.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

from app.clients.embedding_client import EmbeddingClient
from app.config import settings
from app.schemas import AggregatedEntity, NormalizedEntity

logger = logging.getLogger(__name__)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _pick_canonical(texts: List[str]) -> str:
    """Choose the longest non-negated variant as canonical label."""
    return max(texts, key=len)


async def normalize_entities(
    aggregated: List[AggregatedEntity],
    client: EmbeddingClient,
) -> List[NormalizedEntity]:
    if not aggregated:
        return []

    texts = [e.text for e in aggregated]
    logger.info("[Embed] Embedding %d entities", len(texts))

    embeddings: List[Optional[List[float]]] = await client.embed_batch(texts)

    # Pair each entity with its embedding (None on failure)
    pairs: List[Tuple[AggregatedEntity, Optional[List[float]]]] = list(zip(aggregated, embeddings))

    threshold = settings.entity_similarity_threshold
    clusters: List[List[int]] = []  # list of index groups
    assigned = [False] * len(pairs)

    for i, (_, emb_i) in enumerate(pairs):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        if emb_i is None:
            clusters.append(cluster)
            continue
        for j in range(i + 1, len(pairs)):
            if assigned[j]:
                continue
            emb_j = pairs[j][1]
            if emb_j is not None and pairs[j][0].label == pairs[i][0].label and _cosine(emb_i, emb_j) >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    normalized: List[NormalizedEntity] = []
    for cluster in clusters:
        members = [pairs[idx][0] for idx in cluster]
        texts_in_cluster = [m.text for m in members]
        canonical = _pick_canonical(texts_in_cluster)
        label = members[0].label  # all members share same label after dedup
        negated = all(m.negated for m in members)

        normalized.append(
            NormalizedEntity(
                canonical=canonical,
                label=label,
                variants=list(dict.fromkeys(t for t in texts_in_cluster if t != canonical)),
                mention_count=len(members),
                negated=negated,
            )
        )

    logger.info("[Embed] %d entities → %d clusters", len(aggregated), len(normalized))
    return normalized
