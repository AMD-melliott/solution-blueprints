# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Dict, List, Optional

from app.config import settings

from .base import BaseClient, ServiceError

logger = logging.getLogger(__name__)


class EmbeddingClient(BaseClient):
    def __init__(self) -> None:
        super().__init__(
            settings.embedding_base_url,
            "embedding-service",
            settings.embedding_timeout,
        )
        # In-process LRU-style cache keyed by SHA-256 of the input text.
        # Prevents re-embedding the same string within a request (or across
        # requests in the same process lifetime).
        # Replace with Redis for multi-instance deployments.
        self._cache: Dict[str, List[float]] = {}

    async def embed(self, text: str) -> Optional[List[float]]:
        """
        Embed a single text string.

        Returns the embedding vector, or None if the service is unavailable.
        Callers must handle None (e.g. skip clustering for that entity).
        """
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if cache_key in self._cache:
            logger.debug("[Embed] Cache hit for '%s...'", text[:40])
            return self._cache[cache_key]

        try:
            data = await self._post("/embed", {"text": text})
        except ServiceError as exc:
            logger.error("[Embed] Service error for '%s...': %s", text[:40], exc)
            return None

        embedding = data.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            logger.error(
                "[Embed] Unexpected response format for '%s...': %s",
                text[:40],
                str(data)[:200],
            )
            return None

        self._cache[cache_key] = embedding
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Embed a list of texts concurrently.

        Returns a list of the same length as `texts`.  Each element is either:
          - List[float]  – the embedding vector
          - None         – the embed call failed for that text (logged above)

        Individual failures do NOT abort the batch; the pipeline degrades
        gracefully (unembedded entities stay as independent clusters).
        """
        if not texts:
            return []

        # Kick off all coroutines concurrently; capture exceptions per-item
        results = await asyncio.gather(
            *[self.embed(t) for t in texts],
            return_exceptions=True,
        )

        output: List[Optional[List[float]]] = []
        for text, result in zip(texts, results):
            if isinstance(result, BaseException):
                logger.error("[Embed] Unexpected exception for '%s...': %s", text[:40], result)
                output.append(None)
            else:
                output.append(result)  # may itself be None from embed()

        failed = sum(1 for v in output if v is None)
        if failed:
            logger.warning("[Embed] %d / %d texts could not be embedded", failed, len(texts))

        return output

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()
