# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from typing import List, Optional

from app.config import settings
from app.schemas import RawEntity

from .base import BaseClient, ServiceError

logger = logging.getLogger(__name__)


class NERClient(BaseClient):
    def __init__(self) -> None:
        super().__init__(settings.ner_base_url, "ner-service", settings.ner_timeout)

    async def extract(
        self,
        text: str,
        model: str,
        threshold: Optional[float] = None,
        labels: Optional[List[str]] = None,
    ) -> List[RawEntity]:
        """
        Call a NER endpoint and return extracted entities.

        Parameters
        ----------
        text      : Normalised clinical text to annotate.
        model     : NER endpoint name, e.g. "gliner-biomed".
        threshold : Optional confidence threshold.
        labels    : Optional entity type labels.

        Returns an empty list on service failure so the pipeline can continue.
        """
        payload: dict = {"text": text}
        if threshold is not None:
            payload["threshold"] = threshold
        if labels is not None:
            payload["labels"] = labels

        try:
            data = await self._post(f"/ner/{model}", payload)
        except ServiceError as exc:
            logger.error("[NER/%s] Service error – returning empty entity list: %s", model, exc)
            return []

        raw_entities = data.get("entities")
        if not isinstance(raw_entities, list):
            logger.error(
                "[NER/%s] Unexpected response format – 'entities' missing or not a list. Got: %s",
                model,
                str(data)[:200],
            )
            return []

        entities: List[RawEntity] = []
        for i, e in enumerate(raw_entities):
            try:
                entities.append(
                    RawEntity(
                        text=str(e["text"]),
                        label=str(e["label"]),
                        start=int(e["start"]),
                        end=int(e["end"]),
                        source=model,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "[NER/%s] Skipping malformed entity at index %d (%s): %s",
                    model,
                    i,
                    exc,
                    e,
                )

        logger.debug("[NER/%s] Parsed %d / %d entities", model, len(entities), len(raw_entities))
        return entities
