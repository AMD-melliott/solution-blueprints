# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MedCAT Service Client

Wraps the CogStack medcat-service REST API.

OpenAPI contract (openapi.json):
  POST /api/process
    Request:  ProcessAPIInput  → {"content": {"text": "..."}, "meta_anns_filters": [...]}
    Response: ProcessAPIResponse → {"medcat_info": ..., "result": ProcessResult | ProcessErrorsResult}

  ProcessResult.annotations: List[Dict[str, Entity]] | {}
    Each list item is a dict keyed by entity ID string mapping to an Entity object.
    In practice the service returns a single dict {id: Entity}, hence the anyOf.

  Entity fields used:
    source_value  – original surface form in the note
    pretty_name   – normalised display name (fallback)
    cui           – UMLS/SNOMED concept ID
    type_ids      – semantic type IDs (e.g. ["T047"])
    acc           – confidence / accuracy score [0.0, 1.0]
    start / end   – character offsets
    meta_anns     – {name: MetaAnnotation} from MetaCAT models

  GET /api/health/live  → HealthCheckResponseContainer {"status": "UP"|"DOWN", "checks": [...]}
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
from app.schemas import RawEntity

logger = logging.getLogger(__name__)


class MedCATClient:
    """Async HTTP client for the medcat-service."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
        )

    async def process(
        self,
        text: str,
        meta_anns_filters: Optional[List] = None,
    ) -> List[RawEntity]:
        """
        POST /api/process — annotate a single document.

        Args:
            text: Clinical note text.
            meta_anns_filters: Optional MetaCAT filter list, e.g.
                [["Presence", ["True"]], ["Subject", ["Patient", "Family"]]]
                Passed through as-is; None omits the key entirely.

        Returns:
            Parsed RawEntity list. Empty on failure (graceful degradation).
        """
        # ── Build request payload (ProcessAPIInput) ───────────────────────────
        payload: Dict[str, Any] = {"content": {"text": text}}
        if meta_anns_filters is not None:
            payload["meta_anns_filters"] = meta_anns_filters

        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 2):
            try:
                resp = await self._client.post("/api/process", json=payload)
                resp.raise_for_status()
                return self._parse_response(resp.json())
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[MedCAT] attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries + 1,
                    exc,
                )

        logger.error("[MedCAT] all retries exhausted: %s", last_exc)
        return []

    async def health(self) -> bool:
        """
        GET /api/health/live → HealthCheckResponseContainer.
        Returns True only when top-level status is "UP".
        """
        try:
            resp = await self._client.get("/api/health/live", timeout=5.0)
            resp.raise_for_status()
            return resp.json().get("status") == "UP"
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()

    # ── Response parsing ──────────────────────────────────────────────────────

    def _parse_response(self, data: Dict[str, Any]) -> List[RawEntity]:
        """
        Unpack ProcessAPIResponse and delegate to annotation parser.

        Schema:
          ProcessAPIResponse
            medcat_info: ServiceInfo
            result: ProcessResult | ProcessErrorsResult

          ProcessResult
            text, annotations, success, timestamp, elapsed_time

          ProcessErrorsResult
            success: false, timestamp, errors: [str]
        """
        result = data.get("result", {})

        # Detect ProcessErrorsResult (success=False)
        if not result.get("success", True):
            errors = result.get("errors", [])
            logger.error("[MedCAT] service returned errors: %s", errors)
            return []

        # annotations: List[Dict[str, Entity]] | {} (see OpenAPI anyOf)
        annotations = result.get("annotations", [])
        return self._parse_annotations(annotations)

    def _parse_annotations(self, annotations: Any) -> List[RawEntity]:
        """
        Convert the annotations field to List[RawEntity].

        The OpenAPI spec declares annotations as:
          anyOf:
            - type: array, items: {additionalProperties: {$ref: Entity}}
            - {}   (bare dict fallback — same as a single-item list)

        Both cases are normalised to an iterable of {entity_id: Entity} dicts
        before flattening.
        """
        if not annotations:
            return []

        # Normalise: wrap a bare dict into a list so iteration is uniform
        if isinstance(annotations, dict):
            annotation_dicts = [annotations]
        elif isinstance(annotations, list):
            annotation_dicts = annotations
        else:
            logger.warning("[MedCAT] Unexpected annotations type: %s", type(annotations))
            return []

        out: List[RawEntity] = []
        for ann_dict in annotation_dicts:
            if not isinstance(ann_dict, dict):
                continue
            for entity_id, entity in ann_dict.items():
                parsed = self._parse_entity(entity_id, entity)
                if parsed is not None:
                    out.append(parsed)
        return out

    def _parse_entity(self, entity_id: str, ann: Dict[str, Any]) -> Optional[RawEntity]:
        """
        Convert a single Entity object (from the OpenAPI Entity schema) to RawEntity.

        Entity fields (from openapi.json → components.schemas.Entity):
          pretty_name, cui, type_ids, source_value, detected_name,
          acc, context_similarity, start, end, id, meta_anns,
          context_left, context_center, context_right
        """
        start = ann.get("start")
        end = ann.get("end")

        # Surface form: prefer source_value (exact match in note), then pretty_name
        text = ann.get("source_value") or ann.get("pretty_name") or ann.get("detected_name", "")

        if start is None or end is None or not text:
            logger.debug("[MedCAT] Skipping entity %s — missing span or text", entity_id)
            return None

        cui: Optional[str] = ann.get("cui")
        # acc is the primary confidence score per OpenAPI spec
        confidence = float(ann.get("acc", 1.0))

        # meta_anns: Dict[str, MetaAnnotation] — {name: {value, confidence, name}}
        # Store as-is for downstream use (e.g. Presence/Subject MetaCAT filters)
        meta_anns: Dict[str, Any] = ann.get("meta_anns") or {}

        return RawEntity(
            text=text,
            label=cui,
            start=int(start),
            end=int(end),
            source="medcat",
            confidence=confidence,
            negated=False,  # tagged later in step2 after span merge
            cui=cui,
            meta=meta_anns,
        )
