# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Pipeline Orchestrator

Wires all pipeline steps together.  Each step is wrapped in a timer stage.
Intermediate outputs are logged in debug mode.
"""

from __future__ import annotations

import logging

from app.clients.embedding_client import EmbeddingClient
from app.clients.llm_client import LLMClient
from app.clients.medcat_client import MedCATClient
from app.clients.ner_client import NERClient
from app.config import settings
from app.schemas import AnalyzeResponse, PipelineMeta, StageTimer
from app.utils.timing import PipelineTimer

from .step1_preprocess import preprocess
from .step2_ner import extract_entities
from .step3_aggregate import aggregate_entities
from .step4_embeddings import normalize_entities
from .step5_categorize import categorize_entities
from .step6_candidates import generate_candidates
from .step7_reasoning import reason_over_candidates
from .step8_ranking import rank_diagnoses
from .step9_consistency import check_consistency

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Holds shared client instances and executes the pipeline.
    Instantiate once at application startup (lifespan).
    """

    def __init__(self) -> None:
        self.ner = NERClient()
        self.embedder = EmbeddingClient()
        self.llm = LLMClient()
        self.medcat = MedCATClient(base_url=settings.medcat_base_url)

    async def run(self, text: str) -> AnalyzeResponse:
        timer = PipelineTimer()
        debug = settings.debug_pipeline

        # ── Step 1: Advanced Preprocessing ───────────────────────────────────
        with timer.stage("step1_preprocess"):
            preprocessed = await preprocess(
                text,
                expand_abbreviations=settings.expand_abbreviations,
                use_medgemma=getattr(settings, "preprocess_use_medgemma", False),
                llm_client=self.llm,
            )
        logger.info(
            "[Step 1] sentences=%d sections=%s negations=%d abbrevs=%d",
            len(preprocessed.sentences),
            [k for k, v in preprocessed.section_map.model_dump().items() if v],
            len(preprocessed.negations),
            len(preprocessed.abbreviations_expanded),
        )
        if debug:
            logger.debug("[Step 1] Normalized text:\n%s", preprocessed.normalized[:500])

        # ── Step 2: Hybrid NER (GLiNER + MedCAT) ─────────────────────────────
        with timer.stage("step2_ner"):
            raw_entities = await extract_entities(
                preprocessed,
                ner_client=self.ner,
                medcat_client=self.medcat,
            )
        logger.info("[Step 2] Raw entities after hybrid merge: %d", len(raw_entities))
        if debug:
            for e in raw_entities:
                logger.debug(
                    "  [%s|%s] %s (neg=%s, cui=%s, conf=%.2f)",
                    e.source,
                    e.label,
                    e.text,
                    e.negated,
                    getattr(e, "cui", None),
                    getattr(e, "confidence", 1.0),
                )

        # ── Step 3: Aggregation ───────────────────────────────────────────────
        with timer.stage("step3_aggregate"):
            aggregated = aggregate_entities(raw_entities)
        logger.info("[Step 3] Aggregated entities: %d", len(aggregated))

        # ── Step 4: Semantic Normalization ────────────────────────────────────
        with timer.stage("step4_embeddings"):
            normalized = await normalize_entities(aggregated, self.embedder)
        logger.info("[Step 4] Normalized clusters: %d", len(normalized))

        # ── Step 5: Categorization ────────────────────────────────────────────
        with timer.stage("step5_categorize"):
            categorized = await categorize_entities(normalized, self.llm)
        logger.info(
            "[Step 5] symptoms=%d diseases=%d findings=%d procedures=%d",
            len(categorized.symptoms),
            len(categorized.diseases),
            len(categorized.findings),
            len(categorized.procedures),
        )

        # ── Step 6: Candidate Generation ──────────────────────────────────────
        with timer.stage("step6_candidates"):
            candidates = await generate_candidates(categorized, self.llm)
        logger.info("[Step 6] Candidates: %d", len(candidates))

        # ── Step 7: LLM Reasoning ─────────────────────────────────────────────
        with timer.stage("step7_reasoning"):
            diagnoses = await reason_over_candidates(
                original_note=preprocessed.normalized,
                categorized=categorized,
                candidates=candidates,
                client=self.llm,
            )
        logger.info("[Step 7] Diagnoses from reasoning: %d", len(diagnoses))

        # ── Step 8: Ranking ───────────────────────────────────────────────────
        with timer.stage("step8_ranking"):
            diagnoses = rank_diagnoses(diagnoses)
        logger.info("[Step 8] Diagnoses after ranking/filtering: %d", len(diagnoses))

        # ── Step 9: Consistency Check ──────────────────────────────────────────
        with timer.stage("step9_consistency"):
            diagnoses = await check_consistency(diagnoses, categorized, self.llm)
        logger.info("[Step 9] Final diagnoses: %d", len(diagnoses))

        # ── Assemble response ──────────────────────────────────────────────────
        timings = timer.timings()
        meta = PipelineMeta(
            stage_timings=[StageTimer(stage=k, duration_ms=v) for k, v in timings.items()],
            total_duration_ms=timer.total_ms(),
            entity_count_raw=len(raw_entities),
            entity_count_normalized=len(normalized),
        )

        return AnalyzeResponse(
            entities_raw=raw_entities,
            entities_normalized=normalized,
            categorized_entities=categorized,
            diagnoses=diagnoses,
            pipeline_meta=meta,
        )
