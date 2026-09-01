// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

/**
 * API types — mirror of services/orchestrator/app/schemas.py.
 *
 * This file is the contract between the UI and the orchestrator.
 * If schemas.py changes, change this file to match. Field names and
 * optionality are intentionally identical to the Pydantic models.
 */

// ── POST /analyze request ────────────────────────────────────────────────────

export interface AnalyzeRequest {
  text: string; // 1..50_000 chars
}

// ── Step 2 — NER ─────────────────────────────────────────────────────────────

export interface RawEntity {
  text: string;
  label: string;
  start: number;
  end: number;
  source: string; // "medmentions" | "ncbi" (free string on the wire)
  negated: boolean;
}

// ── Step 4 — semantic normalization ──────────────────────────────────────────
// Note: `embedding` is excluded server-side (Field(exclude=True)) — never sent.

export interface NormalizedEntity {
  canonical: string;
  label: string;
  variants: string[];
  mention_count: number;
  negated: boolean;
}

// ── Step 5 — categorization ──────────────────────────────────────────────────

export interface CategorizedEntities {
  symptoms: string[];
  diseases: string[];
  findings: string[];
  procedures: string[];
  other: string[];
}

// ── Steps 6-9 — diagnoses ────────────────────────────────────────────────────

export interface Diagnosis {
  name: string;
  confidence: number; // 0.0 .. 1.0
  evidence: string[];
  reasoning?: string | null;
  consistency_note?: string | null;
}

// ── Observability ────────────────────────────────────────────────────────────

export interface StageTimer {
  stage: string;
  duration_ms: number;
}

export interface PipelineMeta {
  stage_timings: StageTimer[];
  total_duration_ms: number;
  entity_count_raw: number;
  entity_count_normalized: number;
}

// ── POST /analyze response ───────────────────────────────────────────────────

export interface AnalyzeResponse {
  entities_raw: RawEntity[];
  entities_normalized: NormalizedEntity[];
  categorized_entities: CategorizedEntities;
  diagnoses: Diagnosis[];
  pipeline_meta?: PipelineMeta | null;
}

// ── Error envelope (FastAPI HTTPException detail) ─────────────────────────────

export interface ErrorResponse {
  error: string;
  stage?: string | null;
  detail?: unknown;
}

// ── GET /sys/health ──────────────────────────────────────────────────────────
// Shape from main.py sys_health(): per-service status + latency.

export type ServiceStatus = "ok" | "degraded" | "down";

export interface ServiceHealth {
  status: ServiceStatus;
  latency_ms: number;
  http_status?: number;
  error?: string;
}

export interface SysHealthResponse {
  status: "ok" | "degraded";
  services: {
    orchestrator: ServiceHealth;
    ner: ServiceHealth;
    embedding: ServiceHealth;
    llm: ServiceHealth;
  };
}
