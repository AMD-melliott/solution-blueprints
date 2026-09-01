// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Thin wrappers around middleware REST endpoints.
// SSE stream is handled separately in hooks/useStreamEvents.ts

import type { ScoredTransaction, StreamStatus, Thresholds, WarmupStatus } from "../types/api";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

async function patchJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} ${res.status}`);
  return res.json();
}

export interface BatchScoreResponse {
  results: ScoredTransaction[];
  n_transactions: number;
}

export const api = {
  base: BASE,
  health: () => getJson<{ status: string; scorer_mode: string | null }>("/health"),
  warmup: () => getJson<WarmupStatus>("/warmup/status"),
  status: () => getJson<StreamStatus>("/stream/status"),
  start: () => postJson<StreamStatus>("/stream/start"),
  pause: () => postJson<StreamStatus>("/stream/pause"),
  stop: () => postJson<StreamStatus>("/stream/stop"),
  setThresholds: (t: Thresholds) => patchJson<StreamStatus>("/config/thresholds", t),
  batchScore: (count: number, mode = "xgb_ensemble") =>
    postJson<BatchScoreResponse>(`/batch/score?count=${count}&mode=${mode}`),
};
