// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useCallback, useRef, useState } from "react";
import type { AnalyzeResponse, ErrorResponse } from "../types/api";

/**
 * Wraps POST /analyze. Mirrors the legacy callAnalyze + runAnalysis flow:
 * sends { text }, surfaces loading/error, and reports wall-clock time.
 *
 * Results are cached by note id so re-selecting an analyzed note shows its
 * result instantly without re-running the (slow, costly) pipeline. The cache
 * lives here and is exposed so the sidebar (status pills) and pipeline column
 * can read it.
 */

export type AnalyzePhase = "idle" | "loading" | "success" | "error";

export interface AnalyzeState {
  phase: AnalyzePhase;
  result: AnalyzeResponse | null;
  error: string | null;
  wallMs: number | null;
}

const IDLE: AnalyzeState = { phase: "idle", result: null, error: null, wallMs: null };

async function postAnalyze(text: string): Promise<AnalyzeResponse> {
  const resp = await fetch("/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try {
      const j = (await resp.json()) as ErrorResponse & { detail?: unknown };
      const detail = j.detail;
      if (typeof detail === "string") msg = detail;
      else if (detail && typeof detail === "object") msg = JSON.stringify(detail);
      else if (j.error) msg = j.error;
    } catch {
      /* keep HTTP status message */
    }
    throw new Error(msg);
  }
  return resp.json() as Promise<AnalyzeResponse>;
}

export function useAnalyze() {
  const [state, setState] = useState<AnalyzeState>(IDLE);
  const cache = useRef<Map<number, AnalyzeResponse>>(new Map());

  const showCached = useCallback((noteId: number) => {
    const cached = cache.current.get(noteId);
    if (cached) {
      setState({
        phase: "success",
        result: cached,
        error: null,
        wallMs: cached.pipeline_meta?.total_duration_ms ?? null,
      });
    } else {
      setState(IDLE);
    }
  }, []);

  const run = useCallback(async (noteId: number, text: string) => {
    setState({ phase: "loading", result: null, error: null, wallMs: null });
    const t0 = performance.now();
    try {
      const result = await postAnalyze(text);
      const wallMs = Math.round(performance.now() - t0);
      cache.current.set(noteId, result);
      setState({
        phase: "success",
        result,
        error: null,
        wallMs: result.pipeline_meta?.total_duration_ms ?? wallMs,
      });
    } catch (err) {
      setState({
        phase: "error",
        result: null,
        error: err instanceof Error ? err.message : String(err),
        wallMs: null,
      });
    }
  }, []);

  const isCached = useCallback((noteId: number) => cache.current.has(noteId), []);

  return { state, run, showCached, isCached, cache };
}
