// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import type { SysHealthResponse, ServiceStatus } from "../types/api";

/**
 * Polls GET /sys/health on mount and every 30s (matches the legacy
 * pollSystemHealth cadence). Used by the header service pills.
 *
 * While a request is in flight on first load, services read as 'unknown'.
 * If the orchestrator is unreachable, all services read 'down'.
 */

export type PollState = ServiceStatus | "unknown";

export interface SystemHealthState {
  overall: "ok" | "degraded" | "unknown";
  ner: PollState;
  embedding: PollState;
  llm: PollState;
  orchestrator: PollState;
}

const UNKNOWN: SystemHealthState = {
  overall: "unknown",
  ner: "unknown",
  embedding: "unknown",
  llm: "unknown",
  orchestrator: "unknown",
};

const POLL_MS = 30_000;

export function useSystemHealth(): SystemHealthState {
  const [state, setState] = useState<SystemHealthState>(UNKNOWN);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const resp = await fetch("/sys/health", {
          signal: AbortSignal.timeout(10_000),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data: SysHealthResponse = await resp.json();
        if (cancelled) return;
        const s = data.services;
        setState({
          overall: data.status,
          ner: s.ner?.status ?? "down",
          embedding: s.embedding?.status ?? "down",
          llm: s.llm?.status ?? "down",
          orchestrator: s.orchestrator?.status ?? "down",
        });
      } catch {
        if (cancelled) return;
        setState({
          overall: "degraded",
          ner: "down",
          embedding: "down",
          llm: "down",
          orchestrator: "down",
        });
      }
    }

    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return state;
}
