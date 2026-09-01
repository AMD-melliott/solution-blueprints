// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Polls the middleware /warmup/status endpoint for the readiness checks
// (dataset, backend orchestrator, GNN and XGBoost services). Polling never
// stops: fast while warming up, relaxed once ready, so a service that dies
// later flips the UI back to "warming up" without a page reload.

import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { WarmupStatus } from "../types/api";

const WARMUP_POLL_MS = 2000;
const READY_POLL_MS = 10000;

export interface WarmupState {
  status: WarmupStatus | null;
  ready: boolean;
  /** true when the middleware itself cannot be reached */
  unreachable: boolean;
}

export function useWarmupStatus(): WarmupState {
  const [status, setStatus] = useState<WarmupStatus | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      let delay = WARMUP_POLL_MS;
      try {
        const s = await api.warmup();
        if (cancelled) return;
        setStatus(s);
        setUnreachable(false);
        if (s.ready) delay = READY_POLL_MS;
      } catch {
        if (cancelled) return;
        setUnreachable(true);
      }
      timer = window.setTimeout(poll, delay);
    };
    poll();

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  // A stale green status must not count as ready while the middleware is down
  return { status, ready: (status?.ready ?? false) && !unreachable, unreachable };
}
