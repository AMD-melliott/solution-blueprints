// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Thin wrapper exposing stream control actions to components.
// Components shouldn't import `api` directly — go through this hook.

import { useCallback } from "react";
import { api, type BatchScoreResponse } from "../services/api";
import type { Thresholds } from "../types/api";

export function useStreamControl() {
  const start = useCallback(() => api.start().catch(console.error), []);
  const pause = useCallback(() => api.pause().catch(console.error), []);
  const stop = useCallback(() => api.stop().catch(console.error), []);
  const setThresholds = useCallback((t: Thresholds) => api.setThresholds(t).catch(console.error), []);
  const batchScore = useCallback(
    (count: number, mode?: string): Promise<BatchScoreResponse> => api.batchScore(count, mode),
    [],
  );
  return { start, pause, stop, setThresholds, batchScore };
}
