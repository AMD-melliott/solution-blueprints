// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Connects to /stream/events SSE endpoint, accumulates state in React.
//
// Two buffers, deliberately separate:
//  - `transactions`: capped recent window (MAX_ROWS) for the live table.
//  - `metricRows`:   lightweight all-time buffer (up to MAX_METRIC_ROWS) holding
//                    only the fields computeMetrics needs. This is what makes the
//                    slider an instant all-history what-if (BRD §2 P2, §8).
//
// The backend `metrics` aggregate is still received but no longer drives the
// Live view (BRD P1) — the listener is kept so other consumers/tabs don't break.
// Review Queue membership is NOT accumulated here anymore: it's recomputed from
// current client thresholds in LiveView, so there's a single band() source (§8).

import { useEffect, useRef, useState, useCallback } from "react";
import { api } from "../services/api";
import type { ScoredTransaction, Metrics, StreamState, Thresholds } from "../types/api";
import type { MetricRow } from "../utils/computeMetrics";

interface StateEvent {
  state: StreamState;
  metrics?: Metrics;
}

interface ThresholdsEvent {
  review: number;
  fraud: number;
}

const MAX_ROWS = 500;
const MAX_METRIC_ROWS = 20_000;

export interface StreamEventsState {
  connected: boolean;
  streamState: StreamState;
  metrics: Metrics;
  thresholds: Thresholds;
  transactions: ScoredTransaction[];
  metricRows: MetricRow[];
  metricsTruncated: boolean;
}

const emptyMetrics: Metrics = {
  total: 0,
  clean: 0,
  review: 0,
  fraud: 0,
  latency_min_ms: null,
  latency_avg_ms: null,
  latency_max_ms: null,
  accuracy_pct: null,
  missed_fraud_pct: null,
  false_positive_pct: null,
  missed_fraud_usd: 0,
  false_positive_usd: 0,
};

export function useStreamEvents(): StreamEventsState & { reset: () => void } {
  const [connected, setConnected] = useState(false);
  const [streamState, setStreamState] = useState<StreamState>("idle");
  const [metrics, setMetrics] = useState<Metrics>(emptyMetrics);
  const [thresholds, setThresholds] = useState<Thresholds>({ review: 0.65, fraud: 0.9 });
  const [transactions, setTransactions] = useState<ScoredTransaction[]>([]);
  const [metricRows, setMetricRows] = useState<MetricRow[]>([]);
  const [metricsTruncated, setMetricsTruncated] = useState(false);

  const seenIds = useRef<Set<number>>(new Set());

  const esRef = useRef<EventSource | null>(null);

  const reset = useCallback(() => {
    setTransactions([]);
    setMetricRows([]);
    setMetricsTruncated(false);
    setMetrics(emptyMetrics);
    seenIds.current.clear();
  }, []);

  useEffect(() => {
    const es = new EventSource(`${api.base}/stream/events`);
    esRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    es.addEventListener("state", (e: MessageEvent) => {
      const data = JSON.parse(e.data) as StateEvent;
      setStreamState(data.state);
      if (data.metrics) setMetrics(data.metrics);
    });

    es.addEventListener("transaction", (e: MessageEvent) => {
      const txn = JSON.parse(e.data) as ScoredTransaction;

      setTransactions((prev) => [txn, ...prev].slice(0, MAX_ROWS));

      if (!seenIds.current.has(txn.transaction_id)) {
        seenIds.current.add(txn.transaction_id);
        const mr: MetricRow = {
          score: txn.score,
          isFraud: txn.is_fraud_gt,
          amt: txn.transaction_amt,
          latency: txn.latency_ms,
        };
        setMetricRows((prev) => {
          if (prev.length >= MAX_METRIC_ROWS) {
            return [...prev.slice(prev.length - MAX_METRIC_ROWS + 1), mr];
          }
          return [...prev, mr];
        });
        if (seenIds.current.size > MAX_METRIC_ROWS) setMetricsTruncated(true);
      }
    });

    es.addEventListener("metrics", (e: MessageEvent) => {
      const m = JSON.parse(e.data) as Metrics;
      setMetrics(m);
    });

    es.addEventListener("thresholds", (e: MessageEvent) => {
      const t = JSON.parse(e.data) as ThresholdsEvent;
      setThresholds({ review: t.review, fraud: t.fraud });
    });

    return () => {
      es.close();
      esRef.current = null;
    };
  }, []);

  return {
    connected,
    streamState,
    metrics,
    thresholds,
    transactions,
    metricRows,
    metricsTruncated,
    reset,
  };
}
