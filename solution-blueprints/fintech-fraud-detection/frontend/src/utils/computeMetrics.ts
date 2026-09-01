// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Client-side metrics engine. Single source of truth for the Live view:
// the metrics strip, the per-row Decision/Result chips, and Review Queue
// membership all derive from band() + the SAME thresholds (BRD §8).
//
// Implements FT-1_Frontend_Metrics_BRD.md Appendix B (computeMetrics) verbatim
// in intent. All aggregates are computed on the frontend from per-row fields
// that arrive on every SSE `transaction` event (score, is_fraud_gt,
// transaction_amt, latency_ms) — the backend `metrics` aggregate is no longer
// used for display (BRD P1).

export type Band = "BLOCK" | "REVIEW" | "APPROVE";

export interface MetricRow {
  score: number;
  isFraud: number | null; // 1 = fraud, 0 = clean, null = unlabeled
  amt: number;
  latency: number;
}

export interface Tile {
  n: number;
  usd: number;
  pctN: number; // share of N_total (count)
  pctUsd: number;
}

export interface SentReviewTile extends Tile {
  fraudInsidePct: number; // N(fraud in REVIEW) / N(all REVIEW)
  fraudInsideUsd: number; // $ of fraud inside the review zone
}

export interface ComputedMetrics {
  processed: Tile;
  autoBlocked: Tile;
  sentReview: SentReviewTile;
  missed: Tile;
  falsePositive: Tile;

  // Hero summary line — denominator is real fraud only.
  detectionRate: number | null; // caught / all fraud (count)
  protectedUsd: number; // $ caught (auto-block + fraud-in-review)
  protectionRate: number | null; // $ caught / $ all fraud

  latency: { min: number | null; avg: number | null; max: number | null };

  // Bookkeeping for callers / debugging.
  fraudN: number; // total real fraud rows (denominator)
  truncated: boolean; // true if buffer hit its cap (show "last N tx")
  bufferSize: number;
}

export function band(score: number, reviewTh: number, fraudTh: number): Band {
  if (score >= fraudTh) return "BLOCK";
  if (score >= reviewTh) return "REVIEW";
  return "APPROVE";
}

const pct = (x: number, base: number): number => (base ? (100 * x) / base : 0);

export function computeMetrics(
  rows: MetricRow[],
  reviewTh: number,
  fraudTh: number,
  truncated = false,
): ComputedMetrics {
  let N = 0;
  let USD = 0;
  let latMin: number | null = null;
  let latMax: number | null = null;
  let latSum = 0;
  let fraudN = 0;
  let fraudUsd = 0;

  const z = () => ({ n: 0, usd: 0 });
  const autoBlocked = z();
  const fraudReview = z();
  const missed = z();
  const fp = z();
  const sentReview = z();

  for (const r of rows) {
    N++;
    USD += r.amt;
    latSum += r.latency;
    latMin = latMin === null ? r.latency : Math.min(latMin, r.latency);
    latMax = latMax === null ? r.latency : Math.max(latMax, r.latency);

    const b = band(r.score, reviewTh, fraudTh);
    if (b === "REVIEW") {
      sentReview.n++;
      sentReview.usd += r.amt;
    }

    if (r.isFraud === null) continue;

    if (r.isFraud === 1) {
      fraudN++;
      fraudUsd += r.amt;
      if (b === "BLOCK") {
        autoBlocked.n++;
        autoBlocked.usd += r.amt;
      } else if (b === "REVIEW") {
        fraudReview.n++;
        fraudReview.usd += r.amt;
      } else {
        missed.n++;
        missed.usd += r.amt;
      }
    } else {
      if (b === "BLOCK") {
        fp.n++;
        fp.usd += r.amt;
      }
    }
  }

  const caughtN = autoBlocked.n + fraudReview.n;
  const caughtUsd = autoBlocked.usd + fraudReview.usd;
  const tile = (g: { n: number; usd: number }): Tile => ({
    n: g.n,
    usd: g.usd,
    pctN: pct(g.n, N),
    pctUsd: pct(g.usd, USD),
  });

  const result: ComputedMetrics = {
    processed: { n: N, usd: USD, pctN: N ? 100 : 0, pctUsd: USD ? 100 : 0 },
    autoBlocked: tile(autoBlocked),
    sentReview: {
      ...tile(sentReview),
      fraudInsidePct: pct(fraudReview.n, sentReview.n),
      fraudInsideUsd: fraudReview.usd,
    },
    missed: tile(missed),
    falsePositive: tile(fp),

    detectionRate: fraudN ? pct(caughtN, fraudN) : null,
    protectedUsd: caughtUsd,
    protectionRate: fraudUsd ? pct(caughtUsd, fraudUsd) : null,

    latency: { min: latMin, avg: N ? Math.round(latSum / N) : null, max: latMax },

    fraudN,
    truncated,
    bufferSize: N,
  };

  const dev = typeof import.meta !== "undefined" && (import.meta as { env?: { DEV?: boolean } }).env?.DEV;
  if (dev) {
    const lhs = autoBlocked.n + fraudReview.n + missed.n;
    if (lhs !== fraudN) {
      // eslint-disable-next-line no-console
      console.error(`computeMetrics invariant broken: ${lhs} !== fraudN ${fraudN}`);
    }
  }

  return result;
}
