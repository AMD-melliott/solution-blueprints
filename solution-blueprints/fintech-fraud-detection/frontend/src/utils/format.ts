// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Shared formatters used by table + drawer + metrics strip.
export function fmtDT(dt: number | null | undefined): string {
  if (dt === null || dt === undefined) return "—";
  const total = Math.floor(dt);
  const days = Math.floor(total / 86400);
  const hours = Math.floor((total % 86400) / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${days}d ${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;
}

// Compact USD for the metrics strip: $4.82M / $95K / $3.1K / $312 / $3.10.
// Big sums use M/K with enough precision to stay legible; small sums (< $1K)
// show whole dollars, and < $10 keeps 2 decimals so they don't collapse to "$3".
export function fmtCompactUsd(n: number): string {
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (abs >= 1_000_000) {
    const m = abs / 1_000_000;
    const dp = m < 10 ? 2 : m < 100 ? 1 : 0;
    return `${sign}$${m.toFixed(dp)}M`;
  }
  if (abs >= 1_000) {
    const k = abs / 1_000;
    const dp = k < 10 ? 1 : 0; // $3.1K but $95K
    return `${sign}$${k.toFixed(dp)}K`;
  }
  if (abs < 10 && abs > 0) {
    return `${sign}$${abs.toFixed(2)}`; // $3.10
  }
  return `${sign}$${Math.round(abs).toLocaleString("en-US")}`; // $312
}

// Percent to 1 decimal: 91.0%. null -> em dash.
export function fmtPct1(n: number | null | undefined): string {
  return n === null || n === undefined ? "—" : `${n.toFixed(1)}%`;
}

// Whole-count with thousands separators: 20,000. null -> em dash.
export function fmtCount(n: number | null | undefined): string {
  return n === null || n === undefined ? "—" : n.toLocaleString("en-US");
}
