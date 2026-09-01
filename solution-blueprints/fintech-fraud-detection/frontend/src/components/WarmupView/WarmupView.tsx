// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Warm-up screen backed by the middleware /warmup/status endpoint: shows the
// real readiness of the dataset, backend orchestrator and GNN/XGB services.

import { useEffect, useState } from "react";
import type { WarmupStatus } from "../../types/api";
import styles from "./WarmupView.module.css";

interface WarmupViewProps {
  onGoLive: () => void;
  status: WarmupStatus | null;
  unreachable: boolean;
}

const STATS: { label: string; value: number | string; sub: string; format?: "number" }[] = [
  { label: "Total members", value: 590540, sub: "Full IEEE-CIS data slice", format: "number" },
  { label: "Pre-train rows", value: 472432, sub: "Historical training split", format: "number" },
  { label: "Demo stream rows", value: 118108, sub: "Rows available for live replay", format: "number" },
  { label: "History window", value: "30 days", sub: "Graph history window" },
  { label: "Graph nodes", value: 590540, sub: "Built during warm-up", format: "number" },
  { label: "Model version", value: "XGB+GNN v1.2", sub: "Active ensemble version" },
];

function useCountUp(target: number, durationMs = 2000): number {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const elapsed = now - start;
      const t = Math.min(1, elapsed / durationMs);
      // easeOutCubic — fast start, gentle settle
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(target * eased));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs]);
  return value;
}

function StatCard({
  label,
  value,
  sub,
  isNumber,
}: {
  label: string;
  value: number | string;
  sub: string;
  isNumber: boolean;
}) {
  const animated = useCountUp(isNumber ? (value as number) : 0);
  return (
    <div className={styles.statCard}>
      <div className={styles.statLbl}>{label}</div>
      <div className={styles.statVal}>{isNumber ? animated.toLocaleString() : value}</div>
      <div className={styles.statSub}>{sub}</div>
    </div>
  );
}

interface CheckRow {
  name: string;
  ok: boolean;
}

function buildChecks(status: WarmupStatus | null, unreachable: boolean): CheckRow[] {
  const rows: CheckRow[] = [
    { name: "API / middleware reachable", ok: status !== null && !unreachable },
    {
      name:
        status && status.dataset_rows > 0
          ? `IEEE-CIS transaction data loaded (${status.dataset_rows.toLocaleString()} rows)`
          : "IEEE-CIS transaction data loaded",
      ok: status?.checks.dataset ?? false,
    },
  ];
  // In fake scoring mode there is no backend to probe
  if (status?.scorer_mode !== "fake") {
    rows.push(
      { name: "Backend orchestrator ready (rules engine)", ok: status?.checks.backend ?? false },
      { name: "XGBoost service ready (ROCm)", ok: status?.checks.xgb ?? false },
      { name: "GNN service ready (PyG / ROCm)", ok: status?.checks.gnn ?? false },
    );
  }
  return rows;
}

export function WarmupView({ onGoLive, status, unreachable }: WarmupViewProps) {
  const checks = buildChecks(status, unreachable);
  const doneCount = checks.filter((c) => c.ok).length;
  const allOk = status?.ready ?? false;

  const stats = STATS.map((s) =>
    s.label === "Demo stream rows" && status ? { ...s, value: status.dataset_rows } : s,
  );

  return (
    <div className={styles.view}>
      <div className={styles.topRow}>
        <div className={styles.titleBlock}>
          <h1 className={styles.title}>System initialization</h1>
          <p className={styles.subtitle}>IEEE-CIS dataset</p>
        </div>
        <div className={styles.actions}>
          <button type='button' className={`${styles.btn} ${styles.btnPrimary}`} disabled={!allOk} onClick={onGoLive}>
            Start live stream
          </button>
        </div>
      </div>

      <div className={styles.grid}>
        <section className={styles.checksPanel}>
          <div className={styles.checksHead}>
            <h2 className={styles.checksTitle}>Backend health checks</h2>
            <p className={styles.checksSub}>
              {unreachable
                ? "Middleware unreachable — retrying"
                : allOk
                  ? "All systems ready"
                  : "Warming up"}{" "}
              · {doneCount}/{checks.length} checks passed
            </p>
          </div>
          <div className={styles.checkList}>
            {checks.map((c) => {
              const stroke = c.ok ? "#17B26A" : "var(--blue)";
              return (
                <div key={c.name} className={styles.checkRow}>
                  <svg
                    className={styles.checkSvg}
                    width='22'
                    height='22'
                    viewBox='0 0 22 22'
                    fill='none'
                    xmlns='http://www.w3.org/2000/svg'
                    aria-hidden='true'
                  >
                    <path
                      d='M21 11C21 16.5228 16.5228 21 11 21C5.47715 21 1 16.5228 1 11C1 5.47715 5.47715 1 11 1C16.5228 1 21 5.47715 21 11Z'
                      stroke={stroke}
                      strokeWidth='2'
                      strokeLinecap='round'
                      strokeLinejoin='round'
                    />

                    {c.ok && (
                      <path
                        d='M6.5 11L9.5 14L15.5 8'
                        stroke={stroke}
                        strokeWidth='2'
                        strokeLinecap='round'
                        strokeLinejoin='round'
                      />
                    )}
                  </svg>
                  <span className={styles.checkName}>{c.name}</span>
                </div>
              );
            })}
          </div>
        </section>

        <section className={styles.statGrid}>
          {stats.map((s) => (
            <StatCard key={s.label} label={s.label} value={s.value} sub={s.sub} isNumber={s.format === "number"} />
          ))}
        </section>
      </div>
    </div>
  );
}
