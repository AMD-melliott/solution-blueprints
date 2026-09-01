// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import styles from "./Header.module.css";
import type { SystemHealthState, PollState } from "../../hooks/useSystemHealth";

interface HeaderProps {
  health: SystemHealthState;
}

const SERVICES: { key: keyof Omit<SystemHealthState, "overall">; label: string }[] = [
  { key: "ner", label: "NER" },
  { key: "embedding", label: "Embeddings" },
  { key: "llm", label: "LLM" },
  { key: "orchestrator", label: "Pipeline" },
];

function statusLabel(overall: SystemHealthState["overall"]): string {
  if (overall === "ok") return "System ready";
  if (overall === "degraded") return "Degraded";
  return "Connecting…";
}

export function Header({ health }: HeaderProps) {
  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        MED<span className={styles.logoAccent}>WORKUP</span>
      </div>
      <span className={styles.subtitle}>Clinical Diagnostic Assistant</span>

      <div className={styles.right}>
        {SERVICES.map(({ key, label }) => {
          const status = health[key] as PollState;
          return (
            <span key={key} className={styles.pill}>
              <span className={`${styles.dot} ${styles[status]}`} />
              {label}
            </span>
          );
        })}
        <span className={`${styles.statusBadge} ${styles[health.overall]}`}>
          <span className={`${styles.dot} ${styles[health.overall]}`} />
          {statusLabel(health.overall)}
        </span>
      </div>
    </header>
  );
}
