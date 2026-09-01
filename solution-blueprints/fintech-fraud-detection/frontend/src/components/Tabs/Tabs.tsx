// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import type { TabId } from "../../types/api";
import styles from "./Tabs.module.css";

interface TabsProps {
  active: TabId;
  onChange: (id: TabId) => void;
}

const TABS: { id: TabId; label: string }[] = [
  { id: "warmup", label: "Warm Up" },
  { id: "live", label: "Live" },
];

export function Tabs({ active, onChange }: TabsProps) {
  return (
    <nav className={styles.tabs}>
      {TABS.map((t) => (
        <button
          key={t.id}
          className={`${styles.tab} ${active === t.id ? styles.active : ""}`}
          onClick={() => onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </nav>
  );
}
