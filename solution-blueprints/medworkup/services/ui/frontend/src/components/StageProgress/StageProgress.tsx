// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useEffect, useState } from "react";
import styles from "./StageProgress.module.css";

const STAGES = [
  "Preprocessing",
  "Entity extraction",
  "Entity aggregation",
  "Semantic embeddings",
  "Clinical categorization",
  "Diagnosis candidate generation",
  "Clinical reasoning",
  "Ranking",
  "Consistency check",
];

const TICK_MS = 380;

export function StageProgress() {
  const [active, setActive] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setActive((i) => {
        if (i >= STAGES.length - 1) {
          clearInterval(id);
          return i;
        }
        return i + 1;
      });
    }, TICK_MS);
    return () => clearInterval(id);
  }, []);

  return (
    <div className={styles.wrap}>
      <div className={styles.spinner} />
      <div className={styles.label}>Running analysis…</div>
      <div className={styles.stages}>
        {STAGES.map((name, i) => {
          const state = i < active ? "done" : i === active ? "active" : "pending";
          const isLast = i === STAGES.length - 1;
          const connectorClass = state === "done" ? styles.connDotted : styles.connLine;
          return (
            <div key={name} className={`${styles.stage} ${styles[state]}`}>
              <div className={styles.markerCol}>
                <span className={styles.marker}>
                  {state === "done" && (
                    <svg width='12' height='12' viewBox='0 0 12 12' fill='none' xmlns='http://www.w3.org/2000/svg'>
                      <path
                        d='M10 3L4.5 8.5L2 6'
                        stroke='white'
                        strokeWidth='2'
                        strokeLinecap='round'
                        strokeLinejoin='round'
                      />
                    </svg>
                  )}
                  {state === "active" && <span className={styles.activeDot} />}
                </span>
                {!isLast && <span className={connectorClass} />}
              </div>
              <span className={styles.stageName}>{name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
