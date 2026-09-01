// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import styles from "./DiagnosisCard.module.css";
import type { Diagnosis } from "../../types/api";

interface DiagnosisCardProps {
  diagnosis: Diagnosis;
  rank: number; // 1-based
}

export function DiagnosisCard({ diagnosis, rank }: DiagnosisCardProps) {
  const pct = Math.round((diagnosis.confidence ?? 0) * 100);
  const tier = pct >= 60 ? "high" : pct >= 30 ? "mid" : "low";

  return (
    <div className={styles.card}>
      <div className={styles.head}>
        <span className={styles.rank}>{rank}.</span>
        <span className={styles.name}>{diagnosis.name}</span>
        <span className={`${styles.confDot} ${styles[tier]}`} />
      </div>

      <div className={styles.confRow}>
        <div className={styles.barTrack}>
          <div className={`${styles.barFill} ${styles[tier]}`} style={{ width: `${pct}%` }} />
        </div>
      </div>

      {diagnosis.reasoning && <div className={styles.reasoning}>{diagnosis.reasoning}</div>}

      {diagnosis.consistency_note && (
        <div className={styles.callout}>
          <span className={styles.calloutIcon}>
            <svg width='17' height='17' viewBox='0 0 19 19' fill='none' xmlns='http://www.w3.org/2000/svg'>
              <path
                d='M9.16668 5.8335V9.16683M9.16668 12.5002H9.17501M17.5 9.16683C17.5 13.7692 13.769 17.5002 9.16668 17.5002C4.5643 17.5002 0.833344 13.7692 0.833344 9.16683C0.833344 4.56446 4.5643 0.833496 9.16668 0.833496C13.769 0.833496 17.5 4.56446 17.5 9.16683Z'
                stroke='#F79009'
                strokeWidth='1.66667'
                strokeLinecap='round'
                strokeLinejoin='round'
              />
            </svg>
          </span>
          <span>{diagnosis.consistency_note}</span>
        </div>
      )}

      {diagnosis.evidence.length > 0 && (
        <>
          <div className={styles.evidenceLabel}>Evidence</div>
          <div className={styles.chips}>
            {diagnosis.evidence.map((e, i) => (
              <span key={i} className={styles.chip}>
                {e}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
