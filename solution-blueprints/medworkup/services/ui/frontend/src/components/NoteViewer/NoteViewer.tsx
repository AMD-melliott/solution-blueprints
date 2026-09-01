// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useMemo } from "react";
import styles from "./NoteViewer.module.css";
import type { ClinicalNote } from "../../utils/sampleNotes";
import type { AnalyzeState } from "../../hooks/useAnalyze";
import { checkSufficiency, SUFFICIENCY_BLOCKS } from "../../utils/sufficiency";

interface NoteViewerProps {
  note: ClinicalNote | null;
  analyze: AnalyzeState;
  onAnalyze: () => void;
}

function fmtSeconds(ms: number | null): string {
  if (ms == null) return "";
  return `${(ms / 1000).toFixed(1)}s`;
}

export function NoteViewer({ note, analyze, onAnalyze }: NoteViewerProps) {
  const sufficiency = useMemo(() => (note ? checkSufficiency(note.text) : { sufficient: false, missing: [] }), [note]);

  if (!note) {
    return (
      <div className={styles.body} style={{ color: "var(--text-xs)" }}>
        Select a clinical note from the list.
      </div>
    );
  }

  const isLoading = analyze.phase === "loading";
  const isDone = analyze.phase === "success";
  const blocked = !sufficiency.sufficient && SUFFICIENCY_BLOCKS;
  const canAnalyze = !isLoading && !blocked;

  // banner variant
  const bannerKind = isDone ? "analyzed" : sufficiency.sufficient ? "ready" : "insufficient";

  const result = analyze.result;

  return (
    <>
      <div className={styles.head}>
        <span className={styles.title}>{note.title}</span>

        {isDone && analyze.wallMs != null && <span className={styles.timingBadge}>{fmtSeconds(analyze.wallMs)}</span>}

        {isDone ? (
          <div>
            <span className={styles.doneBadge}>
              <svg width='15' height='11' viewBox='0 0 15 11' fill='none' xmlns='http://www.w3.org/2000/svg'>
                <path
                  d='M14.1666 0.833252L4.99998 9.99992L0.833313 5.83325'
                  stroke='#0086C9'
                  strokeWidth='1.66667'
                  strokeLinecap='round'
                  strokeLinejoin='round'
                />
              </svg>
              Done
            </span>
          </div>
        ) : (
          <button
            className={`${styles.analyzeBtn} ${isLoading ? styles.loading : ""}`}
            onClick={onAnalyze}
            disabled={!canAnalyze}
            title={blocked ? "Required clinical sections are missing" : undefined}
          >
            {isLoading ? (
              <>
                <span className={styles.spinner} />
                Analysis…
              </>
            ) : (
              <>Analyze</>
            )}
          </button>
        )}
      </div>

      {bannerKind === "insufficient" && (
        <div className={`${styles.banner} ${styles.insufficient}`}>
          <div className={styles.bannerRow}>
            <span className={styles.bannerIcon}>
              <svg width='19' height='19' viewBox='0 0 19 19' fill='none' xmlns='http://www.w3.org/2000/svg'>
                <path
                  d='M9.16668 5.8335V9.16683M9.16668 12.5002H9.17501M17.5 9.16683C17.5 13.7692 13.769 17.5002 9.16668 17.5002C4.5643 17.5002 0.833344 13.7692 0.833344 9.16683C0.833344 4.56446 4.5643 0.833496 9.16668 0.833496C13.769 0.833496 17.5 4.56446 17.5 9.16683Z'
                  stroke='#F04438'
                  strokeWidth='1.66667'
                  strokeLinecap='round'
                  strokeLinejoin='round'
                />
              </svg>
            </span>
            <span className={styles.bannerTitle}>Insufficient data</span>
          </div>
          <div className={styles.bannerSub}>Required clinical sections are missing.</div>
          <div className={styles.chips}>
            {sufficiency.missing.map((m) => (
              <span key={m} className={styles.chip}>
                Missing {m}
              </span>
            ))}
          </div>
        </div>
      )}

      {bannerKind === "analyzed" && result && (
        <div className={`${styles.banner} ${styles.analyzed}`}>
          <div className={styles.bannerRow}>
            <span className={styles.bannerIcon}>
              <svg width='19' height='19' viewBox='0 0 19 19' fill='none' xmlns='http://www.w3.org/2000/svg'>
                <path
                  d='M5.41668 9.16659L7.91668 11.6666L12.9167 6.66659M17.5 9.16659C17.5 13.769 13.769 17.4999 9.16668 17.4999C4.5643 17.4999 0.833344 13.769 0.833344 9.16659C0.833344 4.56421 4.5643 0.833252 9.16668 0.833252C13.769 0.833252 17.5 4.56421 17.5 9.16659Z'
                  stroke='#17B26A'
                  strokeWidth='1.66667'
                  strokeLinecap='round'
                  strokeLinejoin='round'
                />
              </svg>
            </span>
            <span className={styles.bannerTitle}>Analysis complete</span>
            <div className={styles.summaryBadges}>
              <span className={styles.summaryBadge}>{result.entities_raw.length} entities</span>
              <span className={styles.summaryBadge}>{result.entities_normalized.length} clusters</span>
              <span className={styles.summaryBadge}>{result.diagnoses.length} diagnoses</span>
            </div>
          </div>
        </div>
      )}

      <div className={styles.body}>{note.text}</div>
    </>
  );
}
