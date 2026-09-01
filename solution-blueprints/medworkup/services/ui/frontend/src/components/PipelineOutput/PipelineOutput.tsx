// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import { useState } from "react";
import styles from "./PipelineOutput.module.css";

import type { AnalyzeState } from "../../hooks/useAnalyze";
import type { AnalyzeResponse, CategorizedEntities } from "../../types/api";
import { StageProgress } from "../StageProgress/StageProgress";
import { DiagnosisCard } from "../DiagnosisCard/DiagnosisCard";
import { Section } from "../Section/Section";

interface PipelineOutputProps {
  analyze: AnalyzeState;
  /** True when the selected note fails the client-side sufficiency check. */
  insufficient: boolean;
  onRetry: () => void;
}

const CAT_GROUPS: { key: keyof CategorizedEntities; label: string; cls: string }[] = [
  { key: "symptoms", label: "Symptoms", cls: "symptom" },
  { key: "diseases", label: "Diseases", cls: "disease" },
  { key: "findings", label: "Findings", cls: "finding" },
  { key: "procedures", label: "Procedures", cls: "procedure" },
  { key: "other", label: "Other", cls: "other" },
];

function Header() {
  return (
    <div className={styles.head}>
      <span className={styles.headTitle}>Pipeline Output</span>
    </div>
  );
}

export function PipelineOutput({ analyze, insufficient, onRetry }: PipelineOutputProps) {
  if (analyze.phase === "loading") {
    return (
      <>
        <Header />
        <StageProgress />
      </>
    );
  }

  if (analyze.phase === "error") {
    return (
      <>
        <Header />
        <div className={styles.centered}>
          <div className={`${styles.centeredIcon} ${styles.error}`}>!</div>
          <div className={styles.centeredTitle}>Pipeline error</div>
          <div className={styles.errorDetail}>{analyze.error}</div>
          <button className={styles.retryBtn} onClick={onRetry}>
            ↻ Retry
          </button>
        </div>
      </>
    );
  }

  if (analyze.phase === "success" && analyze.result) {
    return (
      <>
        <Header />
        <ResultView data={analyze.result} />
      </>
    );
  }

  return (
    <>
      <Header />
      <div className={styles.centered}>
        {insufficient ? (
          <>
            <div className={`${styles.centeredIcon} ${styles.error}`}>
              <svg width='24' height='24' viewBox='0 0 19 19' fill='none' xmlns='http://www.w3.org/2000/svg'>
                <path
                  d='M9.16668 5.8335V9.16683M9.16668 12.5002H9.17501M17.5 9.16683C17.5 13.7692 13.769 17.5002 9.16668 17.5002C4.5643 17.5002 0.833344 13.7692 0.833344 9.16683C0.833344 4.56446 4.5643 0.833496 9.16668 0.833496C13.769 0.833496 17.5 4.56446 17.5 9.16683Z'
                  stroke='#FECDCA'
                  strokeWidth='1.66667'
                  strokeLinecap='round'
                  strokeLinejoin='round'
                />
              </svg>
            </div>
            <div className={styles.centeredTitle}>Insufficient data</div>
            <div className={styles.centeredText}>
              The pipeline cannot generate reliable diagnosis candidates from this note.
            </div>
          </>
        ) : (
          <>
            <div className={styles.centeredTitle}>Awaiting analysis</div>
            <div className={styles.centeredText}>
              Click Analyze to extract clinical entities, map medical codes, and generate diagnosis candidates.
            </div>
            <button className={styles.awaitingBtn} onClick={onRetry}>
              Analyze
            </button>
          </>
        )}
      </div>
    </>
  );
}

function ResultView({ data }: { data: AnalyzeResponse }) {
  const diagnoses = [...data.diagnoses].sort((a, b) => b.confidence - a.confidence);
  const cat = data.categorized_entities;
  const catTotal = CAT_GROUPS.reduce((n, g) => n + (cat[g.key]?.length ?? 0), 0);
  const meta = data.pipeline_meta;
  const maxMs = meta ? Math.max(...meta.stage_timings.map((t) => t.duration_ms), 1) : 1;

  return (
    <div className={styles.scroll}>
      {/* diagnoses */}
      <Section title='Differential Diagnoses' dot='blue' count={diagnoses.length} defaultOpen>
        {diagnoses.length ? (
          diagnoses.map((d, i) => <DiagnosisCard key={i} diagnosis={d} rank={i + 1} />)
        ) : (
          <div className={styles.emptyCat}>No diagnoses returned.</div>
        )}
      </Section>

      {/* categorized */}
      <Section title='Categorized Entities' dot='amber' count={catTotal}>
        {catTotal === 0 ? (
          <div className={styles.emptyCat}>No entities categorized.</div>
        ) : (
          CAT_GROUPS.filter((g) => (cat[g.key]?.length ?? 0) > 0).map((g) => (
            <div key={g.key} className={styles.catGroup}>
              <div className={styles.catGroupLabel}>
                {g.label} ({cat[g.key].length})
              </div>
              <div className={styles.catChips}>
                {cat[g.key].map((e, i) => (
                  <span key={i} className={`${styles.catChip} ${styles[g.cls]}`}>
                    {e}
                  </span>
                ))}
              </div>
            </div>
          ))
        )}
      </Section>

      {/* normalized */}
      <Section title='Normalized Entities' dot='green' count={data.entities_normalized.length}>
        {data.entities_normalized.length === 0 ? (
          <div className={styles.emptyCat}>No normalized entities.</div>
        ) : (
          <div className={styles.clusterGrid}>
            {data.entities_normalized.map((e, i) => (
              <div key={i} className={styles.clusterCard}>
                <div className={styles.clusterCanonical}>
                  {e.canonical}
                  <span className={styles.clusterBadge}>{e.label}</span>
                  {e.negated && <span className={styles.negFlag}>negated</span>}
                </div>
                <div className={styles.clusterMeta}>
                  <span>
                    {e.mention_count} mention{e.mention_count !== 1 ? "s" : ""}
                  </span>
                  {e.variants.length > 0 && (
                    <span>
                      {e.variants.length} variant{e.variants.length !== 1 ? "s" : ""}
                    </span>
                  )}
                </div>
                {e.variants.length > 0 && (
                  <div className={styles.clusterVariants}>
                    {e.variants.map((v, j) => (
                      <span key={j} className={styles.variantChip}>
                        {v}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* raw entities */}
      <Section title='Raw Entities' dot='blue' count={data.entities_raw.length}>
        {data.entities_raw.length === 0 ? (
          <div className={styles.emptyCat}>No raw entities.</div>
        ) : (
          <div className={styles.rawTableWrap}>
            <table className={styles.rawTable}>
              <thead>
                <tr>
                  <th>Entity</th>
                  <th>Label</th>
                  <th>Source</th>
                  <th>Offsets</th>
                </tr>
              </thead>
              <tbody>
                {data.entities_raw.map((e, i) => (
                  <tr key={i}>
                    <td>
                      <span className={e.negated ? styles.entNegated : ""}>{e.text}</span>
                    </td>
                    <td>
                      <span className={styles.entLabel}>{e.label}</span>
                    </td>
                    <td>
                      <span className={styles.entSource}>{e.source}</span>
                    </td>
                    <td>
                      <span className={styles.entOffset}>
                        {e.start}–{e.end}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* stage timings */}
      {meta && meta.stage_timings.length > 0 && (
        <Section title='Stage Timings' dot='purple'>
          {meta.stage_timings.map((t, i) => {
            const pct = Math.round((t.duration_ms / maxMs) * 100);
            return (
              <div key={i} className={styles.timingRow}>
                <div className={styles.timingHead}>
                  <span className={styles.timingStage}>{t.stage}</span>
                  <span className={styles.timingMs}>{Math.round(t.duration_ms)}ms</span>
                </div>
                <div className={styles.timingTrack}>
                  <div className={styles.timingFill} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
          <div className={styles.timingTotal}>
            Total: <strong>{Math.round(meta.total_duration_ms)}ms</strong> · {meta.entity_count_raw} raw →{" "}
            {meta.entity_count_normalized} clusters
          </div>
        </Section>
      )}

      <RawJson data={data} />
    </div>
  );
}

function RawJson({ data }: { data: AnalyzeResponse }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const json = JSON.stringify(data, null, 2);

  function copy(e: React.MouseEvent) {
    e.stopPropagation();
    navigator.clipboard.writeText(json).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  return (
    <>
      <div className={styles.jsonHead} onClick={() => setOpen((o) => !o)}>
        <span className={styles.jsonLabel}>Raw JSON</span>
        <button className={styles.copyBtn} onClick={copy}>
          <svg width='13' height='13' viewBox='0 0 19 19' fill='none' xmlns='http://www.w3.org/2000/svg'>
            <path
              d='M5.83337 5.8335V3.50016C5.83337 2.56674 5.83337 2.10003 6.01503 1.74351C6.17482 1.42991 6.42979 1.17494 6.74339 1.01515C7.09991 0.833496 7.56662 0.833496 8.50004 0.833496H14.8334C15.7668 0.833496 16.2335 0.833496 16.59 1.01515C16.9036 1.17494 17.1586 1.42991 17.3184 1.74351C17.5 2.10003 17.5 2.56674 17.5 3.50016V9.8335C17.5 10.7669 17.5 11.2336 17.3184 11.5901C17.1586 11.9038 16.9036 12.1587 16.59 12.3185C16.2335 12.5002 15.7668 12.5002 14.8334 12.5002H12.5M3.50004 17.5002H9.83337C10.7668 17.5002 11.2335 17.5002 11.59 17.3185C11.9036 17.1587 12.1586 16.9038 12.3184 16.5901C12.5 16.2336 12.5 15.7669 12.5 14.8335V8.50016C12.5 7.56674 12.5 7.10003 12.3184 6.74351C12.1586 6.42991 11.9036 6.17494 11.59 6.01515C11.2335 5.8335 10.7668 5.8335 9.83337 5.8335H3.50004C2.56662 5.8335 2.09991 5.8335 1.74339 6.01515C1.42979 6.17494 1.17482 6.42991 1.01503 6.74351C0.833374 7.10003 0.833374 7.56674 0.833374 8.50016V14.8335C0.833374 15.7669 0.833374 16.2336 1.01503 16.5901C1.17482 16.9038 1.42979 17.1587 1.74339 17.3185C2.09991 17.5002 2.56662 17.5002 3.50004 17.5002Z'
              stroke='#0086C9'
              strokeWidth='1.66667'
              strokeLinecap='round'
              strokeLinejoin='round'
            />
          </svg>
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      {open && (
        <div className={styles.jsonBody}>
          <pre className={styles.jsonPre}>{json}</pre>
        </div>
      )}
    </>
  );
}
