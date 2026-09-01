// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import styles from "./MetricsStrip.module.css";

import type { ComputedMetrics } from "../../utils/computeMetrics";
import { fmtCompactUsd, fmtPct1, fmtCount } from "../../utils/format";
import { InfoTip } from "../InfoTip/InfoTip";

const TIP_PROTECTED = (
  <>
    Total fraud value we stopped — auto-blocked <strong>plus</strong> fraud caught in the review queue. The % is this as
    a share of all fraud dollars in the stream.
  </>
);
const TIP_LATENCY = <>Per-transaction scoring time in ms over the current window: fastest / average / slowest.</>;
const TIP_PROCESSED = <>Every transaction scored so far — the denominator for all shares. 100% of volume and count.</>;
const TIP_AUTOBLOCKED = (
  <>
    Transactions blocked outright because score ≥ <strong>block</strong> threshold. Fraud caught with a hard stop. Shows
    $, count and share of total.
  </>
);
const TIP_SENTREVIEW = (
  <>
    Transactions flagged for a human (score between <strong>review</strong> and <strong>block</strong>). "Real fraud
    inside" = how much of this queue is actually fraud.
  </>
);
const TIP_MISSED = (
  <>
    Fraud that scored below the <strong>review</strong> threshold and was approved — the only real leak. Lower is
    better.
  </>
);
const TIP_FALSEPOS = (
  <>Clean transactions we blocked by mistake (score ≥ block, but not fraud). The cost of blocking too aggressively.</>
);

interface MetricsStripProps {
  m: ComputedMetrics;
}

function latencyStr(l: ComputedMetrics["latency"]): string {
  if (l.min === null || l.avg === null || l.max === null) return "—";
  return `${l.min} / ${l.avg} / ${l.max} ms`;
}

export function MetricsStrip({ m }: MetricsStripProps) {
  const protectedStr =
    m.protectionRate === null
      ? fmtCompactUsd(m.protectedUsd)
      : `${fmtCompactUsd(m.protectedUsd)} · ${m.protectionRate.toFixed(0)}%`;

  return (
    <div className={styles.metrics}>
      <div className={styles.mcard}>
        <div className={styles.mHead}>
          <span className={styles.mTitle}>Fraud outcome — live</span>
        </div>

        <div className={styles.summary}>
          <div className={styles.sumStat}>
            <span className={styles.k}>
              <InfoTip label='Fraud $ protected' title='Fraud $ protected' body={TIP_PROTECTED} />
            </span>
            <span className={`${styles.v} ${styles.g}`}>{protectedStr}</span>
          </div>
          <div className={styles.sumStat}>
            <span className={styles.k}>
              <InfoTip label='Latency min / avg / max' title='Scoring latency' body={TIP_LATENCY} />
            </span>
            <span className={`${styles.v} ${styles.b}`}>{latencyStr(m.latency)}</span>
          </div>
          <span className={styles.mHint}>
            <span className={styles.dot} />
            recomputes as you move the thresholds
          </span>
        </div>

        <div className={styles.ogrid}>
          {/* Processed — neutral, the denominator */}
          <div className={styles.otile}>
            <div className={styles.otHead}>
              <span className={styles.otHeadLabel}>
                <InfoTip label='Processed' title='Processed' body={TIP_PROCESSED} />
              </span>
              <span className={styles.otDot} style={{ background: "var(--text-dim)" }} />
            </div>
            <div className={styles.otMoney}>{fmtCompactUsd(m.processed.usd)}</div>
            <div className={styles.otMoneyPct}>100% of volume</div>
            <div className={styles.otCount}>
              {fmtCount(m.processed.n)} <span>tx · 100%</span>
            </div>
            <div className={styles.otSub}>all scored</div>
          </div>

          {/* Auto-Blocked — green, caught fraud, hard stop */}
          <div className={`${styles.otile} ${styles.tAuto}`}>
            <div className={styles.otHead}>
              <span className={styles.otHeadLabel}>
                <InfoTip label='Auto-Blocked' title='Auto-Blocked' body={TIP_AUTOBLOCKED} />
              </span>
              <span className={styles.otDot} style={{ background: "var(--green)" }} />
            </div>
            <div className={`${styles.otMoney} ${styles.g}`}>{fmtCompactUsd(m.autoBlocked.usd)}</div>
            <div className={styles.otMoneyPct}>{fmtPct1(m.autoBlocked.pctUsd)} of total $</div>
            <div className={`${styles.otCount} ${styles.g}`}>
              {fmtCount(m.autoBlocked.n)} <span>tx · {fmtPct1(m.autoBlocked.pctN)} of total</span>
            </div>
            <div className={styles.otSub}>fraud caught · hard stop</div>
          </div>

          {/* Sent to Review — amber, the queue, with % real fraud inside */}
          <div className={`${styles.otile} ${styles.tReview}`}>
            <div className={styles.otHead}>
              <span className={styles.otHeadLabel}>
                <InfoTip label='Sent to Review' title='Sent to Review' body={TIP_SENTREVIEW} />
              </span>
              <span className={styles.otDot} style={{ background: "var(--amber)" }} />
            </div>
            <div className={`${styles.otMoney} ${styles.a}`}>{fmtCompactUsd(m.sentReview.usd)}</div>
            <div className={styles.otMoneyPct}>
              {fmtPct1(m.sentReview.pctUsd)} of total $ · {fmtCompactUsd(m.sentReview.fraudInsideUsd)} real fraud
            </div>
            <div className={`${styles.otCount} ${styles.a}`}>
              {fmtCount(m.sentReview.n)} <span>tx · {fmtPct1(m.sentReview.pctN)} of total</span>
            </div>
            <div className={styles.otSub}>
              <span className={styles.otTag}>{fmtPct1(m.sentReview.fraudInsidePct)} real fraud inside</span>
            </div>
          </div>

          {/* Missed — red, fraud that got through */}
          <div className={`${styles.otile} ${styles.tMissed}`}>
            <div className={styles.otHead}>
              <span className={styles.otHeadLabel}>
                <InfoTip label='Missed' title='Missed' body={TIP_MISSED} />
              </span>
              <span className={styles.otDot} style={{ background: "var(--red)" }} />
            </div>
            <div className={`${styles.otMoney} ${styles.r}`}>{fmtCompactUsd(m.missed.usd)}</div>
            <div className={styles.otMoneyPct}>{fmtPct1(m.missed.pctUsd)} of total $</div>
            <div className={`${styles.otCount} ${styles.r}`}>
              {fmtCount(m.missed.n)} <span>tx · {fmtPct1(m.missed.pctN)} of total</span>
            </div>
            <div className={styles.otSub}>fraud got through</div>
          </div>

          {/* False positive — amber, clean wrongly blocked */}
          <div className={`${styles.otile} ${styles.tFp}`}>
            <div className={styles.otHead}>
              <span className={styles.otHeadLabel}>
                <InfoTip label='False positive' title='False positive' body={TIP_FALSEPOS} />
              </span>
              <span className={styles.otDot} style={{ background: "var(--amber)" }} />
            </div>
            <div className={`${styles.otMoney} ${styles.a}`}>{fmtCompactUsd(m.falsePositive.usd)}</div>
            <div className={styles.otMoneyPct}>{fmtPct1(m.falsePositive.pctUsd)} of total $</div>
            <div className={`${styles.otCount} ${styles.a}`}>
              {fmtCount(m.falsePositive.n)} <span>tx · {fmtPct1(m.falsePositive.pctN)} of total</span>
            </div>
            <div className={styles.otSub}>clean wrongly blocked</div>
          </div>
        </div>

        {m.truncated && <div className={styles.truncNote}>Metrics over the last {fmtCount(m.bufferSize)} tx</div>}
      </div>
    </div>
  );
}
