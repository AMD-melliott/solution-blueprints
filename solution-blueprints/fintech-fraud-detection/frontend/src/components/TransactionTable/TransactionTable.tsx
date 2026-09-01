// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import type React from "react";
import styles from "./TransactionTable.module.css";
import type { ScoredTransaction } from "../../types/api";
import { fmtDT } from "../../utils/format";
import { classifyRow, type DecisionLabel, type ResultLabel } from "../../utils/rowVerdict";
import { InfoTip } from "../InfoTip/InfoTip";

const TIP_SCORE = (
  <>
    Fraud-risk score from 0 to 1. Higher means more likely fraud. It's compared to the thresholds to decide Approve /
    Review / Block.
  </>
);
const TIP_DECISION = (
  <>
    What the system did, from score vs current thresholds: <strong>Approved</strong> — let through (below review);{" "}
    <strong>Review</strong> — sent to a human (review–block); <strong>Blocked</strong> — stopped (≥ block).
  </>
);
const TIP_RESULT = (
  <>
    Outcome once ground truth is known: <strong>Caught</strong> — fraud stopped; <strong>Missed</strong> — fraud slipped
    through; <strong>False alert</strong> — clean wrongly blocked; <strong>Reviewed clean</strong> — clean, sent to
    review; <strong>Clean</strong> — clean, approved. Empty until the transaction is labeled.
  </>
);
const TIP_LATENCY = <>Time to score this single transaction, in ms.</>;

interface TransactionTableProps {
  rows: ScoredTransaction[];
  variant?: "stream" | "queue";
  reviewTh: number;
  fraudTh: number;
  onRowClick?: (txn: ScoredTransaction) => void;
  selectedId?: number | null;
}

function fmtAmount(amt: number): string {
  return `$${amt.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function decisionClass(d: DecisionLabel): string {
  if (d === "Blocked") return styles.stBlock;
  if (d === "Review") return styles.stReview;
  return styles.stApprove;
}

function resultClass(r: ResultLabel): string {
  switch (r) {
    case "Caught":
      return styles.resCaught;
    case "Missed":
      return styles.resMissed;
    case "False alert":
      return styles.resFalse;
    case "Reviewed clean":
      return styles.resReviewClean;
    case "Clean":
      return styles.resClean;
    default:
      return styles.resNa;
  }
}

function resultText(r: ResultLabel): string {
  switch (r) {
    case "Caught":
      return "✓ Caught";
    case "Missed":
      return "✗ Missed";
    case "False alert":
      return "False alert";
    case "Reviewed clean":
      return "Reviewed clean";
    case "Clean":
      return "Clean";
    default:
      return "—";
  }
}

function resultCell(r: ResultLabel): React.JSX.Element {
  return <span className={`${styles.res} ${resultClass(r)}`}>{resultText(r)}</span>;
}

export function TransactionTable({
  rows,
  variant = "stream",
  reviewTh,
  fraudTh,
  onRowClick,
  selectedId,
}: TransactionTableProps) {
  const isQueue = variant === "queue";

  if (rows.length === 0) {
    return <div className={styles.empty}>{isQueue ? "No flagged transactions" : "No transactions yet"}</div>;
  }
  return (
    <table className={styles.table}>
      <thead>
        <tr>
          <th className={styles.colId}>ID</th>
          <th className={styles.colTime}>Time</th>
          {!isQueue && <th className={styles.colCard}>Card</th>}
          <th className={styles.colAmount}>Amount</th>
          <th className={styles.colScore}>
            <span className={styles.thLabel}>
              <InfoTip label='Score' title='Model score' body={TIP_SCORE} />
            </span>
          </th>
          <th className={styles.colStatus}>
            <span className={styles.thLabel}>
              <InfoTip label='Decision' title='Decision' body={TIP_DECISION} />
            </span>
          </th>
          <th className={styles.colResult}>
            <span className={styles.thLabel}>
              <InfoTip label='Result' title='Result' body={TIP_RESULT} />
            </span>
          </th>
          {!isQueue && (
            <th className={styles.colLatency}>
              <span className={styles.thLabel}>
                <InfoTip label='Latency' title='Latency' body={TIP_LATENCY} />
              </span>
            </th>
          )}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const v = classifyRow(r.score, r.is_fraud_gt, reviewTh, fraudTh);
          return (
            <tr
              key={`${r.transaction_id}-${r.transaction_dt}`}
              className={`${onRowClick ? styles.clickable : ""} ${
                selectedId === r.transaction_id ? styles.selected : ""
              }`}
              onClick={onRowClick ? () => onRowClick(r) : undefined}
            >
              <td className={styles.colId}>TXN-{r.transaction_id}</td>
              <td className={styles.colTime}>{fmtDT(r.transaction_dt)}</td>
              {!isQueue && <td className={styles.colCard}>{r.card_label}</td>}
              <td className={styles.colAmount}>{fmtAmount(r.transaction_amt)}</td>
              <td className={styles.colScore}>{r.score.toFixed(2)}</td>
              <td className={styles.colStatus}>
                <span className={`${styles.st} ${decisionClass(v.decision)}`}>{v.decision}</span>
              </td>
              <td className={styles.colResult}>{resultCell(v.result)}</td>
              {!isQueue && <td className={styles.colLatency}>{r.latency_ms}ms</td>}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
