// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

import styles from "./FraudDrawer.module.css";
import type { ScoredTransaction, FraudDetail } from "../../types/api";
import { fmtDT } from "../../utils/format";

interface FraudDrawerProps {
  transaction: ScoredTransaction | null;
  fraudThreshold: number;
  onClose: () => void;
}

function buildDetail(txn: ScoredTransaction, fraudThreshold: number): FraudDetail {
  const gtKnown = txn.is_fraud_gt !== null;
  const gtLabel = gtKnown
    ? `${txn.is_fraud_gt === 1 ? "fraud" : "clean"} · ${txn.ground_truth_match ? "match" : "mismatch"}`
    : "unknown";

  const e = txn.explainability;

  return {
    // Real fields, sourced from the transaction:
    transaction_id: txn.transaction_id,
    timestamp: fmtDT(txn.transaction_dt),
    amount: txn.transaction_amt,
    card_label: txn.card_label,
    latency_ms: txn.latency_ms,
    score: txn.score,
    decision: txn.decision,
    ground_truth: gtLabel,
    fraud_threshold: fraudThreshold,
    product: txn.product ?? "—",
    card_type: txn.card_type ?? "—",
    billing_region_code: txn.billing_region_code != null ? `addr1 ${txn.billing_region_code}` : "—",
    billing_country_code: txn.billing_country_code != null ? `addr2 ${txn.billing_country_code}` : "—",
    email_domain: txn.email_domain ?? "—",
    device_type: txn.device_type ?? "—",
    device_info: txn.device_info ?? "N/A",
    triggered_rules: e?.triggered_rules ?? [],
  };
}

function Row({ label, value, dim }: { label: string; value: string; dim?: boolean }) {
  return (
    <div className={styles.row}>
      <span className={styles.rowLabel}>{label}</span>
      <span className={`${styles.rowValue} ${dim ? styles.rowValueDim : ""}`}>{value}</span>
    </div>
  );
}

function decisionPillClass(d: FraudDetail["decision"]): string {
  if (d === "FRAUD") return styles.pillFraud;
  if (d === "REVIEW") return styles.pillReview;
  return styles.pillClean;
}
function decisionLabel(d: FraudDetail["decision"]): string {
  if (d === "FRAUD") return "Fraud";
  if (d === "REVIEW") return "Review";
  return "Not Fraud";
}
function scoreCardClass(d: FraudDetail["decision"]): string {
  if (d === "FRAUD") return styles.scoreCardFraud;
  if (d === "REVIEW") return styles.scoreCardReview;
  return styles.scoreCardClean;
}

export function FraudDrawer({ transaction, fraudThreshold, onClose }: FraudDrawerProps) {
  if (!transaction) return null;
  const d = buildDetail(transaction, fraudThreshold);
  const fmtAmount = `$${d.amount.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <>
      <div className={styles.backdrop} onClick={onClose} />
      <aside className={styles.drawer} role='dialog' aria-label='Fraud detail'>
        <div className={styles.header}>
          <h2 className={styles.title}>Fraud Detail</h2>
          <button className={styles.close} onClick={onClose} aria-label='Close'>
            ✕
          </button>
        </div>

        <div className={styles.body}>
          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Transaction</h3>
            <Row label='ID' value={`TXN-${d.transaction_id}`} />
            <Row label='Timestamp' value={d.timestamp} />
            <Row label='Amount' value={fmtAmount} />
            <Row label='Product' value={d.product} />
            <Row label='Card' value={d.card_label} />
            <Row label='Card type' value={d.card_type} />
            <Row label='Billing region code' value={d.billing_region_code} />
            <Row label='Billing country code' value={d.billing_country_code} />
            <Row label='Email domain' value={d.email_domain} />
            <Row label='Device type' value={d.device_type} />
            <Row label='Device info' value={d.device_info} />
            <Row label='Latency' value={`${d.latency_ms}ms`} />
            <Row label='Ground truth' value={d.ground_truth} />
          </section>

          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Model score</h3>
            <div className={`${styles.scoreCard} ${scoreCardClass(d.decision)}`}>
              <div className={styles.scoreLabel}>Score</div>
              <div className={styles.scoreRow}>
                <span className={styles.scoreVal}>{d.score.toFixed(2)}</span>
                <span className={`${styles.pill} ${decisionPillClass(d.decision)}`}>{decisionLabel(d.decision)}</span>
              </div>
              <div className={styles.scoreThreshold}>Threshold: Fraud ≥ {d.fraud_threshold.toFixed(2)}</div>
            </div>
          </section>

          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Explainability</h3>

            <div className={styles.subCard}>
              <div className={styles.subTitle}>Rule Engine</div>
              {d.triggered_rules.length === 0 ? (
                <div className={styles.rowValue}>No rules triggered</div>
              ) : (
                d.triggered_rules.map((rule) => (
                  <div key={rule.rule_id} className={styles.ruleItem}>
                    <div className={styles.ruleHead}>
                      <span className={styles.ruleId}>{rule.rule_id}</span>
                      <span className={`${styles.riskPill} ${styles[`risk${rule.risk_level}`]}`}>
                        {rule.risk_level}
                      </span>
                      <span className={styles.ruleAction}>{rule.action}</span>
                    </div>
                    <div className={styles.ruleDesc}>{rule.description}</div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </aside>
    </>
  );
}
