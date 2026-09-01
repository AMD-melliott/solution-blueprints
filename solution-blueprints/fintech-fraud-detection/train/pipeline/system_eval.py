#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
End-to-end evaluation of GNN + XGBoost + Rules Engine.

Replays the temporal test split in chronological order so that velocity-based
rules (R005, R006, R012, R015, R020, R021, R023, R024) accumulate per-card
state exactly as they would in production.

Metrics reported
----------------
  decision_distribution
      APPROVE / REVIEW / DECLINE counts overall, for fraud, and for legit.

  system_metrics  vs  ml_alone_metrics
      Binary classification at two operating points:
        strict  — DECLINE = predicted fraud
        lenient — DECLINE + REVIEW = predicted fraud
      Reported for combined system AND ML-only (rules disabled) side-by-side.

  rules_impact
      How many decisions the rules layer changed vs ML alone; hard-blocked
      transactions (ML skipped entirely); precision of the hard-block gate.

  per_rule
      For each rule that fired: total fires, % of test set, precision
      (fraud / fires), recall (fraud caught / total fraud).

Output
------
  artifacts/system_metrics.json          — full metrics JSON
  artifacts/system_predictions.csv       — per-transaction decisions
  artifacts/plots/system_roc_pr_curves.png — ROC and PR curves with operating points

Usage
-----
  python -m pipeline.system_eval
  python -m pipeline.system_eval --n 5000        # first N chronological test rows
  python -m pipeline.system_eval --mode gnn_only
"""

import argparse
import json
import math
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# app/ is a sibling of pipeline/ — ensure it is importable when running as
# `python -m pipeline.system_eval` from the service root directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.predictor import CombinedPredictor  # noqa: E402

# ── Data helpers ──────────────────────────────────────────────────────────────


def load_test_split(cfg: dict) -> pd.DataFrame:
    data_dir = cfg["data"]["data_dir"]
    sort_col = cfg["split"]["sort_col"]
    test_frac = cfg["split"]["test_fraction"]

    tx_path = os.path.join(data_dir, "train_transaction.csv")
    id_path = os.path.join(data_dir, "train_identity.csv")

    print(f"Loading {tx_path} ...", file=sys.stderr)
    df = pd.read_csv(tx_path)
    if os.path.exists(id_path):
        df = df.merge(pd.read_csv(id_path), on="TransactionID", how="left")

    df = df.sort_values(sort_col).reset_index(drop=True)
    n_train = int(len(df) * (1.0 - test_frac))
    test_df = df.iloc[n_train:].reset_index(drop=True)

    print(
        f"Test split : {len(test_df):,} rows  " f"fraud={test_df['isFraud'].sum():,} ({test_df['isFraud'].mean():.2%})",
        file=sys.stderr,
    )
    return test_df


def row_to_tx(row: pd.Series) -> dict:
    exclude = {"isFraud"}
    return {
        k: (None if (isinstance(v, float) and math.isnan(v)) else v)
        for k, v in row.to_dict().items()
        if k not in exclude
    }


# ── Metric helpers ────────────────────────────────────────────────────────────


def binary_metrics(y_true, y_prob, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")
    try:
        auc_pr = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auc_pr = float("nan")
    return {
        "roc_auc": round(roc_auc, 6),
        "auc_pr": round(auc_pr, 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def decision_dist(series: pd.Series) -> dict:
    total = len(series)
    counts = series.value_counts().to_dict()
    return {
        d: {"n": counts.get(d, 0), "pct": round(counts.get(d, 0) / max(total, 1) * 100, 2)}
        for d in ["APPROVE", "REVIEW", "DECLINE"]
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────


def plot_roc_pr_curves(df_res: pd.DataFrame, y_true, y_prob, metrics: dict, plot_dir: str) -> str:
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    auc = metrics["system_metrics"]["strict_decline_only"]["roc_auc"]

    axes[0].plot(fpr_arr, tpr_arr, lw=1.5, label=f"ROC AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)

    for op_label, col, marker, color in [
        ("strict (DECLINE)", "decision", "o", "tab:red"),
        ("lenient (DECLINE+REVIEW)", "decision", "s", "tab:orange"),
    ]:
        if op_label.startswith("strict"):
            y_pred = (df_res["decision"] == "DECLINE").astype(int)
        else:
            y_pred = df_res["decision"].isin(["DECLINE", "REVIEW"]).astype(int)
        from sklearn.metrics import confusion_matrix as _cm

        tn, fp, fn, tp = _cm(y_true, y_pred).ravel()
        op_fpr = fp / max(fp + tn, 1)
        op_tpr = tp / max(tp + fn, 1)
        axes[0].plot(
            op_fpr,
            op_tpr,
            marker=marker,
            color=color,
            ms=9,
            zorder=5,
            label=f"{op_label}\n  FPR={op_fpr:.3f} TPR={op_tpr:.3f}",
        )

    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve — System (ML + Rules)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # ── PR curve ──────────────────────────────────────────────────────────────
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = metrics["system_metrics"]["strict_decline_only"]["auc_pr"]
    baseline = sum(y_true) / max(len(y_true), 1)

    axes[1].plot(rec_arr, prec_arr, lw=1.5, label=f"AUC-PR = {auc_pr:.4f}")
    axes[1].axhline(baseline, color="k", ls="--", lw=0.8, label=f"Baseline = {baseline:.4f}")

    for op_label, marker, color in [
        ("strict (DECLINE)", "o", "tab:red"),
        ("lenient (DECLINE+REVIEW)", "s", "tab:orange"),
    ]:
        if op_label.startswith("strict"):
            sm = metrics["system_metrics"]["strict_decline_only"]
        else:
            sm = metrics["system_metrics"]["lenient_decline_or_review"]
        axes[1].plot(
            sm["recall"],
            sm["precision"],
            marker=marker,
            color=color,
            ms=9,
            zorder=5,
            label=f"{op_label}\n  P={sm['precision']:.3f} R={sm['recall']:.3f}",
        )

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — System (ML + Rules)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(plot_dir, "system_roc_pr_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end system evaluation: GNN + XGBoost + Rules Engine")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Evaluate only the first N test rows (chronological). Default: all.",
    )
    parser.add_argument(
        "--mode",
        default="xgb_ensemble",
        choices=["xgb_ensemble", "gnn_only", "graph"],
        help="ML scoring mode passed to CombinedPredictor.",
    )
    parser.add_argument("--output", "-o", default=None, help="metrics JSON output path")
    parser.add_argument("--predictions", "-p", default=None, help="predictions CSV path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    artifacts_dir = cfg["data"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    print("Loading predictor (GNN + XGBoost + Rules Engine)...", file=sys.stderr)
    predictor = CombinedPredictor(args.config)

    test_df = load_test_split(cfg)
    if args.n is not None:
        test_df = test_df.iloc[: args.n].copy()
        print(f"Truncated to first {len(test_df):,} rows.", file=sys.stderr)

    n = len(test_df)
    n_fraud = int(test_df["isFraud"].sum())

    # ── Scoring loop ──────────────────────────────────────────────────────────
    # Transactions are already in chronological order (sorted by TransactionDT).
    # Processing them sequentially lets velocity rules (R005, R006, etc.) build
    # per-card state just as they would in production.

    records = []
    rule_hits: dict[str, dict] = {}  # rule_id → {fires, fraud_fires, risk_level, action}

    t0 = time.time()
    for _, row in tqdm(test_df.iterrows(), total=n, desc="Scoring", unit="tx", file=sys.stderr):
        tx = row_to_tx(row)
        true_label = int(row["isFraud"])

        result = predictor.score(tx, mode=args.mode)

        prob = result["fraud_probability"]
        decision = result["decision"]
        rule_action = result.get("rule_action", "NONE")
        triggered = result.get("triggered_rules", [])

        # ML-alone decision: apply threshold to the probability the model returned.
        # For hard-blocked rows prob=1.0 (ML was skipped), so ml_decision=DECLINE
        # too — we mark them separately via rule_action=BLOCK.
        ml_decision = predictor._decide(prob)

        for r in triggered:
            rid = r["rule_id"]
            if rid not in rule_hits:
                rule_hits[rid] = {
                    "fires": 0,
                    "fraud_fires": 0,
                    "risk_level": r["risk_level"],
                    "action": r["action"],
                }
            rule_hits[rid]["fires"] += 1
            rule_hits[rid]["fraud_fires"] += true_label

        records.append(
            {
                "TransactionID": row.get("TransactionID"),
                "isFraud": true_label,
                "fraud_probability": prob,
                "decision": decision,
                "ml_decision": ml_decision,
                "rule_action": rule_action,
                "decision_changed": decision != ml_decision,
                "triggered_rules": ",".join(r["rule_id"] for r in triggered),
            }
        )

    elapsed = time.time() - t0
    df_res = pd.DataFrame(records)

    # ── Aggregate metrics ─────────────────────────────────────────────────────

    y_true = df_res["isFraud"].tolist()
    y_prob = df_res["fraud_probability"].tolist()

    # Combined system predictions
    y_pred_strict = (df_res["decision"] == "DECLINE").astype(int).tolist()
    y_pred_lenient = df_res["decision"].isin(["DECLINE", "REVIEW"]).astype(int).tolist()

    # ML-alone predictions
    y_pred_ml_strict = (df_res["ml_decision"] == "DECLINE").astype(int).tolist()
    y_pred_ml_lenient = df_res["ml_decision"].isin(["DECLINE", "REVIEW"]).astype(int).tolist()

    # Per-rule stats
    per_rule = {}
    for rid, h in sorted(rule_hits.items()):
        fires = h["fires"]
        fraud_fires = h["fraud_fires"]
        per_rule[rid] = {
            "risk_level": h["risk_level"],
            "action": h["action"],
            "fires": fires,
            "fires_pct": round(fires / n * 100, 3),
            "fraud_fires": fraud_fires,
            "precision": round(fraud_fires / fires, 4) if fires else 0.0,
            "recall": round(fraud_fires / n_fraud, 4) if n_fraud else 0.0,
        }

    # Rules impact
    changed = df_res["decision_changed"]
    hard_blocked = df_res["rule_action"] == "BLOCK"
    hb_total = int(hard_blocked.sum())
    hb_fraud = int((hard_blocked & (df_res["isFraud"] == 1)).sum())
    hb_legit = int((hard_blocked & (df_res["isFraud"] == 0)).sum())

    metrics = {
        "n_test": n,
        "n_fraud": n_fraud,
        "fraud_rate": round(n_fraud / n, 6),
        "mode": args.mode,
        "elapsed_seconds": round(elapsed, 1),
        "tx_per_second": round(n / elapsed, 1),
        "decision_distribution": {
            "overall": decision_dist(df_res["decision"]),
            "fraud": decision_dist(df_res.loc[df_res["isFraud"] == 1, "decision"]),
            "legitimate": decision_dist(df_res.loc[df_res["isFraud"] == 0, "decision"]),
        },
        "system_metrics": {
            "strict_decline_only": binary_metrics(y_true, y_prob, y_pred_strict),
            "lenient_decline_or_review": binary_metrics(y_true, y_prob, y_pred_lenient),
        },
        "ml_alone_metrics": {
            "strict_decline_only": binary_metrics(y_true, y_prob, y_pred_ml_strict),
            "lenient_decline_or_review": binary_metrics(y_true, y_prob, y_pred_ml_lenient),
        },
        "rules_impact": {
            "decisions_changed": int(changed.sum()),
            "decisions_changed_pct": round(float(changed.mean()) * 100, 3),
            "changed_were_fraud": int((changed & (df_res["isFraud"] == 1)).sum()),
            "changed_were_legit": int((changed & (df_res["isFraud"] == 0)).sum()),
            "hard_blocked_total": hb_total,
            "hard_blocked_fraud": hb_fraud,
            "hard_blocked_legit": hb_legit,
            "hard_block_precision": round(hb_fraud / max(hb_total, 1), 4),
        },
        "per_rule": per_rule,
    }

    # ── Print summary ─────────────────────────────────────────────────────────

    W = 58
    print(f"\n{'═' * W}", file=sys.stderr)
    print("  System Evaluation — GNN + XGBoost + Rules Engine", file=sys.stderr)
    print(f"{'═' * W}", file=sys.stderr)
    print(f"  Test set : {n:,} tx   fraud={n_fraud:,} ({n_fraud/n:.2%})", file=sys.stderr)
    print(f"  Mode     : {args.mode}   {elapsed:.1f}s ({n/elapsed:.0f} tx/s)", file=sys.stderr)
    print(file=sys.stderr)

    # Decision distribution
    print("  Decision distribution", file=sys.stderr)
    print(f"  {'':10s}  {'APPROVE':>8}  {'REVIEW':>8}  {'DECLINE':>8}", file=sys.stderr)
    for label, key in [("Overall", "overall"), ("Fraud", "fraud"), ("Legit", "legitimate")]:
        dd = metrics["decision_distribution"][key]
        print(
            f"  {label:10s}  "
            f"{dd['APPROVE']['pct']:>7.1f}%  "
            f"{dd['REVIEW']['pct']:>7.1f}%  "
            f"{dd['DECLINE']['pct']:>7.1f}%",
            file=sys.stderr,
        )
    print(file=sys.stderr)

    # ML alone vs combined
    hdr = f"  {'':28s}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'AUC-PR':>7}"
    for op_label, sys_key, ml_key in [
        ("strict  (DECLINE = fraud)", "strict_decline_only", "strict_decline_only"),
        ("lenient (DECLINE+REVIEW = fraud)", "lenient_decline_or_review", "lenient_decline_or_review"),
    ]:
        sm = metrics["system_metrics"][sys_key]
        mm = metrics["ml_alone_metrics"][ml_key]
        print(f"  Operating point — {op_label}", file=sys.stderr)
        print(hdr, file=sys.stderr)
        print(
            f"  {'ML alone':28s}  {mm['f1']:>6.4f}  {mm['precision']:>6.4f}"
            f"  {mm['recall']:>6.4f}  {mm['auc_pr']:>7.4f}",
            file=sys.stderr,
        )
        print(
            f"  {'Combined (ML + rules)':28s}  {sm['f1']:>6.4f}  {sm['precision']:>6.4f}"
            f"  {sm['recall']:>6.4f}  {sm['auc_pr']:>7.4f}",
            file=sys.stderr,
        )
        print(file=sys.stderr)

    # Rules impact
    ri = metrics["rules_impact"]
    print("  Rules impact", file=sys.stderr)
    print(
        f"  Decisions changed : {ri['decisions_changed']:,} ({ri['decisions_changed_pct']:.2f}%)"
        f"  — fraud={ri['changed_were_fraud']}  legit={ri['changed_were_legit']}",
        file=sys.stderr,
    )
    print(
        f"  Hard blocked      : {ri['hard_blocked_total']:,}"
        f"  fraud={ri['hard_blocked_fraud']}  legit={ri['hard_blocked_legit']}"
        f"  precision={ri['hard_block_precision']:.2%}",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    # Per-rule table
    if per_rule:
        print("  Per-rule stats", file=sys.stderr)
        print(
            f"  {'ID':6}  {'Level':6}  {'Action':7}  {'Fires':>7}  {'%test':>5}  " f"{'Prec':>6}  {'Recall':>7}",
            file=sys.stderr,
        )
        for rid, rm in per_rule.items():
            print(
                f"  {rid:6}  {rm['risk_level']:6}  {rm['action']:7}  "
                f"{rm['fires']:>7,}  {rm['fires_pct']:>4.1f}%  "
                f"{rm['precision']:>6.4f}  {rm['recall']:>7.4f}",
                file=sys.stderr,
            )
    else:
        print("  No rules fired on this test split.", file=sys.stderr)

    print(f"{'═' * W}\n", file=sys.stderr)

    # ── Save outputs ──────────────────────────────────────────────────────────

    out_json = args.output or os.path.join(artifacts_dir, "system_metrics.json")
    out_csv = args.predictions or os.path.join(artifacts_dir, "system_predictions.csv")
    plot_dir = cfg.get("viz", {}).get("output_dir", os.path.join(artifacts_dir, "plots"))

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    df_res.to_csv(out_csv, index=False)

    out_plot = plot_roc_pr_curves(df_res, y_true, y_prob, metrics, plot_dir)

    print(f"Saved metrics     → {out_json}", file=sys.stderr)
    print(f"Saved predictions → {out_csv}", file=sys.stderr)
    print(f"Saved ROC/PR plot → {out_plot}", file=sys.stderr)


if __name__ == "__main__":
    main()
