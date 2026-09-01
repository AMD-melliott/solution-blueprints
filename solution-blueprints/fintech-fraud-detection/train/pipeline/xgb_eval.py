#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Evaluate the trained XGBoost model on the temporal test split.

Loads artifacts/model.pt (for graph features, embeddings, test_idx)
and artifacts/xgb_model.joblib (trained model).

Output:
  artifacts/xgb_metrics.json
  artifacts/xgb_predictions.csv

Usage:
  python -m pipeline.xgb_eval
  python -m pipeline.xgb_eval -i artifacts/model.pt -o artifacts/xgb_metrics.json
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd
import torch
import yaml

from .xgb_train import build_feature_matrix, compute_metrics, read_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XGBoost on test nodes")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--input", "-i", default=None, help="GNN bundle path (model.pt)")
    parser.add_argument("--output", "-o", default=None, help="metrics JSON output path")
    parser.add_argument("--predictions", "-p", default=None, help="predictions CSV path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    artifacts_dir = cfg["data"]["artifacts_dir"]

    inp = args.input or os.path.join(artifacts_dir, "model.pt")
    print(f"Reading bundle : {inp}", file=sys.stderr)
    bundle = read_bundle(inp)

    xgb_path = os.path.join(artifacts_dir, "xgb_model.joblib")
    print(f"Reading model  : {xgb_path}", file=sys.stderr)
    xgb_model = joblib.load(xgb_path)

    _, X_test, _, y_test, _ = build_feature_matrix(bundle, cfg)

    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    metrics = compute_metrics(xgb_model, X_test, y_test, threshold)

    print("\n=== XGBoost Test-Set Metrics ===", file=sys.stderr)
    print(f"ROC-AUC : {metrics['roc_auc']:.4f}", file=sys.stderr)
    print(f"AUC-PR  : {metrics['auc_pr']:.4f}", file=sys.stderr)
    print(f"F1      : {metrics['f1']:.4f}", file=sys.stderr)
    print(f"Acc     : {metrics['accuracy']:.4f}", file=sys.stderr)
    print(metrics["classification_report"], file=sys.stderr)

    out = args.output or os.path.join(artifacts_dir, "xgb_metrics.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved   → {out}", file=sys.stderr)

    pred_out = args.predictions or os.path.join(artifacts_dir, "xgb_predictions.csv")
    proba = xgb_model.predict_proba(X_test)[:, 1]
    test_idx = bundle["test_idx"]
    tx_ids = bundle.get("transaction_ids")
    tx_dts = bundle.get("transaction_dts")
    data: dict = {}
    if tx_ids is not None:
        data["TransactionID"] = tx_ids[test_idx]
    if tx_dts is not None:
        data["TransactionDT"] = tx_dts[test_idx]
    data.update(
        y_true=y_test.astype(int),
        y_prob=proba,
        y_pred=(proba >= threshold).astype(int),
    )
    os.makedirs(os.path.dirname(os.path.abspath(pred_out)), exist_ok=True)
    pd.DataFrame(data).to_csv(pred_out, index=False)
    print(f"Preds   → {pred_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
