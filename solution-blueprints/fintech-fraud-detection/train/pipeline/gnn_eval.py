#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Evaluate the trained GNN on the temporal test split.

Usage:
  python -m pipeline.gnn_eval
  python -m pipeline.gnn_eval -i artifacts/model.pt -o artifacts/gnn_metrics.json
  python -m pipeline.gnn_eval --inductive   # use train↔test edges, no test-test leakage
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .model import FraudGNN


def run_inference(bundle: dict, device: torch.device, cfg: dict, split: str = "test", inductive: bool = False):
    graph_data = bundle["graph"].to(device)
    model_cfg = bundle.get("model_cfg", cfg["model"])

    model = FraudGNN(
        in_channels=graph_data.num_node_features,
        hidden=model_cfg["hidden_channels"],
        embed_dim=model_cfg["embed_dim"],
        head_hidden=model_cfg["head_hidden"],
        dropout=model_cfg["dropout"],
    ).to(device)
    model.load_state_dict({k: v.to(device) for k, v in bundle["model_state"].items()})
    model.eval()

    if inductive:
        ind_ei = bundle.get("inductive_edge_index")
        if ind_ei is None:
            raise RuntimeError("inductive_edge_index not found — rebuild graph")
        graph_data = graph_data.clone()
        graph_data.edge_index = ind_ei.to(device)
        print(f"Edges   : inductive ({graph_data.edge_index.shape[1]:,} endpoints)", file=sys.stderr)

    with torch.no_grad():
        logits, _ = model(graph_data.x, graph_data.edge_index)

    def _extract(idx):
        proba = F.softmax(logits[idx].cpu(), dim=1)[:, 1].numpy()
        y_true = graph_data.y[idx].cpu().numpy()
        return proba, y_true

    test_idx = bundle["test_idx"]
    val_mask = getattr(graph_data, "val_mask", None)
    val_idx = val_mask.nonzero(as_tuple=True)[0].cpu().numpy() if val_mask is not None else None

    if split == "val":
        if val_idx is None:
            raise RuntimeError("val_mask not found — was the model trained?")
        vp, vy = _extract(val_idx)
        return vp, vy, val_idx
    elif split == "both":
        if val_idx is None:
            raise RuntimeError("val_mask not found — was the model trained?")
        vp, vy = _extract(val_idx)
        tp, ty = _extract(test_idx)
        return np.concatenate([vp, tp]), np.concatenate([vy, ty]), np.concatenate([val_idx, test_idx])
    else:
        tp, ty = _extract(test_idx)
        return tp, ty, test_idx


def compute_metrics(proba: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    y_pred = (proba >= threshold).astype(int)
    try:
        roc_auc = float(roc_auc_score(y_true, proba))
    except ValueError:
        roc_auc = float("nan")
    try:
        auc_pr = float(average_precision_score(y_true, proba))
    except ValueError:
        auc_pr = float("nan")

    return dict(
        roc_auc=round(roc_auc, 6),
        auc_pr=round(auc_pr, 6),
        f1=round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 6),
        accuracy=round(float((y_pred == y_true).mean()), 6),
        threshold=threshold,
        n_test=int(len(y_true)),
        n_fraud=int(y_true.sum()),
        fraud_rate=round(float(y_true.mean()), 6),
        classification_report=classification_report(y_true, y_pred, target_names=["Legit", "Fraud"], output_dict=False),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def read_bundle(src: str, device: torch.device) -> dict:
    kw = dict(weights_only=False, map_location=device)
    if src == "-":
        return torch.load(io.BytesIO(sys.stdin.buffer.read()), **kw)
    return torch.load(src, **kw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GNN on test nodes")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--input", "-i", default=None)
    parser.add_argument("--output", "-o", default=None, help="metrics JSON path")
    parser.add_argument("--predictions", "-p", default=None, help="predictions CSV path")
    parser.add_argument("--split", "-s", default="test", choices=["test", "val", "both"])
    parser.add_argument(
        "--no-inductive",
        dest="inductive",
        action="store_false",
        default=True,
        help="legacy: score test nodes with edges dropped (isolated). The "
        "default now aggregates test nodes over their train neighbours "
        "(inductive_edge_index), matching the served/exported regime.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}", file=sys.stderr)

    artifacts_dir = cfg["data"]["artifacts_dir"]
    inp = args.input or os.path.join(artifacts_dir, "model.pt")
    print(f"Reading : {inp if inp != '-' else 'stdin'}", file=sys.stderr)
    bundle = read_bundle(inp, device)

    cfg_merged = bundle["config"] if "config" in bundle else cfg
    threshold = cfg_merged.get("eval", {}).get("threshold", 0.5)

    proba, y_true, used_idx = run_inference(bundle, device, cfg_merged, split=args.split, inductive=args.inductive)
    metrics = compute_metrics(proba, y_true, threshold)

    print("\n=== GNN Test-Set Metrics ===", file=sys.stderr)
    print(f"ROC-AUC : {metrics['roc_auc']:.4f}", file=sys.stderr)
    print(f"AUC-PR  : {metrics['auc_pr']:.4f}", file=sys.stderr)
    print(f"F1      : {metrics['f1']:.4f}", file=sys.stderr)
    print(f"Acc     : {metrics['accuracy']:.4f}", file=sys.stderr)
    print(metrics["classification_report"], file=sys.stderr)

    out = args.output or os.path.join(artifacts_dir, "gnn_metrics.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved   → {out}", file=sys.stderr)

    pred_out = args.predictions or os.path.join(artifacts_dir, "gnn_predictions.csv")
    tx_ids = bundle.get("transaction_ids")
    tx_dts = bundle.get("transaction_dts")
    data: dict = {}
    if tx_ids is not None:
        data["TransactionID"] = tx_ids[used_idx]
    if tx_dts is not None:
        data["TransactionDT"] = tx_dts[used_idx]
    data.update(y_true=y_true.astype(int), y_prob=proba, y_pred=(proba >= threshold).astype(int))
    os.makedirs(os.path.dirname(os.path.abspath(pred_out)), exist_ok=True)
    pd.DataFrame(data).to_csv(pred_out, index=False)
    print(f"Preds   → {pred_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
