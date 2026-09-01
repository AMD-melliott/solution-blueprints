#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Generate diagnostic plots for the trained XGBoost model.

Plots produced (saved to viz.output_dir/xgb_*.png):
  xgb_feature_importance.png – top-30 features by gain
  xgb_roc_pr_curves.png      – ROC and Precision-Recall curves
  xgb_confusion_matrix.png   – confusion matrix heatmap
  xgb_score_dist.png         – fraud score distribution

Usage:
  python -m pipeline.xgb_visualize
  python -m pipeline.xgb_visualize -i artifacts/model.pt
"""

import argparse
import os
import sys

import joblib
import matplotlib
import numpy as np
import torch
import xgboost  # noqa: F401  # must import before torch: avoids libomp double-load segfault on macOS
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .xgb_train import build_feature_matrix, read_bundle


def _savefig(fig, path: str, dpi: int, show: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved   → {path}", file=sys.stderr)
    import matplotlib.pyplot as plt

    if show:
        plt.show()
    plt.close(fig)


def plot_feature_importance(model, feat_names: list, cfg: dict, out_dir: str, top_n: int = 30) -> None:
    import matplotlib.pyplot as plt

    scores = model.get_booster().get_score(importance_type="gain")
    # Map f0, f1, ... names back to actual feature names
    named = {}
    for k, v in scores.items():
        idx = int(k[1:]) if k.startswith("f") and k[1:].isdigit() else None
        name = feat_names[idx] if idx is not None and idx < len(feat_names) else k
        named[name] = v

    items = sorted(named.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*items) if items else ([], [])

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.3)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel="Gain", title=f"XGBoost Feature Importance (top {len(names)})")
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "xgb_feature_importance.png"), cfg["viz"]["dpi"], cfg["viz"]["show"])


def plot_roc_pr(proba: np.ndarray, y_true: np.ndarray, cfg: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    roc_auc = roc_auc_score(y_true, proba)
    auc_pr = average_precision_score(y_true, proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("XGBoost — Test-Set Curves", fontsize=13)

    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, color="C2", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[0].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_true, proba)
    axes[1].plot(rec, prec, color="C3", lw=2, label=f"AUC-PR = {auc_pr:.4f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "xgb_roc_pr_curves.png"), cfg["viz"]["dpi"], cfg["viz"]["show"])


def plot_confusion(proba: np.ndarray, y_true: np.ndarray, threshold: float, cfg: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], ax=ax
    )
    ax.set(title=f"XGBoost Confusion Matrix (threshold={threshold})", ylabel="True", xlabel="Predicted")
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "xgb_confusion_matrix.png"), cfg["viz"]["dpi"], cfg["viz"]["show"])


def plot_score_dist(proba: np.ndarray, y_true: np.ndarray, cfg: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(proba[y_true == 0], bins=80, alpha=0.6, color="C2", density=True, label="Legit")
    ax.hist(proba[y_true == 1], bins=80, alpha=0.6, color="C3", density=True, label="Fraud")
    ax.set(xlabel="Fraud score", ylabel="Density", title="XGBoost Score Distribution — Test Set")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "xgb_score_dist.png"), cfg["viz"]["dpi"], cfg["viz"]["show"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate XGBoost diagnostic plots")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--input", "-i", default=None, help="GNN bundle path (model.pt)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if not cfg["viz"].get("show", False):
        matplotlib.use("Agg")

    artifacts_dir = cfg["data"]["artifacts_dir"]
    out_dir = cfg["viz"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    inp = args.input or os.path.join(artifacts_dir, "model.pt")
    print(f"Reading bundle : {inp}", file=sys.stderr)
    bundle = read_bundle(inp)

    xgb_path = os.path.join(artifacts_dir, "xgb_model.joblib")
    print(f"Reading model  : {xgb_path}", file=sys.stderr)
    xgb_model = joblib.load(xgb_path)

    _, X_test, _, y_test, feat_names = build_feature_matrix(bundle, cfg)

    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    proba = xgb_model.predict_proba(X_test)[:, 1]

    plot_feature_importance(xgb_model, feat_names, cfg, out_dir)
    plot_roc_pr(proba, y_test, cfg, out_dir)
    plot_confusion(proba, y_test, threshold, cfg, out_dir)
    plot_score_dist(proba, y_test, cfg, out_dir)

    print(f"\nAll XGBoost plots saved to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
