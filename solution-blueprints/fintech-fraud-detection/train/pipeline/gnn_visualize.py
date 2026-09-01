#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Generate diagnostic plots for the trained GNN.

Plots produced (saved to viz.output_dir/gnn_*.png):
  gnn_training_curves.png  – loss, LR, val ROC-AUC, val PR-AUC / F1
  gnn_roc_pr_curves.png    – ROC and Precision-Recall on the test split
  gnn_confusion_matrix.png – confusion matrix heatmap
  gnn_score_dist.png       – fraud score distribution (legit vs fraud)

Usage:
  python -m pipeline.gnn_visualize
  python -m pipeline.gnn_visualize -i artifacts/model.pt
"""

import argparse
import io
import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .model import FraudGNN


def _decimate(y: list, max_pts: int):
    n = len(y)
    if n <= max_pts:
        return np.arange(1, n + 1), np.asarray(y, dtype=float)
    idx = np.linspace(0, n - 1, max_pts).astype(int)
    return idx + 1, np.asarray(y, dtype=float)[idx]


def _savefig(fig, path: str, dpi: int, show: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved   → {path}", file=sys.stderr)
    import matplotlib.pyplot as plt

    if show:
        plt.show()
    plt.close(fig)


def plot_training_curves(history: dict, cfg: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    max_pts = cfg["viz"]["plot_max_points"]
    be = int(history.get("best_epoch", 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GNN Training History", fontsize=13)

    ep, tl = _decimate(history["train_loss"], max_pts)
    _, vl = _decimate(history["val_loss"], max_pts)
    axes[0, 0].plot(ep, tl, label="train")
    axes[0, 0].plot(ep, vl, label="val")
    axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Cross-entropy loss")
    axes[0, 0].legend()

    ep2, lr = _decimate(history["lr"], max_pts)
    axes[0, 1].plot(ep2, lr, color="C4")
    axes[0, 1].set(xlabel="Epoch", ylabel="LR", title="Learning rate")
    axes[0, 1].set_yscale("log")

    ep3, vauc = _decimate(history["val_auc"], max_pts)
    axes[1, 0].plot(ep3, vauc, color="C2", label="val ROC-AUC")
    axes[1, 0].axvline(be, color="gray", ls="--", lw=1, label=f"best @ ep {be:,}")
    lo = np.nanmin(history["val_auc"])
    axes[1, 0].set_ylim(max(0.0, lo - 0.05), 1.01)
    axes[1, 0].set(xlabel="Epoch", ylabel="ROC-AUC", title="Validation ROC-AUC")
    axes[1, 0].legend(loc="lower right")

    ep4, aupr = _decimate(history["val_aupr"], max_pts)
    _, vf1 = _decimate(history["val_f1"], max_pts)
    axes[1, 1].plot(ep4, aupr, color="C5", label="val AUC-PR")
    axes[1, 1].plot(ep4, vf1, color="C3", alpha=0.7, label="val F1 (fraud)")
    axes[1, 1].axvline(be, color="gray", ls="--", lw=1)
    axes[1, 1].set_ylim(0, 1.01)
    axes[1, 1].set(xlabel="Epoch", ylabel="Score", title="PR-AUC + F1 (fraud class)")
    axes[1, 1].legend(loc="lower right")

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "gnn_training_curves.png"), cfg["viz"]["dpi"], cfg["viz"]["show"])


def plot_roc_pr(proba: np.ndarray, y_true: np.ndarray, cfg: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    roc_auc = roc_auc_score(y_true, proba)
    auc_pr = average_precision_score(y_true, proba)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("GNN — Test-Set Curves", fontsize=13)

    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, color="C0", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[0].legend(loc="lower right")

    prec, rec, _ = precision_recall_curve(y_true, proba)
    axes[1].plot(rec, prec, color="C1", lw=2, label=f"AUC-PR = {auc_pr:.4f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "gnn_roc_pr_curves.png"), cfg["viz"]["dpi"], cfg["viz"]["show"])


def plot_confusion(proba: np.ndarray, y_true: np.ndarray, threshold: float, cfg: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], ax=ax
    )
    ax.set(title=f"GNN Confusion Matrix (threshold={threshold})", ylabel="True", xlabel="Predicted")
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "gnn_confusion_matrix.png"), cfg["viz"]["dpi"], cfg["viz"]["show"])


def plot_score_dist(proba: np.ndarray, y_true: np.ndarray, cfg: dict, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(proba[y_true == 0], bins=80, alpha=0.6, color="C0", density=True, label="Legit")
    ax.hist(proba[y_true == 1], bins=80, alpha=0.6, color="C3", density=True, label="Fraud")
    ax.set(xlabel="Fraud score", ylabel="Density", title="GNN Score Distribution — Test Set")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "gnn_score_dist.png"), cfg["viz"]["dpi"], cfg["viz"]["show"])


def get_test_proba(bundle: dict, device: torch.device, cfg: dict):
    graph_data = bundle["graph"].to(device)
    test_idx = bundle["test_idx"]
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

    with torch.no_grad():
        logits, _ = model(graph_data.x, graph_data.edge_index)

    proba = F.softmax(logits[test_idx].cpu(), dim=1)[:, 1].numpy()
    y_true = graph_data.y[test_idx].cpu().numpy()
    return proba, y_true


def read_bundle(src: str) -> dict:
    if src == "-":
        return torch.load(io.BytesIO(sys.stdin.buffer.read()), weights_only=False)
    return torch.load(src, weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GNN diagnostic plots")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--input", "-i", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if not cfg["viz"].get("show", False):
        matplotlib.use("Agg")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg["viz"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    inp = args.input or os.path.join(cfg["data"]["artifacts_dir"], "model.pt")
    print(f"Reading : {inp if inp != '-' else 'stdin'}", file=sys.stderr)
    bundle = read_bundle(inp)

    cfg_merged = bundle["config"] if "config" in bundle else cfg
    threshold = cfg_merged.get("eval", {}).get("threshold", 0.5)

    if "history" in bundle:
        plot_training_curves(bundle["history"], cfg_merged, out_dir)
    else:
        print("No training history — skipping training curves", file=sys.stderr)

    proba, y_true = get_test_proba(bundle, device, cfg_merged)
    plot_roc_pr(proba, y_true, cfg_merged, out_dir)
    plot_confusion(proba, y_true, threshold, cfg_merged, out_dir)
    plot_score_dist(proba, y_true, cfg_merged, out_dir)

    print(f"\nAll GNN plots saved to {out_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
