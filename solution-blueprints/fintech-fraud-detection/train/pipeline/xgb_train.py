#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Train XGBoost on tabular features + GNN embeddings.

Reads the GNN bundle (artifacts/model.pt) which already contains:
  graph.x    – normalized tabular feature matrix [n_nodes, n_feats]
  graph.y    – labels [n_nodes]
  embeddings – GNN encoder output [n_nodes, embed_dim]
  train_idx / test_idx – temporal split indices

Stacks tabular + GNN embeddings and trains XGBoost with early stopping.

Output artifacts:
  artifacts/xgb_model.joblib   — trained XGBoost classifier
  artifacts/xgb_features.json  — ordered feature name list
  artifacts/xgb_metrics.json   — test-set metrics

Usage:
  python -m pipeline.xgb_train
  python -m pipeline.xgb_train -i artifacts/model.pt
"""

import argparse
import io
import json
import os
import sys

import joblib
import numpy as np
import torch
import xgboost  # noqa: F401  # must import before torch: avoids libomp double-load segfault on macOS
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def read_bundle(src: str) -> dict:
    if src == "-":
        return torch.load(io.BytesIO(sys.stdin.buffer.read()), weights_only=False)
    return torch.load(src, weights_only=False)


def build_feature_matrix(bundle: dict, cfg: dict):
    """
    Returns (X_train, X_test, y_train, y_test, feature_names).
    Tabular features from graph.x (already normalized) are concatenated
    with GNN embeddings so training and online inference are consistent.
    """
    graph = bundle["graph"]
    train_idx = bundle["train_idx"]
    test_idx = bundle["test_idx"]
    tabular_names = list(bundle["feature_cols"])

    X_tab = graph.x.cpu().numpy()  # (n_nodes, n_tabular)
    y = graph.y.cpu().numpy()  # (n_nodes,)

    use_emb = cfg.get("xgb", {}).get("use_gnn_embeddings", True)
    if use_emb and "embeddings" in bundle:
        emb = bundle["embeddings"]  # (n_nodes, embed_dim)
        n_emb = emb.shape[1]
        emb_names = [f"gnn_emb_{i}" for i in range(n_emb)]
        X = np.hstack([X_tab, emb]).astype(np.float32)
        feat_names = tabular_names + emb_names
        print(
            f"Features: {len(tabular_names)} tabular + {n_emb} GNN embeddings" f" = {len(feat_names)} total",
            file=sys.stderr,
        )
    else:
        if use_emb:
            print(
                "WARNING: use_gnn_embeddings=true but 'embeddings' not in bundle. "
                "Run gnn_train first to generate embeddings.",
                file=sys.stderr,
            )
        X = X_tab.astype(np.float32)
        feat_names = tabular_names
        print(f"Features: {len(feat_names)} tabular only", file=sys.stderr)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Train   : {X_train.shape}  fraud={y_train.mean():.3%}", file=sys.stderr)
    print(f"Test    : {X_test.shape}   fraud={y_test.mean():.3%}", file=sys.stderr)

    return X_train, X_test, y_train, y_test, feat_names


def _xgb_device(xcfg: dict) -> str:
    """
    Resolve XGBoost device. "auto" picks "cuda" when torch sees a GPU
    (works for both NVIDIA CUDA and AMD ROCm via HIP), else "cpu".
    XGBoost >= 2.0 uses device="cuda" for GPU (replaces tree_method="gpu_hist").
    """
    device = xcfg.get("device", "auto")
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"XGB device : {device}", file=sys.stderr)
    return device


def train_xgb(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
) -> XGBClassifier:
    xcfg = cfg.get("xgb", {})
    neg = max(int((y_train == 0).sum()), 1)
    pos = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight = {scale_pos_weight:.2f}  (neg={neg:,}, pos={pos:,})", file=sys.stderr)

    device = _xgb_device(xcfg)
    model = XGBClassifier(
        n_estimators=xcfg.get("n_estimators", 1000),
        max_depth=xcfg.get("max_depth", 6),
        learning_rate=xcfg.get("learning_rate", 0.05),
        subsample=xcfg.get("subsample", 0.8),
        colsample_bytree=xcfg.get("colsample_bytree", 0.8),
        scale_pos_weight=scale_pos_weight,
        eval_metric=xcfg.get("eval_metric", "aucpr"),
        early_stopping_rounds=xcfg.get("early_stopping_rounds", 30),
        random_state=xcfg.get("random_state", cfg.get("training", {}).get("seed", 42)),
        # device="cuda" works for both NVIDIA CUDA and AMD ROCm (HIP compat layer)
        # n_jobs is ignored on GPU; kept for CPU fallback
        device=device,
        tree_method="hist",
        n_jobs=xcfg.get("n_jobs", -1) if device == "cpu" else 1,
    )

    # Early-stop on the VAL tail (carved from train), NOT the test set: selecting
    # best_iteration on test leaks it into the reported metric (optimistic + noisy).
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"Best iteration : {model.best_iteration}", file=sys.stderr)
    return model


def compute_metrics(model: XGBClassifier, X: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    try:
        roc_auc = float(roc_auc_score(y, proba))
    except ValueError:
        roc_auc = float("nan")
    try:
        auc_pr = float(average_precision_score(y, proba))
    except ValueError:
        auc_pr = float("nan")

    return dict(
        roc_auc=round(roc_auc, 6),
        auc_pr=round(auc_pr, 6),
        f1=round(float(f1_score(y, y_pred, pos_label=1, zero_division=0)), 6),
        accuracy=round(float(accuracy_score(y, y_pred)), 6),
        threshold=threshold,
        n_test=int(len(y)),
        n_fraud=int(y.sum()),
        fraud_rate=round(float(y.mean()), 6),
        best_iteration=int(model.best_iteration) if model.best_iteration is not None else -1,
        classification_report=classification_report(y, y_pred, target_names=["Legit", "Fraud"], output_dict=False),
        confusion_matrix=confusion_matrix(y, y_pred).tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost on tabular + GNN embeddings")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--input", "-i", default=None, help="GNN bundle path (model.pt)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    artifacts_dir = cfg["data"]["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    inp = args.input or os.path.join(artifacts_dir, "model.pt")
    print(f"Reading : {inp}", file=sys.stderr)
    bundle = read_bundle(inp)

    X_train, X_test, y_train, y_test, feat_names = build_feature_matrix(bundle, cfg)

    # Carve a temporal validation tail from TRAIN for early stopping (mirrors the
    # GNN's val split, gnn_train._attach_masks). X_train is in temporal order
    # (train_idx = arange over the DT-sorted frame), so the tail is the latest
    # train rows. The test set is never used for fitting or early stopping, so the
    # reported test metric is honest (no selection-on-test leak).
    val_fraction = cfg.get("training", {}).get("val_fraction", 0.15)
    n_val = max(1, int(round(len(X_train) * val_fraction)))
    X_fit, X_val = X_train[:-n_val], X_train[-n_val:]
    y_fit, y_val = y_train[:-n_val], y_train[-n_val:]
    print(f"XGB fit : {X_fit.shape}  fraud={y_fit.mean():.3%}", file=sys.stderr)
    print(f"XGB val : {X_val.shape}  fraud={y_val.mean():.3%}  (temporal tail, early-stopping set)", file=sys.stderr)

    model = train_xgb(X_fit, X_val, y_fit, y_val, cfg)

    threshold = cfg.get("eval", {}).get("threshold", 0.5)
    val_metrics = compute_metrics(model, X_val, y_val, threshold)
    metrics = compute_metrics(model, X_test, y_test, threshold)
    metrics["val_roc_auc"] = val_metrics["roc_auc"]
    metrics["val_auc_pr"] = val_metrics["auc_pr"]

    print("\n=== XGBoost Test-Set Metrics (held out — no selection on it) ===", file=sys.stderr)
    print(f"ROC-AUC : {metrics['roc_auc']:.4f}   (val {val_metrics['roc_auc']:.4f})", file=sys.stderr)
    print(f"AUC-PR  : {metrics['auc_pr']:.4f}   (val {val_metrics['auc_pr']:.4f})", file=sys.stderr)
    print(f"F1      : {metrics['f1']:.4f}", file=sys.stderr)
    print(f"Acc     : {metrics['accuracy']:.4f}", file=sys.stderr)
    print(metrics["classification_report"], file=sys.stderr)

    model_path = os.path.join(artifacts_dir, "xgb_model.joblib")
    feat_path = os.path.join(artifacts_dir, "xgb_features.json")
    metrics_path = os.path.join(artifacts_dir, "xgb_metrics.json")

    joblib.dump(model, model_path)
    with open(feat_path, "w") as f:
        json.dump(feat_names, f)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model   → {model_path}", file=sys.stderr)
    print(f"Saved features→ {feat_path}", file=sys.stderr)
    print(f"Saved metrics → {metrics_path}", file=sys.stderr)

    # ── Preprocessing artifacts for the standalone XGBoost inference service ──
    # Saved separately so that service needs no torch / GNN dependencies — only
    # joblib + numpy to featurize raw transactions (via RowFeaturizer) before
    # predict_proba. Carries the FULL fitted state (medians, card stats, product
    # risk, norm clip, edge/uid cols) so the service's featurization matches
    # graph_builder.preprocess exactly — not just the partial set the old pipeline
    # persisted.
    n_tabular = len(bundle["feature_cols"])
    cfg_merged = bundle["config"] if "config" in bundle else cfg
    preprocessor = {
        "feature_cols": bundle["feature_cols"],
        "label_maps": bundle.get("label_maps", {}),
        "m_cols": list(bundle.get("m_cols", [])),
        "freq_maps": bundle.get("freq_maps", {}),
        "count_maps": bundle.get("count_maps", {}),
        "norm_mu": bundle.get("norm_mu", np.zeros(n_tabular, dtype=np.float32)),
        "norm_sig": bundle.get("norm_sig", np.ones(n_tabular, dtype=np.float32)),
        "norm_clip": float(bundle.get("norm_clip", 0.0)),
        "train_medians": bundle.get("train_medians", {}),
        "card_stats": bundle.get("card_stats", {}),
        "prod_risk": bundle.get("prod_risk", {}),
        "prod_risk_default": float(bundle.get("prod_risk_default", 0.0)),
        "uid_cols": cfg_merged.get("graph", {}).get("uid_cols", []),
        "edge_cols": cfg_merged.get("graph", {}).get("edge_cols", []),
        "embed_dim": len(feat_names) - n_tabular,
    }
    preprocessor_path = os.path.join(artifacts_dir, "xgb_preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preproc → {preprocessor_path}", file=sys.stderr)

    # ── ONNX export of the trained XGBoost classifier ────────────────────────
    from pipeline.export_onnx import export_xgb, verify_xgb

    n_features = len(feat_names)
    onnx_path = export_xgb(model, n_features, artifacts_dir)
    verify_xgb(onnx_path, n_features)


if __name__ == "__main__":
    main()
