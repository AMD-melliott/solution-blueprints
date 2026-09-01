#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Train a GraphSAGE fraud detection model on a pre-built transaction graph.

Reads the bundle produced by graph_builder.py.  Trains with cosine annealing
warm restarts; saves best weights by val ROC-AUC.  Augments the bundle with:
  model_state  – state_dict of the best checkpoint
  embeddings   – np.float32 [n_nodes, embed_dim] from the encoder
  history      – training metrics dict (losses, aucs, lr, …)
  model_cfg    – model hyperparameters snapshot

Usage:
  python -m pipeline.gnn_train                           # read artifacts/graph.pt
  python -m pipeline.gnn_train -i path/to/graph.pt
  python -m pipeline.graph_builder -o - | python -m pipeline.gnn_train -i -
"""

import argparse
import io
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from .model import FraudGNN


def _attach_masks(data, train_idx: np.ndarray, val_fraction: float, seed: int):
    # Clamp to [1, len-1] so neither split is empty: n_val=0 would make
    # train_idx[:-0] empty and train_idx[-0:] the full set (empty train, val=all),
    # and n_val=len would empty the train side. Mirrors xgb_train.py's max(1, …).
    n_val = min(max(1, int(round(len(train_idx) * val_fraction))), len(train_idx) - 1)
    tr = train_idx[:-n_val]
    va = train_idx[-n_val:]
    n = data.num_nodes
    dev = data.x.device
    train_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    val_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    train_mask[tr] = True
    val_mask[va] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    y_val = data.y[va].cpu().numpy()
    print(
        f"GNN masks: train={int(train_mask.sum()):,}  val={int(val_mask.sum()):,} (temporal tail, fraud={y_val.mean():.4%})",
        file=sys.stderr,
    )
    return data


def _class_weights(y: torch.Tensor, mask: torch.Tensor, device) -> torch.Tensor:
    y_sub = y[mask].cpu().numpy()
    n0 = max(1, int((y_sub == 0).sum()))
    n1 = max(1, int((y_sub == 1).sum()))
    total = n0 + n1
    return torch.tensor([total / (2 * n0), total / (2 * n1)], dtype=torch.float32, device=device)


def train_gnn(model: FraudGNN, data, cfg: dict) -> dict:
    tcfg = cfg["training"]
    _unused_seed = tcfg["seed"]
    lr = tcfg["lr"]
    min_lr = tcfg["min_lr"]
    weight_decay = tcfg["weight_decay"]
    grad_clip = tcfg["grad_clip"]
    max_wall = tcfg["max_wall_seconds"]
    epochs_fixed = tcfg["epochs"]
    warm_t0 = tcfg["warm_t0"]
    warm_tmult = tcfg["warm_t_mult"]
    log_every = tcfg["log_every_sec"]

    weights = _class_weights(data.y, data.train_mask, data.x.device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=warm_t0, T_mult=warm_tmult, eta_min=min_lr
    )

    use_wall = max_wall and max_wall > 0
    if use_wall:
        print(f"Training: wall={max_wall}s  epochs={epochs_fixed:,}  T_0={warm_t0}", file=sys.stderr)
    else:
        print(f"Training: {epochs_fixed} epochs", file=sys.stderr)

    history: dict[str, Any] = dict(train_loss=[], val_loss=[], val_acc=[], val_f1=[], val_auc=[], val_aupr=[], lr=[])
    best_auc, best_epoch, best_state = -1.0, 0, None

    t0 = time.perf_counter()
    last_log = t0

    for epoch in range(1, 10_000_000):
        elapsed = time.perf_counter() - t0
        if epoch > epochs_fixed:
            break
        if use_wall and elapsed >= max_wall:
            break

        model.train()
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        history["train_loss"].append(float(loss.detach()))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        model.eval()
        with torch.no_grad():
            logits, _ = model(data.x, data.edge_index)
            vloss = criterion(logits[data.val_mask], data.y[data.val_mask])
        history["val_loss"].append(float(vloss))

        lv = logits[data.val_mask].cpu()
        yv = data.y[data.val_mask].cpu().numpy()
        proba = F.softmax(lv, dim=1)[:, 1].numpy()
        pred = lv.argmax(dim=1).numpy()

        history["val_acc"].append(float(accuracy_score(yv, pred)))
        history["val_f1"].append(float(f1_score(yv, pred, pos_label=1, zero_division=0)))

        try:
            vauc = float(roc_auc_score(yv, proba))
        except ValueError:
            vauc = float("nan")
        history["val_auc"].append(vauc)

        try:
            history["val_aupr"].append(float(average_precision_score(yv, proba)))
        except ValueError:
            history["val_aupr"].append(float("nan"))

        if np.isfinite(vauc) and vauc > best_auc:
            best_auc = vauc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        now = time.perf_counter()
        if epoch == 1 or (now - last_log) >= log_every:
            eps = epoch / max(now - t0, 1e-6)
            wall_info = f"{now-t0:6.0f}s/{max_wall}s  {eps:.0f}ep/s" if use_wall else f"ep={epoch}"
            print(
                f"[{wall_info}]  lr={history['lr'][-1]:.2e}"
                f"  train={history['train_loss'][-1]:.4f}"
                f"  val={history['val_loss'][-1]:.4f}"
                f"  auc={vauc:.4f}  aupr={history['val_aupr'][-1]:.4f}"
                f"  f1={history['val_f1'][-1]:.4f}"
                f"  best={best_auc:.4f}@{best_epoch}",
                file=sys.stderr,
            )
            last_log = now

    total_s = time.perf_counter() - t0
    print(
        f"Done    : {epoch:,} epochs  {total_s:.1f}s  best val_auc={best_auc:.4f}@ep{best_epoch}",
        file=sys.stderr,
    )

    if best_state is not None:
        dev = next(model.parameters()).device
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    history.update(best_epoch=best_epoch, best_val_auc=best_auc, total_wall_sec=total_s, epochs_run=epoch)
    return history


def extract_embeddings(model: FraudGNN, data, edge_index=None) -> np.ndarray:
    """Export node embeddings.

    Pass edge_index=inductive_edge_index so TEST nodes aggregate over their
    (train) neighbours — the canonical inductive-inference regime — instead of
    being isolated, which is what data.edge_index (the stored train-train set)
    gives them (test nodes have zero edges there). This keeps the embedding
    regime consistent between what XGBoost trains on (train rows) and scores
    (test rows), and with the serving path.
    """
    model.eval()
    ei = edge_index if edge_index is not None else data.edge_index
    with torch.no_grad():
        _, emb = model(data.x, ei)
    return emb.cpu().numpy()


def read_bundle(src: str) -> dict:
    if src == "-":
        return torch.load(io.BytesIO(sys.stdin.buffer.read()), weights_only=False)
    return torch.load(src, weights_only=False)


def write_bundle(bundle: dict, out: str) -> None:
    if out == "-":
        buf = io.BytesIO()
        torch.save(bundle, buf)
        sys.stdout.buffer.write(buf.getvalue())
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        torch.save(bundle, out)
        print(f"Saved   → {out}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN on transaction graph")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--input", "-i", default=None, help='bundle path or "-" for stdin')
    parser.add_argument("--output", "-o", default=None, help='bundle path or "-" for stdout')
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}", file=sys.stderr)

    inp = args.input or os.path.join(cfg["data"]["artifacts_dir"], "graph.pt")
    print(f"Reading : {inp if inp != '-' else 'stdin'}", file=sys.stderr)
    bundle = read_bundle(inp)

    graph_data = bundle["graph"].to(device)
    train_idx = bundle["train_idx"]
    cfg_merged = bundle["config"] if "config" in bundle else cfg

    mcfg = cfg_merged["model"]
    _attach_masks(graph_data, train_idx, cfg_merged["training"]["val_fraction"], seed)

    model = FraudGNN(
        in_channels=graph_data.num_node_features,
        hidden=mcfg["hidden_channels"],
        embed_dim=mcfg["embed_dim"],
        head_hidden=mcfg["head_hidden"],
        dropout=mcfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model   : {n_params:,} parameters", file=sys.stderr)

    history = train_gnn(model, graph_data, cfg_merged)

    ind_ei = bundle.get("inductive_edge_index")
    ind_ei = ind_ei.to(device) if ind_ei is not None else None
    embeddings = extract_embeddings(model, graph_data, edge_index=ind_ei)
    print(f"Embs    : {embeddings.shape}", file=sys.stderr)

    bundle.update(
        model_state={k: v.cpu() for k, v in model.state_dict().items()},
        embeddings=embeddings,
        history=history,
        model_cfg=mcfg,
    )

    out = args.output or os.path.join(cfg["data"]["artifacts_dir"], "model.pt")
    write_bundle(bundle, out)

    # Export the trained encoder to ONNX (skipped on the stdout pipe). The ONNX
    # graph takes node_features AND edge_index as inputs, so the inference service
    # can run all three modes (isolated / rolling-graph / batch) through ONNX —
    # not just isolated. See pipeline/export_onnx.py.
    if out != "-":
        from pipeline.export_onnx import export_gnn, verify_gnn

        onnx_path = export_gnn(bundle, cfg["data"]["artifacts_dir"])
        verify_gnn(onnx_path, graph_data.num_node_features)


if __name__ == "__main__":
    main()
