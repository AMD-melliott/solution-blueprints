#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
ONNX export utilities — called at the end of gnn_train.py and xgb_train.py.

GNN export  → artifacts/gnn.onnx
  Inputs  : node_features  [N, n_feats]   float32
            edge_index     [2, E]         int64
  Outputs : fraud_prob     [N]            float32
            embedding      [N, embed_dim] float32

  edge_index is a real input (dynamic E), so the SAME ONNX graph runs all three
  serving modes — isolated (E=0), rolling-graph (star), and batch (time-forward
  edges) — exactly like the PyTorch model. The inference service binds inputs by
  name; passing a [2, 0] edge_index reproduces the old isolated behaviour.
  Dynamic axes: N (num_nodes) and E (num_edges).

XGBoost export → artifacts/xgb.onnx
  Inputs  : features  [N, n_tabular + embed_dim]  float32
  Outputs : probabilities  [N, 2]                 float32
            label          [N]                     int64   (argmax)
"""

import os
import sys

import numpy as np

# ── GNN ───────────────────────────────────────────────────────────────────────


def export_gnn(bundle: dict, artifacts_dir: str) -> str:
    """
    Build FraudGNN from *bundle*, export isolated-mode ONNX to artifacts_dir.
    Returns the path to the written file.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pipeline.model import FraudGNN

    mcfg = bundle["model_cfg"]
    feature_cols = bundle["feature_cols"]
    n_feats = len(feature_cols)

    model = FraudGNN(
        in_channels=n_feats,
        hidden=mcfg["hidden_channels"],
        embed_dim=mcfg["embed_dim"],
        head_hidden=mcfg["head_hidden"],
        dropout=mcfg["dropout"],
    )
    model.load_state_dict({k: v.cpu() for k, v in bundle["model_state"].items()})
    model.eval()

    class _GNNWrapper(nn.Module):
        """Wraps FraudGNN so the ONNX graph emits fraud_prob + embedding and
        takes edge_index as a real (dynamic) input — supports every serving mode.
        """

        def __init__(self, inner: FraudGNN):
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
            logits, emb = self.inner(x, edge_index)
            prob = F.softmax(logits, dim=1)[:, 1]  # [N]
            return prob, emb

    wrapper = _GNNWrapper(model)
    wrapper.eval()  # ensure the top-level exported module is in eval mode too
    # (silences the "exporting in training mode" warning; inner
    # model is already eval, so dropout was off regardless)

    # Trace with a tiny non-empty graph (2 nodes, one directed edge 0→1) so the
    # message-passing / scatter ops are actually exercised and exported; the
    # dynamic axes then let E vary (including E=0 for isolated mode at serve time).
    dummy_x = torch.zeros((2, n_feats), dtype=torch.float32)
    dummy_ei = torch.tensor([[0], [1]], dtype=torch.long)
    out_path = os.path.join(artifacts_dir, "gnn.onnx")

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_ei),
        out_path,
        input_names=["node_features", "edge_index"],
        output_names=["fraud_prob", "embedding"],
        dynamic_axes={
            "node_features": {0: "num_nodes"},
            "edge_index": {1: "num_edges"},
            "fraud_prob": {0: "num_nodes"},
            "embedding": {0: "num_nodes"},
        },
        # opset 18: the SAGEConv scatter ops need >=18, and the current torch
        # exporter only has implementations for >=18 — requesting 17 forced a
        # fragile down-conversion via the onnx C API. 18 exports cleanly and is
        # well within onnxruntime's supported range.
        opset_version=18,
    )
    print(f"Saved GNN ONNX   → {out_path}", file=sys.stderr)
    return out_path


# ── XGBoost ───────────────────────────────────────────────────────────────────


def export_xgb(xgb_model, n_features: int, artifacts_dir: str) -> str:
    """
    Convert *xgb_model* (sklearn XGBClassifier) to ONNX.
    *n_features* is the total input width: n_tabular + embed_dim.
    Returns the path to the written file.
    """
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("features", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(
        xgb_model,
        initial_types=initial_type,
        target_opset=15,  # onnxmltools XGBoost converter supports up to 15
    )

    out_path = os.path.join(artifacts_dir, "xgb.onnx")
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Saved XGBoost ONNX → {out_path}", file=sys.stderr)
    return out_path


# ── Verification helpers ──────────────────────────────────────────────────────


def verify_gnn(onnx_path: str, n_feats: int) -> None:
    """Run a quick shape-correctness check with onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping GNN ONNX verification", file=sys.stderr)
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_names = {i.name for i in sess.get_inputs()}
    assert "edge_index" in in_names, (
        f"gnn.onnx must expose an 'edge_index' input for graph/batch modes; " f"got inputs {sorted(in_names)}"
    )

    # Isolated: no edges.
    dummy = np.random.RandomState(0).randn(3, n_feats).astype(np.float32)
    empty_ei = np.zeros((2, 0), dtype=np.int64)
    prob_iso, emb = sess.run(None, {"node_features": dummy, "edge_index": empty_ei})
    assert prob_iso.shape == (3,), f"Expected (3,), got {prob_iso.shape}"

    # With an edge 0→1, node 1 must aggregate node 0 → its embedding changes,
    # proving edge_index is wired into the exported graph (not ignored).
    ei = np.array([[0], [1]], dtype=np.int64)
    _, emb_edge = sess.run(None, {"node_features": dummy, "edge_index": ei})
    changed = not np.allclose(emb[1], emb_edge[1], atol=1e-6)
    print(
        f"GNN ONNX verified  : prob={prob_iso.shape}  emb={emb.shape}  " f"edge_index wired={changed}",
        file=sys.stderr,
    )
    if not changed:
        print(
            "  WARNING: edge_index did not change node 1's embedding — " "graph/batch modes may behave like isolated.",
            file=sys.stderr,
        )


def verify_xgb(onnx_path: str, n_features: int) -> None:
    """Run a quick shape-correctness check with onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping XGBoost ONNX verification", file=sys.stderr)
        return

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.zeros((2, n_features), dtype=np.float32)
    outputs = sess.run(None, {"features": dummy})
    print(
        f"XGBoost ONNX verified: outputs={[o.shape for o in outputs]}",
        file=sys.stderr,
    )
