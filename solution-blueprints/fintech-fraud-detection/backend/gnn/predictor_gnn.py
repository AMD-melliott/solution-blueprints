# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
GNN-only predictor for the standalone GNN inference service.

Scoring modes (all share the SAME trained weights and featurization):
  isolated (default)
    Single node, no edges. SAGEConv applies a linear transform on the node's own
    features only. Fast and stateless.

  graph
    Rolling in-memory mini-graph: recent transactions sharing an edge-key value
    (card1, or the composite uid = card1_addr1_D1) become neighbours. Edges are
    DIRECTED earlier→later (neighbour → query), so the query aggregates only its
    past — matching the time-forward training graph and the inductive embedding
    regime XGBoost was trained on.

  batch (graph mode over a list)
    Within-batch mini-graph: transactions sharing an edge-key value are wired
    earlier→later by TransactionDT — the batch analogue of the training graph.

Featurization is delegated to the shared RowFeaturizer (featurize.py), the single
inference-time preprocessing routine — identical in the XGBoost service and the
twin of graph_builder.preprocess in training. Rules are NOT evaluated here — the
fraud-orchestrator handles them.
"""

import os
import threading
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from featurize import RowFeaturizer
from model import FraudGNN


class RollingGraph:
    """Thread-safe buffer of recent transactions for mini-graph inference.

    Stores (feature_vector, edge_key_vals) per transaction. Neighbours are buffer
    entries sharing at least one non-null edge-key value with the query.
    """

    def __init__(self, max_size: int, edge_keys: list[str]):
        self.edge_keys = edge_keys
        self._lock = threading.Lock()
        self._buf: deque[tuple[torch.Tensor, dict]] = deque(maxlen=max_size)

    def neighbors(self, key_vals: dict) -> list[torch.Tensor]:
        with self._lock:
            return [
                x
                for x, kv in self._buf
                if any(key_vals.get(k) is not None and key_vals.get(k) == kv.get(k) for k in self.edge_keys)
            ]

    def push(self, x_cpu: torch.Tensor, key_vals: dict) -> None:
        with self._lock:
            self._buf.append((x_cpu, key_vals))

    def clear(self) -> None:
        """Drop all buffered transactions (used when the stream is stopped)."""
        with self._lock:
            self._buf.clear()


class GNNPredictor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        artifacts_dir = self.cfg["data"]["artifacts_dir"]

        # ── Load bundle on CPU — needed for featurization metadata regardless ─
        bundle_path = os.path.join(artifacts_dir, "model.pt")
        bundle = torch.load(bundle_path, weights_only=False, map_location="cpu")

        self.feature_cols: list[str] = bundle["feature_cols"]
        n_feats = len(self.feature_cols)

        # Unified featurization (shared verbatim with the XGB service and training).
        # All fitted stats travel in the bundle: label/freq/count maps, medians,
        # card stats, product risk, norm μ/σ, ±clip, and the edge/uid key names.
        self.featurizer = RowFeaturizer.from_artifacts(bundle, self.cfg)

        self.threshold: float = self.cfg.get("eval", {}).get("threshold", 0.5)
        self.max_group: int = int(self.cfg.get("graph", {}).get("max_group_size", 300))
        self.model_version: str = str(bundle.get("history", {}).get("best_epoch", "unknown"))

        # ── ONNX session — try first; PyTorch model only loaded as fallback ───
        # CPU-only ORT is discarded: PyTorch+ROCm is faster than ONNX+CPU.
        self._ort_session = None
        self._ort_has_edges = False
        onnx_path = os.path.join(artifacts_dir, "gnn.onnx")
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort

                _sess = ort.InferenceSession(
                    onnx_path,
                    providers=["ROCMExecutionProvider", "CPUExecutionProvider"],
                )
                active = _sess.get_providers()[0]
                if active == "ROCMExecutionProvider":
                    self._ort_session = _sess
                    self._ort_has_edges = any(i.name == "edge_index" for i in _sess.get_inputs())
                    print(
                        f"[gnn-predictor] ONNX loaded  provider=ROCMExecutionProvider"
                        f"  edge_index_input={self._ort_has_edges}",
                        flush=True,
                    )
                    if not self._ort_has_edges:
                        print(
                            "[gnn-predictor] WARNING: gnn.onnx has no edge_index input — "
                            "graph/batch modes will degrade to isolated through ONNX. "
                            "Re-export with the current pipeline (export_onnx.export_gnn).",
                            flush=True,
                        )
                else:
                    print(
                        f"[gnn-predictor] ONNX ROCMExecutionProvider unavailable "
                        f"(got {active}) — using PyTorch+ROCm instead",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[gnn-predictor] ONNX unavailable ({exc}), falling back to PyTorch", flush=True)

        # ── PyTorch model — only when ONNX is not available ──────────────────
        if self._ort_session is not None:
            # Featurize tensors go straight to numpy; CPU is sufficient
            self.device = torch.device("cpu")
            self.model = None
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device.type != "cuda":
                import warnings

                warnings.warn(
                    "ROCm/CUDA not available — running on CPU. "
                    "Pass --device /dev/kfd --device /dev/dri to docker run."
                )
            else:
                print(
                    f"[gnn-predictor] device={self.device}  " f"hip={getattr(torch.version, 'hip', None)}",
                    flush=True,
                )
            mcfg = bundle.get("model_cfg", self.cfg["model"])
            self.model = FraudGNN(
                in_channels=n_feats,
                hidden=mcfg["hidden_channels"],
                embed_dim=mcfg["embed_dim"],
                head_hidden=mcfg["head_hidden"],
                dropout=mcfg["dropout"],
            ).to(self.device)
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in bundle["model_state"].items()},
                strict=False,  # running_mean/var may be absent in older checkpoints
            )
            self.model.eval()

        # Rolling buffer keyed on the SAME edge keys as training (card1, uid).
        graph_cfg = self.cfg.get("graph", {})
        self.rolling_graph = RollingGraph(
            max_size=int(graph_cfg.get("rolling_size", 5000)),
            edge_keys=list(self.featurizer.edge_cols),
        )

        print(
            f"[gnn-predictor] feats={n_feats}  edge_keys={self.featurizer.edge_cols}  " f"version={self.model_version}",
            flush=True,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, transaction: dict, mode: str = "isolated") -> dict:
        if mode == "graph":
            return self._score_graph(transaction)
        return self._score_isolated(transaction)

    def reset_graph(self) -> None:
        """Clear the rolling neighbour buffer so the next graph run starts empty."""
        self.rolling_graph.clear()

    def score_batch(self, transactions: list[dict], mode: str = "isolated") -> list[dict]:
        xs = self._featurize_batch(transactions)

        if mode == "graph":
            edge_index = self._build_batch_edges(transactions)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        probs, embeddings = self._forward(xs, edge_index)
        return [
            {
                "fraud_probability": round(float(prob), 6),
                "decision": self._decide(float(prob)),
                "model_version": self.model_version,
                "model_type": mode,
                "embedding": emb,
                "triggered_rules": [],
                "rule_action": "NONE",
            }
            for prob, emb in zip(probs.tolist(), embeddings)
        ]

    # ── Forward (ONNX or PyTorch) ───────────────────────────────────────────

    def _forward(self, xs: torch.Tensor, edge_index: torch.Tensor):
        """Run the model. Returns (fraud_prob[N] np.ndarray, embeddings list[list])."""
        if self._ort_session is not None:
            ei_np = edge_index.cpu().numpy().astype(np.int64)
            probs_np, emb_np = self._run_ort(xs.cpu().numpy().astype(np.float32), ei_np)
            return np.atleast_1d(probs_np), np.atleast_2d(emb_np).tolist()
        assert self.model is not None
        with torch.no_grad():
            logits, embs = self.model(xs, edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs, embs.cpu().tolist()

    def _run_ort(self, x_np: np.ndarray, ei_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass through the ONNX session. Binds inputs BY NAME.

        The exported graph takes node_features and edge_index; if (for an old
        export) edge_index is absent, we pass only node_features and the model
        scores isolated. ONNX already applies softmax → fraud_prob [N].
        """
        assert self._ort_session is not None
        feed = {"node_features": x_np}
        if self._ort_has_edges:
            feed["edge_index"] = ei_np
        out_names = [o.name for o in self._ort_session.get_outputs()]
        results = self._ort_session.run(out_names, feed)
        by_name = dict(zip(out_names, results))
        return by_name["fraud_prob"], by_name["embedding"]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _featurize(self, tx: dict) -> torch.Tensor:
        """Raw transaction → normalized feature tensor [1, n_feats] (shared routine)."""
        vec = self.featurizer.transform_row(tx)
        return torch.from_numpy(vec).unsqueeze(0).to(self.device)

    def _featurize_batch(self, txs: list[dict]) -> torch.Tensor:
        mat = self.featurizer.transform_batch(txs)
        return torch.from_numpy(mat).to(self.device)

    def _score_isolated(self, tx: dict) -> dict:
        x = self._featurize(tx)
        empty_ei = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        probs, embeddings = self._forward(x, empty_ei)
        return {
            "fraud_probability": round(float(probs[0]), 6),
            "decision": self._decide(float(probs[0])),
            "model_version": self.model_version,
            "model_type": "isolated",
            "embedding": embeddings[0],
            "triggered_rules": [],
            "rule_action": "NONE",
        }

    def _score_graph(self, tx: dict) -> dict:
        x = self._featurize(tx)
        key_vals = self.featurizer.edge_key_vals(tx)
        neighbors = self.rolling_graph.neighbors(key_vals)

        if neighbors:
            x_all = torch.cat([x] + [n.to(self.device) for n in neighbors], dim=0)
            n = len(neighbors)
            # Directed earlier→later: every (earlier) neighbour → query node 0.
            # Buffer entries are all earlier transactions, so the query aggregates
            # only its past — time-forward, matching the training graph.
            src = list(range(1, n + 1))
            dst = [0] * n
            edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        else:
            x_all = x
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        probs, embeddings = self._forward(x_all, edge_index)
        # Push AFTER scoring so the query never sees itself as a neighbour.
        self.rolling_graph.push(x.cpu(), key_vals)
        return {
            "fraud_probability": round(float(probs[0]), 6),
            "decision": self._decide(float(probs[0])),
            "model_version": self.model_version,
            "model_type": "graph",
            "embedding": embeddings[0],
            "triggered_rules": [],
            "rule_action": "NONE",
        }

    def _build_batch_edges(self, transactions: list[dict]) -> torch.Tensor:
        """Within-batch mini-graph, mirroring graph_builder: group by each edge key
        (incl. the composite uid), order each group earlier→later by TransactionDT,
        and wire directed earlier→later pairs. O(sum group²) but groups are capped
        at max_group_size.
        """
        from collections import defaultdict

        kvs = [self.featurizer.edge_key_vals(tx) for tx in transactions]
        src, dst = [], []

        for key in self.rolling_graph.edge_keys:
            groups: dict = defaultdict(list)
            for i, kv in enumerate(kvs):
                val = kv.get(key)
                if val is not None:
                    groups[val].append(i)

            for idxs in groups.values():
                if len(idxs) < 2:
                    continue
                # earlier→later by tx time (fallback: batch order); keep the
                # MOST-RECENT max_group, consistent with training's truncation.
                idxs = sorted(idxs, key=lambda i: transactions[i].get("TransactionDT", i))
                idxs = idxs[-self.max_group :]
                for p in range(len(idxs)):
                    for q in range(p + 1, len(idxs)):
                        src.append(idxs[p])  # earlier
                        dst.append(idxs[q])  # later

        if src:
            return torch.tensor([src, dst], dtype=torch.long, device=self.device)
        return torch.zeros((2, 0), dtype=torch.long, device=self.device)

    def _decide(self, prob: float) -> str:
        if prob >= self.threshold:
            return "DECLINE"
        if prob >= self.threshold * 0.6:
            return "REVIEW"
        return "APPROVE"
