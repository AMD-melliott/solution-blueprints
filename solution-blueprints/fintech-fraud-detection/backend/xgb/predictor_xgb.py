# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
XGBoost inference predictor — no PyTorch, no GNN model, no rules engine.

Accepts raw transaction fields + a pre-computed GNN embedding (returned by
the GNN inference service) and runs XGBoost predict_proba.

Rules are NOT evaluated here — the fraud-orchestrator handles them.

Artifacts required in artifacts_dir:
  xgb_model.joblib        — trained XGBoost classifier (amd_xgboost)
  xgb_preprocessor.joblib — encoding / normalisation stats (saved by xgb_train.py)

AMD ROCm GPU is used for XGBoost prediction when available.
amd_xgboost must be installed from https://pypi.amd.com/rocm-7.1.1/simple
"""

import os

import joblib
import numpy as np
import yaml
from featurize import RowFeaturizer


class XGBPredictor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        artifacts_dir = self.cfg["data"]["artifacts_dir"]

        # ── Preprocessing artifacts (always needed) ───────────────────────────
        prep_path = os.path.join(artifacts_dir, "xgb_preprocessor.joblib")
        prep = joblib.load(prep_path)

        self.feature_cols: list[str] = prep["feature_cols"]
        self.embed_dim: int = prep.get("embed_dim", 0)
        self.threshold: float = self.cfg.get("eval", {}).get("threshold", 0.5)

        # Unified featurization (shared verbatim with the GNN service and training).
        # The preprocessor carries the FULL fitted state (maps, medians, card stats,
        # product risk, norm μ/σ, ±clip), so the tabular vector here is byte-identical
        # to the GNN service's node features and to training's graph.x.
        self.featurizer = RowFeaturizer.from_artifacts(prep, self.cfg)

        # ── ONNX session — try first; sklearn model only loaded as fallback ───
        # CPU-only ORT is discarded: native amd-xgboost+ROCm is faster than ONNX+CPU.
        self._ort_session = None
        onnx_path = os.path.join(artifacts_dir, "xgb.onnx")
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
                    print("[xgb-predictor] ONNX loaded  provider=ROCMExecutionProvider", flush=True)
                else:
                    print(
                        f"[xgb-predictor] ONNX ROCMExecutionProvider unavailable "
                        f"(got {active}) — using native amd-xgboost+ROCm instead",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[xgb-predictor] ONNX unavailable ({exc}), falling back to sklearn XGBoost", flush=True)

        # ── sklearn XGBoost model — only when ONNX is not available ──────────
        self.xgb_model = None
        self.xgb_device = "cpu"
        if self._ort_session is None:
            xgb_path = os.path.join(artifacts_dir, "xgb_model.joblib")
            self.xgb_model = joblib.load(xgb_path)
            xgb_device = "cuda" if self._amd_gpu_available() else "cpu"
            try:
                self.xgb_model.set_params(device=xgb_device)
                self.xgb_device = xgb_device
            except AttributeError:
                # Pickle was saved with an older XGBoost version — sklearn API
                # MRO is broken on deserialisation. GPU can still be set via the
                # underlying booster; if that also fails we fall back to CPU.
                try:
                    self.xgb_model._Booster.set_param("device", xgb_device)
                    self.xgb_device = xgb_device
                except Exception:
                    print(
                        "[xgb-predictor] WARNING: could not set device on old pickle "
                        "(version mismatch). Running on CPU. Regenerate xgb_model.joblib "
                        "with the current XGBoost version to fix.",
                        flush=True,
                    )

        print(
            f"[xgb-predictor] feats={len(self.feature_cols)}  "
            f"embed_dim={self.embed_dim}  xgb_device={self.xgb_device}",
            flush=True,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, transaction: dict, gnn_embedding: list[float]) -> dict:
        tab = self._featurize(transaction)
        emb = np.array(gnn_embedding, dtype=np.float32).reshape(1, -1)
        prob = float(self._predict_proba(np.hstack([tab, emb]))[0, 1])
        return {
            "fraud_probability": round(prob, 6),
            "decision": self._decide(prob),
            "model_version": "xgb",
            "model_type": "xgb_ensemble",
            "triggered_rules": [],
            "rule_action": "NONE",
        }

    def score_batch(
        self,
        transactions: list[dict],
        gnn_embeddings: list[list[float]],
    ) -> list[dict]:
        if len(transactions) != len(gnn_embeddings):
            raise ValueError(
                f"transactions ({len(transactions)}) and gnn_embeddings "
                f"({len(gnn_embeddings)}) must have the same length"
            )
        tab = np.vstack([self._featurize(tx) for tx in transactions])
        emb = np.array(gnn_embeddings, dtype=np.float32)
        probs = self._predict_proba(np.hstack([tab, emb]))[:, 1]
        return [
            {
                "fraud_probability": round(float(p), 6),
                "decision": self._decide(float(p)),
                "model_version": "xgb",
                "model_type": "xgb_ensemble",
                "triggered_rules": [],
                "rule_action": "NONE",
            }
            for p in probs
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [N, 2] array of [p_clean, p_fraud]. Routes to ONNX or native XGBoost."""
        if self._ort_session is not None:
            inp_name = self._ort_session.get_inputs()[0].name
            outputs = self._ort_session.run(None, {inp_name: X.astype(np.float32)})
            probs = outputs[1]
            if isinstance(probs[0], dict):
                # Older XGBoost ONNX format: list of {class: prob} maps
                return np.array([[p[0], p[1]] for p in probs], dtype=np.float32)
            return np.asarray(probs, dtype=np.float32)
        assert self.xgb_model is not None
        try:
            return self.xgb_model.predict_proba(X)
        except AttributeError:
            # sklearn API broken due to version mismatch — use raw booster
            import xgboost as xgb

            dm = xgb.DMatrix(X.astype(np.float32))
            p_fraud = self.xgb_model._Booster.predict(dm)  # shape [N], binary classifier
            return np.column_stack([1.0 - p_fraud, p_fraud]).astype(np.float32)

    def _featurize(self, tx: dict) -> np.ndarray:
        """Raw transaction → normalized tabular vector [1, n_tabular].

        Delegates to the shared RowFeaturizer so the tabular part is byte-identical
        to the GNN service's node features and to training's graph.x. The GNN
        embedding is concatenated by the caller (score / score_batch).
        """
        return self.featurizer.transform_row(tx).reshape(1, -1)

    def _decide(self, prob: float) -> str:
        if prob >= self.threshold:
            return "DECLINE"
        if prob >= self.threshold * 0.6:
            return "REVIEW"
        return "APPROVE"

    @staticmethod
    def _amd_gpu_available() -> bool:
        return os.path.exists("/dev/kfd")
