#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""
Unified inference-time featurization.

This module is the SINGLE source of truth for turning one raw transaction dict
(or a list of them) into the normalized feature vector the models expect. It is
the inference-side mirror of ``graph_builder.preprocess`` (the train-side, batch
/ DataFrame implementation that *fits* these statistics). Both must produce
identical vectors for the same input — any drift shows up silently as
out-of-distribution inputs to the GNN/XGBoost, not as an error.

Design notes
------------
* numpy-only, no torch / pandas — so the standalone XGBoost service (which has no
  torch) and the GNN service share exactly the same code path.
* All fitted statistics travel in the artifact bundle produced by
  ``graph_builder``/``gnn_train`` (``model.pt``) and the XGBoost preprocessor
  (``xgb_preprocessor.joblib``); ``RowFeaturizer.from_artifacts`` accepts either.
* The engineered columns (tx_hour / tx_day_week / tx_day, D1_norm, amt_zscore,
  product_risk_score, *_freq / *_log_freq) are computed here exactly as in
  ``graph_builder.preprocess`` — keep the two in sync.

This file is intentionally duplicated verbatim across the (independently
deployed, self-contained) services. Keep all copies in sync:
  * gxfd_orig/pipeline/featurize.py                       (training repo)
  * fintech-fraud-detection/backend/gnn/featurize.py      (GNN service)
  * fintech-fraud-detection/backend/xgb/featurize.py      (XGBoost service)
When you change one, change the others.
"""

from __future__ import annotations

import numpy as np


class RowFeaturizer:
    """Featurize single transactions (or batches) the same way training did."""

    def __init__(
        self,
        feature_cols: list[str],
        label_maps: dict[str, dict] | None = None,
        m_cols=None,
        freq_maps: dict[str, dict] | None = None,
        count_maps: dict[str, dict] | None = None,
        norm_mu: np.ndarray | None = None,
        norm_sig: np.ndarray | None = None,
        norm_clip: float = 0.0,
        train_medians: dict | None = None,
        card_stats: dict | None = None,
        prod_risk: dict | None = None,
        prod_risk_default: float = 0.0,
        uid_cols: list[str] | None = None,
        edge_cols: list[str] | None = None,
    ):
        self.feature_cols = list(feature_cols)
        n = len(self.feature_cols)
        self.label_maps = label_maps or {}
        self.m_cols = set(m_cols or [])
        self.freq_maps = freq_maps or {}
        self.count_maps = count_maps or {}
        self.norm_mu = np.asarray(norm_mu, dtype=np.float32) if norm_mu is not None else np.zeros(n, dtype=np.float32)
        self.norm_sig = np.asarray(norm_sig, dtype=np.float32) if norm_sig is not None else np.ones(n, dtype=np.float32)
        self.norm_clip = float(norm_clip or 0.0)
        self.train_medians = train_medians or {}
        # card_stats keys are int(card1); values {"mean":..,"std":..}
        self.card_stats = card_stats or {}
        # prod_risk keys are int(label-encoded ProductCD); values float
        self.prod_risk = prod_risk or {}
        self.prod_risk_default = float(prod_risk_default or 0.0)
        self.uid_cols = list(uid_cols or [])
        self.edge_cols = list(edge_cols or [])

    # ── Constructors ────────────────────────────────────────────────────────

    @classmethod
    def from_artifacts(cls, art: dict, cfg: dict | None = None) -> "RowFeaturizer":
        """Build from a loaded ``model.pt`` bundle or ``xgb_preprocessor.joblib``.

        Edge-key names (``edge_cols`` / ``uid_cols``) are resolved with a
        train-wins precedence so serving never silently diverges from how the
        graph was built:
          1. the artifact's own ``edge_cols`` / ``uid_cols`` (xgb_preprocessor),
          2. the TRAINING config embedded in the bundle (``art["config"].graph``),
          3. the passed serving ``cfg`` as a last resort.
        """
        train_gcfg = (art.get("config") or {}).get("graph", {})
        serve_gcfg = (cfg or {}).get("graph", {})
        n = len(art["feature_cols"])
        return cls(
            feature_cols=art["feature_cols"],
            label_maps=art.get("label_maps", {}),
            m_cols=art.get("m_cols", []),
            freq_maps=art.get("freq_maps", {}),
            count_maps=art.get("count_maps", {}),
            norm_mu=art.get("norm_mu", np.zeros(n, dtype=np.float32)),
            norm_sig=art.get("norm_sig", np.ones(n, dtype=np.float32)),
            norm_clip=float(art.get("norm_clip", 0.0)),
            train_medians=art.get("train_medians", {}),
            card_stats=art.get("card_stats", {}),
            prod_risk=art.get("prod_risk", {}),
            prod_risk_default=float(art.get("prod_risk_default", 0.0)),
            uid_cols=art.get("uid_cols") or train_gcfg.get("uid_cols") or serve_gcfg.get("uid_cols", []),
            edge_cols=art.get("edge_cols") or train_gcfg.get("edge_cols") or serve_gcfg.get("edge_cols", []),
        )

    # ── Featurization ─────────────────────────────────────────────────────────

    def _num(self, expanded: dict, col: str) -> float:
        """Numeric value of *col*, falling back to the train median (then 0.0)."""
        v = expanded.get(col)
        if v is None:
            return float(self.train_medians.get(col, 0.0))
        try:
            f = float(v)
        except (TypeError, ValueError):
            return float(self.train_medians.get(col, 0.0))
        if not np.isfinite(f):
            return float(self.train_medians.get(col, 0.0))
        return f

    def transform_row(self, tx: dict) -> np.ndarray:
        """Raw transaction dict → normalized feature vector ``[n_feats]`` float32.

        Mirrors ``graph_builder.preprocess``: NaN numerics → train median,
        categoricals → label code (unseen → ``Unknown``), binary M's → 1/0/-1,
        frequency encoding, then the engineered columns, z-score, ±clip.
        """
        expanded: dict = {}
        for col, raw in tx.items():
            if raw is not None:
                expanded[col] = raw

        # Binary M columns: "T"→1.0, "F"→0.0, missing/other→-1.0. Categorical M's
        # (e.g. M4) are NOT in m_cols; they ride the label-encoding pass below.
        for col in self.m_cols:
            raw = tx.get(col)
            expanded[col] = 1.0 if raw == "T" else (0.0 if raw == "F" else -1.0)

        # Categoricals → label codes (unseen/missing → Unknown).
        for col, label_map in self.label_maps.items():
            raw = expanded.get(col)
            key = "Unknown" if raw is None else str(raw)
            expanded[col] = label_map.get(key, label_map.get("Unknown", 0))

        # Frequency / log-frequency encoding (on the post-label-encode code).
        for col, freq_map in self.freq_maps.items():
            code = expanded.get(col)
            count_map = self.count_maps.get(col, {})
            expanded[f"{col}_freq"] = float(freq_map.get(code, 0.0)) if code is not None else 0.0
            expanded[f"{col}_log_freq"] = float(np.log1p(count_map.get(code, 0))) if code is not None else 0.0

        # ── Engineered columns (must match graph_builder.preprocess exactly) ──
        dt_raw = expanded.get("TransactionDT")
        if dt_raw is not None:
            try:
                dt_i = int(float(dt_raw))
                expanded["tx_hour"] = (dt_i // 3600) % 24
                expanded["tx_day_week"] = (dt_i // 86400) % 7
                expanded["tx_day"] = dt_i // 86400
            except (TypeError, ValueError):
                expanded["tx_hour"] = 0
                expanded["tx_day_week"] = 0
                expanded["tx_day"] = 0

        if "D1" in self.feature_cols or "D1_norm" in self.feature_cols:
            d1_val = self._num(expanded, "D1")
            expanded["D1"] = d1_val
            tx_day = expanded.get("tx_day")
            if tx_day is not None:
                expanded["D1_norm"] = d1_val - tx_day

        if self.card_stats:
            amt = self._num(expanded, "TransactionAmt")
            card1_raw = tx.get("card1")
            try:
                card1_key = int(card1_raw) if card1_raw is not None else None
            except (TypeError, ValueError):
                card1_key = None
            stats = self.card_stats.get(card1_key) if card1_key is not None else None
            if stats:
                mu = float(stats.get("mean", 0.0))
                sd = float(stats.get("std", 0.0))
                # zero/undefined per-card variance → 0.0 (not (amt-mu)/1e-6).
                expanded["amt_zscore"] = (amt - mu) / sd if sd > 0 else 0.0
            else:
                expanded["amt_zscore"] = 0.0

        if self.prod_risk or "product_risk_score" in self.feature_cols:
            prod_code = expanded.get("ProductCD")
            try:
                pk = int(prod_code) if prod_code is not None else None
            except (TypeError, ValueError):
                pk = None
            expanded["product_risk_score"] = (
                float(self.prod_risk.get(pk, self.prod_risk_default)) if pk is not None else self.prod_risk_default
            )

        vec = np.array(
            [self._num(expanded, col) for col in self.feature_cols],
            dtype=np.float32,
        )
        vec = (vec - self.norm_mu) / self.norm_sig
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if self.norm_clip and self.norm_clip > 0:
            vec = np.clip(vec, -self.norm_clip, self.norm_clip)
        return vec.astype(np.float32)

    def transform_batch(self, txs: list[dict]) -> np.ndarray:
        """Featurize a list of transactions → ``[N, n_feats]`` float32."""
        if not txs:
            return np.zeros((0, len(self.feature_cols)), dtype=np.float32)
        return np.vstack([self.transform_row(tx) for tx in txs]).astype(np.float32)

    # ── Edge keys (graph / rolling / batch modes) ───────────────────────────

    def edge_key_vals(self, tx: dict) -> dict:
        """Edge-key values for one transaction, mirroring graph_builder's keys.

        Raw keys (e.g. card1) come straight from tx; the composite 'uid' is built
        from ``uid_cols`` (card1_addr1_D1) and is None if any component is missing
        (→ no edge, matching the train-time NaN drop). Each component is
        canonicalized via ``str(float(p))`` (13926 → "13926.0") so it is
        dtype-independent and byte-identical to graph_builder's uid.
        """
        vals: dict = {}
        for k in self.edge_cols:
            if k == "uid" and self.uid_cols:
                parts = [tx.get(c) for c in self.uid_cols]
                if all(p is not None for p in parts):
                    try:
                        vals[k] = "_".join(str(float(p)) for p in parts)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        vals[k] = None
                else:
                    vals[k] = None
            else:
                vals[k] = tx.get(k)
        return vals
