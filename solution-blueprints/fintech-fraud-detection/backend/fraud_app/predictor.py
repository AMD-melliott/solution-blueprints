# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Combined GNN + XGBoost predictor — orchestration only.

Featurization is handled by the model services (GNN / XGBoost containers).
This service receives raw transactions, calls the rules engine first, then
dispatches to the appropriate remote services.

Three scoring modes:
  xgb_ensemble (default)
    Raw tx → GNN service (graph mode: featurize + neighbour-aggregated embed) →
    concat(tabular, embedding) → XGBoost service → fraud probability. Graph mode
    matches the inductive embedding regime XGBoost was trained on. Best accuracy.

  gnn_only
    Raw tx → GNN service (isolated) → fraud probability. No XGBoost.

  graph
    Raw tx → GNN service (graph mode, rolling neighbor context managed by the
    GNN service itself) → fraud probability.
"""

import os

import httpx
import yaml
from fraud_app.rules_engine import RulesEngine


class CombinedPredictor:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # ── Remote model clients ──────────────────────────────────────────────
        timeout = float(self.cfg.get("inference", {}).get("http_timeout_sec", 10.0))
        self._gnn_client = _ModelClient("GNN", timeout)
        self._xgb_client = _ModelClient("XGB", timeout)

        # ── Threshold + decision logic ────────────────────────────────────────
        self.threshold: float = self.cfg.get("eval", {}).get("threshold", 0.5)

        # ── Deterministic rules engine (runs before ML) ───────────────────────
        self._rules_engine = RulesEngine()

        print(
            f"[predictor] gnn={os.environ.get('GNN_MODEL_URL')}  " f"xgb={os.environ.get('XGB_MODEL_URL')}",
            flush=True,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, transaction: dict, mode: str = "xgb_ensemble") -> dict:
        if mode == "xgb_ensemble":
            return self._score_xgb(transaction)
        elif mode == "graph":
            return self._score_gnn_graph(transaction)
        else:
            return self._score_gnn(transaction)

    def reset(self) -> None:
        """Clear all stateful inference context so a restarted stream starts clean.

        Two pieces of state survive across requests and must both be flushed:
          * the GNN rolling neighbour graph (in the remote GNN service), and
          * the rules engine's velocity/duplicate history (local).
        XGBoost is stateless and needs no reset.
        """
        self._gnn_client.reset()
        self._rules_engine.reset()

    def dependencies_ready(self) -> dict[str, bool]:
        """Probe the remote GNN and XGBoost services. Used by /health/deep."""
        return {"gnn": self._gnn_client.ready(), "xgb": self._xgb_client.ready()}

    def score_batch(self, transactions: list[dict], mode: str = "xgb_ensemble") -> list[dict]:
        """
        Score a list of transactions in one forward pass.

        The GNN service handles graph construction internally:
          - xgb_ensemble / gnn_only: GNN runs in isolated mode (no edges).
          - graph: GNN builds within-batch edges from shared card/email fields.

        Rules are applied per-transaction in iteration order so the velocity
        tracker accumulates state correctly across the batch.
        """
        n = len(transactions)

        if mode == "gnn_only":
            probs, _ = self._call_gnn(transactions, "isolated")

        elif mode == "graph":
            probs, _ = self._call_gnn(transactions, "graph")

        else:  # xgb_ensemble
            # Graph mode → neighbour-aggregated embeddings, matching the inductive
            # regime XGBoost was trained on (training exports embeddings over
            # inductive_edge_index, not isolated). The GNN service builds the
            # within-batch mini-graph internally.
            _, embeddings = self._call_gnn(transactions, "graph")
            xgb_outputs = self._xgb_client.infer(
                [
                    {"name": "transactions", "shape": [n], "datatype": "BYTES", "data": transactions},
                    {"name": "embeddings", "shape": [n, len(embeddings[0])], "datatype": "FP32", "data": embeddings},
                ]
            )
            probs = [row[1] for row in xgb_outputs[0]["data"]]

        results = []
        for tx, p in zip(transactions, probs):
            rules_matched, blocked, worst = self._run_rules(tx)
            if blocked:
                results.append(
                    {
                        "fraud_probability": 1.0,
                        "decision": "DECLINE",
                        "model_version": "rules",
                        "model_type": mode,
                        "rules_matched": rules_matched,
                    }
                )
            else:
                prob = float(p)
                results.append(
                    {
                        "fraud_probability": round(prob, 6),
                        "decision": self._combine(self._decide(prob), worst),
                        "model_version": "remote",
                        "model_type": mode,
                        "rules_matched": rules_matched,
                    }
                )
        return results

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call_gnn(self, transactions: list[dict], mode: str) -> tuple[list[float], list[list[float]]]:
        """
        Call the GNN service. Returns (fraud_probs, embeddings).

        The GNN service applies softmax and returns fraud_probability directly.
        Graph state (rolling neighbor context) is managed inside the GNN service.
        """
        n = len(transactions)
        outputs = self._gnn_client.infer(
            [
                {"name": "transactions", "shape": [n], "datatype": "BYTES", "data": transactions},
                {"name": "mode", "shape": [1], "datatype": "BYTES", "data": [mode]},
            ]
        )
        by_name = {o["name"]: o["data"] for o in outputs}
        return by_name["fraud_probability"], by_name["embedding"]

    def _run_rules(self, tx: dict) -> tuple[list[dict], bool, str | None]:
        """
        Evaluate deterministic rules against *tx*, update velocity state, and
        return (serialized_matches, hard_blocked, worst_action).

        Always calls record() so velocity state stays consistent regardless of
        which scoring path is taken.
        """
        matches = self._rules_engine.evaluate(tx)
        self._rules_engine.record(tx)
        serialized = [
            {
                "rule_id": m.rule_id,
                "description": m.description,
                "risk_level": m.risk_level,
                "action": m.action,
            }
            for m in matches
        ]
        return serialized, RulesEngine.hard_blocked(matches), RulesEngine.worst_action(matches)

    def _combine(self, ml_decision: str, worst_rule_action: str | None) -> str:
        """Merge ML decision with rules: REVIEW upgrades APPROVE; MONITOR is advisory only."""
        if worst_rule_action == "REVIEW" and ml_decision == "APPROVE":
            return "REVIEW"
        return ml_decision

    def _score_xgb(self, tx: dict) -> dict:
        rules_matched, blocked, worst = self._run_rules(tx)
        if blocked:
            return {
                "fraud_probability": 1.0,
                "decision": "DECLINE",
                "model_version": "rules",
                "model_type": "xgb_ensemble",
                "rules_matched": rules_matched,
            }
        # Graph mode → neighbour-aggregated embedding (rolling buffer in the GNN
        # service), matching the inductive regime XGBoost was trained on.
        _, emb = self._call_gnn([tx], "graph")
        xgb_outputs = self._xgb_client.infer(
            [
                {"name": "transactions", "shape": [1], "datatype": "BYTES", "data": [tx]},
                {"name": "embeddings", "shape": [1, len(emb[0])], "datatype": "FP32", "data": emb},
            ]
        )
        prob = float(xgb_outputs[0]["data"][0][1])
        return {
            "fraud_probability": round(prob, 6),
            "decision": self._combine(self._decide(prob), worst),
            "model_version": "remote",
            "model_type": "xgb_ensemble",
            "rules_matched": rules_matched,
        }

    def _score_gnn(self, tx: dict) -> dict:
        rules_matched, blocked, worst = self._run_rules(tx)
        if blocked:
            return {
                "fraud_probability": 1.0,
                "decision": "DECLINE",
                "model_version": "rules",
                "model_type": "gnn_only",
                "rules_matched": rules_matched,
            }
        probs, _ = self._call_gnn([tx], "isolated")
        prob = probs[0]
        return {
            "fraud_probability": round(prob, 6),
            "decision": self._combine(self._decide(prob), worst),
            "model_version": "remote",
            "model_type": "gnn_only",
            "rules_matched": rules_matched,
        }

    def _score_gnn_graph(self, tx: dict) -> dict:
        rules_matched, blocked, worst = self._run_rules(tx)
        if blocked:
            return {
                "fraud_probability": 1.0,
                "decision": "DECLINE",
                "model_version": "rules",
                "model_type": "graph",
                "rules_matched": rules_matched,
            }
        # Rolling graph state is managed inside the GNN service (single-tx
        # score() updates it on every call in graph mode).
        probs, _ = self._call_gnn([tx], "graph")
        prob = probs[0]
        return {
            "fraud_probability": round(prob, 6),
            "decision": self._combine(self._decide(prob), worst),
            "model_version": "remote",
            "model_type": "graph",
            "rules_matched": rules_matched,
        }

    def _decide(self, prob: float) -> str:
        if prob >= self.threshold:
            return "DECLINE"
        if prob >= self.threshold * 0.6:
            return "REVIEW"
        return "APPROVE"


# ── Remote model client ───────────────────────────────────────────────────────


class _ModelClient:
    """
    HTTP client for a single remote model service.

    Configuration via environment variables:
      {PREFIX}_MODEL_URL   — base URL, e.g. http://gnn-service:3000  (required)
      {PREFIX}_MODEL_NAME  — model/endpoint name, e.g. fraud-gnn     (required)
      {PREFIX}_API_TOKEN   — Bearer token                             (optional)
    """

    def __init__(self, prefix: str, timeout: float = 10.0):
        url = os.environ[f"{prefix}_MODEL_URL"].rstrip("/")
        self._model_name = os.environ[f"{prefix}_MODEL_NAME"]
        token = os.environ.get(f"{prefix}_API_TOKEN")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(base_url=url, headers=headers, timeout=timeout)

    def infer(self, inputs: list[dict]) -> list[dict]:
        # BentoML wraps the request body under the parameter name ("request")
        resp = self._client.post(
            f"/models/{self._model_name}/infer",
            json={"request": {"inputs": inputs}},
        )
        resp.raise_for_status()
        return resp.json()["outputs"]

    def reset(self) -> None:
        """Call the service's /reset route (parameterless → empty JSON body)."""
        resp = self._client.post(f"/models/{self._model_name}/reset", json={})
        resp.raise_for_status()

    def ready(self) -> bool:
        """Probe BentoML's built-in /readyz route. False on any error or timeout."""
        try:
            return self._client.get("/readyz", timeout=2.0).is_success
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()
