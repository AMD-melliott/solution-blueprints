# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status
from fraud_app.predictor import CombinedPredictor
from fraud_app.schemas import BatchScoreRequest, BatchScoreResponse, FraudScore, TransactionIn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)
_predictor: CombinedPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = CombinedPredictor()
    yield
    _predictor = None


app = FastAPI(
    title="Fraud Detection API",
    version="0.1.0",
    description="GNN + XGBoost ensemble fraud scoring.",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
def health():
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
    }


@app.get("/health/deep", tags=["ops"])
def health_deep():
    """Readiness of this service plus the remote GNN and XGBoost services.

    Kept separate from /health so the k8s probes stay cheap — this endpoint
    makes a network call to each model service on every request.
    """
    services = _predictor.dependencies_ready() if _predictor is not None else {"gnn": False, "xgb": False}
    ready = _predictor is not None and all(services.values())
    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": _predictor is not None,
        "services": services,
        "ready": ready,
    }


@app.post("/score", response_model=FraudScore, tags=["scoring"])
def score(
    tx: TransactionIn,
    mode: str = Query(
        "xgb_ensemble",
        pattern="^(xgb_ensemble|gnn_only|graph)$",
        description=(
            "xgb_ensemble: tabular+GNN embedding → XGBoost (default, best accuracy). "
            "gnn_only: GNN isolated. "
            "graph: GNN with rolling neighbor context."
        ),
    ),
):
    safe_mode = mode.replace("\r", "").replace("\n", "")
    safe_amt = str(tx.TransactionAmt).replace("\r", "").replace("\n", "")
    _log.info("→ /score mode=%s amt=%s", safe_mode, safe_amt)
    if _predictor is None:
        _log.error("/score called but model not loaded")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        result = _predictor.score(tx.model_dump(), mode=mode)
        _log.info("← /score prob=%.4f decision=%s", result["fraud_probability"], result["decision"])
        return result
    except Exception as exc:
        _log.exception("/score failed: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.post("/reset", tags=["ops"])
def reset():
    """Clear stateful inference context (the GNN rolling graph).

    Called by the middleware when the demo stream is stopped so the next run
    starts from an empty graph instead of carrying over the previous run.
    """
    if _predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        _predictor.reset()
        _log.info("✓ /reset — GNN rolling graph cleared")
        return {"status": "ok"}
    except Exception as exc:
        _log.exception("/reset failed: %s", exc)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["scoring"])
def score_batch(req: BatchScoreRequest):
    """
    Score a list of transactions in a single forward pass.

    In xgb_ensemble and graph modes an intra-batch mini-graph is built:
    transactions that share card1 / P_emaildomain / R_emaildomain become
    neighbors, giving the GNN proper message-passing context.
    """
    if _predictor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")
    try:
        txs = [tx.model_dump() for tx in req.transactions]
        results = _predictor.score_batch(txs, mode=req.mode)
        return BatchScoreResponse(results=results, n_transactions=len(results))
    except Exception as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
