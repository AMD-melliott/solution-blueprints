# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""FastAPI app: control endpoints (REST) + SSE stream for live transactions."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from app.config import ALLOWED_ORIGINS, WORKER_URL
from app.data_loader import get_dataset, init_dataset
from app.stream import _card_label, _decide, engine, sse_event_stream
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    ds = init_dataset()
    log.info("loaded demo dataset: %d rows", ds.total_rows)
    await engine.startup()
    yield
    await engine.shutdown()


app = FastAPI(title="FT-1 Middleware", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST: status & control ─────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "scorer_mode": engine._scorer.mode if engine._scorer else None}


@app.get("/warmup/status")
async def warmup_status():
    """Aggregated readiness for the UI warm-up screen.

    Reports the demo dataset plus the backend orchestrator and its GNN / XGBoost
    model services (via the backend's /health/deep). In fake scoring mode (no
    WORKER_URL) there is no backend to probe, so readiness only requires the
    dataset.
    """
    ds = get_dataset()
    checks: dict[str, bool] = {"dataset": ds.total_rows > 0}
    scorer_mode = engine._scorer.mode if engine._scorer else None
    if scorer_mode == "real":
        backend = gnn = xgb = False
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{WORKER_URL}/health/deep", timeout=10.0)
                data = resp.json()
                backend = resp.is_success and bool(data.get("model_loaded"))
                gnn = bool(data.get("services", {}).get("gnn"))
                xgb = bool(data.get("services", {}).get("xgb"))
        except Exception as exc:
            log.warning("warmup: backend /health/deep unreachable: %s", exc)
        checks |= {"backend": backend, "gnn": gnn, "xgb": xgb}
    return {
        "ready": all(checks.values()),
        "scorer_mode": scorer_mode,
        "dataset_rows": ds.total_rows,
        "checks": checks,
    }


@app.get("/stream/status")
async def stream_status():
    return engine.status()


@app.post("/stream/start")
async def stream_start():
    await engine.start()
    return engine.status()


@app.post("/stream/pause")
async def stream_pause():
    await engine.pause()
    return engine.status()


@app.post("/stream/stop")
async def stream_stop():
    await engine.stop()
    return engine.status()


class ThresholdPatch(BaseModel):
    review: float = Field(..., ge=0.0, le=1.0)
    fraud: float = Field(..., ge=0.0, le=1.0)


@app.patch("/config/thresholds")
async def patch_thresholds(body: ThresholdPatch):
    await engine.set_thresholds(body.review, body.fraud)
    return engine.status()


@app.post("/batch/score")
async def batch_score(
    count: int = Query(default=100, ge=1, le=10_000),
    mode: str = Query(default="xgb_ensemble", pattern="^(xgb_ensemble|gnn_only|graph)$"),
):
    """Score a slice of the dataset in one backend call and return all results."""
    if engine._scorer is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="Scorer not ready")
    rows = list(get_dataset().iter_rows())[:count]
    if not rows:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Dataset is empty")
    probs, latency_ms = await engine._scorer.score_batch(rows, mode=mode)
    th = engine.thresholds
    per_tx_ms = latency_ms // max(len(rows), 1)
    results = [
        {
            "transaction_id": int(row["TransactionID"]),
            "transaction_dt": row.get("TransactionDT"),
            "transaction_amt": float(row.get("TransactionAmt") or 0),
            "card_label": _card_label(row),
            "score": round(prob, 4),
            "decision": _decide(prob, th),
            "latency_ms": per_tx_ms,
            "is_fraud_gt": row.get("isFraud"),
            "ground_truth_match": None,
        }
        for row, prob in zip(rows, probs)
    ]
    return {"results": results, "n_transactions": len(results)}


# ─── SSE: live event stream ─────────────────────────────────────────────────


@app.get("/stream/events")
async def stream_events():
    q = engine.subscribe()

    async def gen():
        try:
            async for event in sse_event_stream(q):
                yield event
        finally:
            engine.unsubscribe(q)

    return EventSourceResponse(gen())


_dist = Path(__file__).resolve().parent.parent / "client_dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")
