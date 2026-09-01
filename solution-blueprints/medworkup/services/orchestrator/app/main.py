# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
MedWorkUp Orchestrator – FastAPI application.

Endpoints
---------
POST /analyze        Run the full 9-step pipeline on a clinical note
GET  /health         Liveness probe (fast – no downstream calls)
GET  /sys/health     System health – probes all downstream services
GET  /pipeline       Describe pipeline stages
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .pipeline import PipelineOrchestrator
from .schemas import AnalyzeRequest, AnalyzeResponse, ErrorResponse
from .utils.logging import configure_root, set_request_id

configure_root(settings.log_level)
logger = logging.getLogger(__name__)

# ── shared state ──────────────────────────────────────────────────────────────
_orchestrator: PipelineOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator
    logger.info("Initialising pipeline orchestrator …")
    _orchestrator = PipelineOrchestrator()
    logger.info("Orchestrator ready.")
    yield
    _orchestrator = None
    logger.info("Orchestrator shut down.")


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MedWorkUp Orchestrator",
    description="Phase 1 clinical pipeline",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Pipeline-Duration-Ms", "X-Request-Id"],
)


# ── middleware: request timing + correlation ID ───────────────────────────────
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or set_request_id()
    set_request_id(rid)
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Pipeline-Duration-Ms"] = str(duration_ms)
    response.headers["X-Request-Id"] = rid
    return response


# ── routes ────────────────────────────────────────────────────────────────────


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Run clinical NLP pipeline on a SOAP note",
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Pipeline error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    if _orchestrator is None:
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(error="Orchestrator not initialised.", stage="startup").model_dump(),
        )
    try:
        return await _orchestrator.run(request.text)
    except Exception as exc:
        logger.exception("Pipeline error")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error=str(exc), stage="pipeline").model_dump(),
        )


@app.get("/health", summary="Liveness probe (fast)")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "orchestrator_ready": _orchestrator is not None,
    }


@app.get("/sys/health", summary="Full system health — probes all downstream services")
async def sys_health() -> Dict[str, Any]:
    """
    Concurrently probes NER, embedding, and LLM health endpoints.
    Returns individual service status and latency.
    Used by the UI header to show live service dots.
    """
    probe_timeout = 5.0  # seconds per service

    async def probe(name: str, url: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=probe_timeout) as client:
                resp = await client.get(url)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            ok = resp.status_code < 400
            return {"status": "ok" if ok else "degraded", "latency_ms": latency_ms, "http_status": resp.status_code}
        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            return {"status": "down", "latency_ms": latency_ms, "error": str(exc)[:120]}

    results = await asyncio.gather(
        probe("ner", f"{settings.ner_base_url}/health"),
        probe("embedding", f"{settings.embedding_base_url}/health"),
        probe("llm", f"{settings.llm_base_url}/health"),
        return_exceptions=False,
    )

    ner_r, emb_r, llm_r = results
    all_up = all(r["status"] == "ok" for r in results)

    return {
        "status": "ok" if all_up else "degraded",
        "services": {
            "orchestrator": {"status": "ok", "latency_ms": 0},
            "ner": ner_r,
            "embedding": emb_r,
            "llm": llm_r,
        },
    }


@app.get("/pipeline", summary="Describe pipeline stages")
async def pipeline_info() -> Dict[str, Any]:
    return {
        "stages": [
            {"step": 1, "name": "preprocess", "type": "deterministic", "service": "local"},
            {"step": 2, "name": "ner", "type": "deterministic", "service": "ner-service"},
            {"step": 3, "name": "aggregate", "type": "deterministic", "service": "local"},
            {"step": 4, "name": "embed", "type": "deterministic", "service": "embedding-service"},
            {"step": 5, "name": "categorize", "type": "llm", "service": "medgemma"},
            {"step": 6, "name": "candidates", "type": "llm", "service": "medgemma"},
            {"step": 7, "name": "reasoning", "type": "llm", "service": "medgemma"},
            {"step": 8, "name": "ranking", "type": "deterministic", "service": "local"},
            {"step": 9, "name": "consistency", "type": "llm", "service": "medgemma"},
        ]
    }


# ── global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal server error", detail=str(exc)).model_dump(),
    )


# ── dev entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.app_port, reload=False)
