# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Embedding Service – FastAPI application.

Endpoints
---------
POST /embed     Return SapBERT embedding vector for input text
GET  /health    Liveness + readiness probe
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .providers import EmbeddingProvider, build_embedding_provider
from .schemas import EmbedRequest, EmbedResponse, HealthResponse

logging.basicConfig(
    stream=sys.stdout,
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

_provider: EmbeddingProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provider
    try:
        _provider = build_embedding_provider()
        logger.info("Embedding service ready.")
    except Exception as exc:
        logger.critical("Startup failed: %s", exc)
        raise RuntimeError(str(exc)) from exc

    yield

    _provider = None
    logger.info("Embedding service shut down.")


app = FastAPI(
    title="Embedding Service",
    description="Biomedical text embedding via SapBERT (ONNX or HuggingFace).",
    version="2.0.0",
    lifespan=lifespan,
)


@app.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Embed text using SapBERT",
)
async def embed(request: EmbedRequest) -> EmbedResponse:
    if _provider is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        vector = _provider.embed(request.text)
    except Exception as exc:
        logger.exception("Embedding inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    return EmbedResponse(embedding=vector, dim=len(vector))


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness + readiness probe",
)
async def health() -> HealthResponse:
    if _provider is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_loaded": False},
        )
    return HealthResponse(status="ok", model_loaded=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", "8002")),
        reload=False,
    )
