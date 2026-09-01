# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""NER Service – FastAPI application.

Endpoints
---------
POST /ner/gliner-biomed   GLiNER biomedical NER (ONNX or HuggingFace)
GET  /health              Liveness + readiness probe
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .providers import NERProvider, build_ner_provider
from .schemas import HealthResponse, NERRequest, NERResponse

logging.basicConfig(
    stream=sys.stdout,
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

_provider: NERProvider | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provider
    try:
        _provider = build_ner_provider()
        logger.info("NER service ready.")
    except Exception as exc:
        logger.critical("Startup failed: %s", exc)
        raise RuntimeError(str(exc)) from exc

    yield

    _provider = None
    logger.info("NER service shut down.")


app = FastAPI(
    title="NER Service",
    description="Biomedical named-entity recognition via GLiNER (ONNX or HuggingFace).",
    version="2.0.0",
    lifespan=lifespan,
)


@app.post(
    "/ner/gliner-biomed",
    response_model=NERResponse,
    summary="Biomedical NER using GLiNER",
)
async def ner_gliner_biomed(request: NERRequest) -> NERResponse:
    if _provider is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        entities = _provider.predict(
            text=request.text,
            labels=request.effective_labels(),
            threshold=request.threshold,
        )
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    return NERResponse(entities=entities)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness + readiness probe",
)
async def health() -> HealthResponse:
    if _provider is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "models_loaded": []},
        )
    return HealthResponse(status="ok", models_loaded=["gliner-biomed"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", "8001")),
        reload=False,
    )
