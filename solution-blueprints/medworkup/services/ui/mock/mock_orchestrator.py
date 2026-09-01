# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Mock orchestrator for UI development.

Why this exists
---------------
The real /analyze pipeline drives four LLM steps through MedGemma, which needs
far more VRAM than a dev workstation typically has. To build and test the UI we
only need responses in the right *shape* — defined by services/orchestrator/app/
schemas.py. This script serves /analyze and /sys/health with hand-authored,
schema-accurate payloads so every UI state can be exercised deterministically
and instantly, with no GPU.

It is NOT part of the production build. It is a dev-only convenience that the
Vite proxy points at by default. The day you want to test against the real
orchestrator, repoint VITE_PROXY_TARGET (see vite.config.ts) — no UI changes.

Run
---
    pip install fastapi uvicorn
    python mock/mock_orchestrator.py
    # serves on http://localhost:8003 (same port the real orchestrator uses)

The responses below are SYNTHESIZED to match schemas.py. They are not recorded
from the real pipeline. If you drop a real example_request_response.json next to
this file, load it instead of CASES for byte-accurate fidelity.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="MOCK Orchestrator (dev only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str


# Artificial latency so the multi-stage loading animation is visible.
# Set to 0 for instant responses while iterating on static layout.
FAKE_LATENCY_S = 3.0


def _matric(text: str) -> str:
    """Pick which canned case to return, by sniffing the note text."""
    t = text.lower()
    if "bull" in t or "erythema migrans" in t or "ranger" in t:
        return "lyme"
    if "cough" in t or "dyspnea" in t or "sputum" in t:
        return "pneumonia"
    if "chest pain" in t or "troponin" in t or "st elevation" in t:
        return "mi"
    return "lyme"


# ── Canned, schema-accurate responses ────────────────────────────────────────
# Each matches AnalyzeResponse: entities_raw, entities_normalized,
# categorized_entities, diagnoses, pipeline_meta.

LYME = {
    "entities_raw": [
        {
            "text": "bull's-eye shaped rash",
            "label": "FINDING",
            "start": 0,
            "end": 22,
            "source": "gliner-biomed",
            "negated": False,
        },
        {
            "text": "erythema migrans",
            "label": "DISEASE",
            "start": 30,
            "end": 46,
            "source": "gliner-biomed",
            "negated": False,
        },
        {
            "text": "bilateral knee joint pain",
            "label": "SYMPTOM",
            "start": 60,
            "end": 85,
            "source": "gliner-biomed",
            "negated": False,
        },
        {
            "text": "low-grade fever",
            "label": "SYMPTOM",
            "start": 100,
            "end": 115,
            "source": "gliner-biomed",
            "negated": False,
        },
        {"text": "fatigue", "label": "SYMPTOM", "start": 130, "end": 137, "source": "gliner-biomed", "negated": False},
        {"text": "tick bite", "label": "FINDING", "start": 150, "end": 159, "source": "gliner-biomed", "negated": True},
        {
            "text": "knee effusion",
            "label": "FINDING",
            "start": 170,
            "end": 183,
            "source": "gliner-biomed",
            "negated": False,
        },
    ],
    "entities_normalized": [
        {
            "canonical": "Erythema migrans",
            "label": "DISEASE",
            "variants": ["bull's-eye shaped rash", "erythema migrans", "annular rash"],
            "mention_count": 3,
            "negated": False,
        },
        {
            "canonical": "Bilateral knee pain",
            "label": "SYMPTOM",
            "variants": ["bilateral knee joint pain", "knee joint pain"],
            "mention_count": 2,
            "negated": False,
        },
        {
            "canonical": "Low-grade fever",
            "label": "SYMPTOM",
            "variants": ["low-grade fever"],
            "mention_count": 1,
            "negated": False,
        },
        {"canonical": "Fatigue", "label": "SYMPTOM", "variants": ["fatigue"], "mention_count": 1, "negated": False},
        {"canonical": "Tick bite", "label": "FINDING", "variants": ["tick bite"], "mention_count": 1, "negated": True},
    ],
    "categorized_entities": {
        "symptoms": ["Bilateral knee pain", "Low-grade fever", "Fatigue", "Morning stiffness"],
        "diseases": ["Erythema migrans", "Lyme disease"],
        "findings": ["Bull's-eye rash", "Knee effusion", "Elevated ESR", "Elevated CRP"],
        "procedures": ["Lyme ELISA", "Knee X-ray"],
        "other": ["Connecticut hiking exposure"],
    },
    "diagnoses": [
        {
            "name": "Infectious Arthritis / Lyme Disease",
            "confidence": 0.80,
            "evidence": [
                "bull's-eye rash",
                "erythema migrans",
                "bilateral knee pain",
                "low-grade fever",
                "fatigue",
                "Connecticut hiking exposure",
                "mild knee effusion",
                "elevated ESR / CRP",
                "Lyme ELISA pending",
            ],
            "reasoning": "The classic erythema migrans rash, recent outdoor exposure in Connecticut, bilateral knee pain, low-grade fever, fatigue, and elevated inflammatory markers strongly suggest Lyme disease with joint involvement.",
            "consistency_note": None,
        },
        {
            "name": "Rheumatoid Arthritis (RA)",
            "confidence": 0.10,
            "evidence": [
                "bilateral knee joint pain",
                "morning stiffness",
                "fatigue",
                "elevated ESR",
                "elevated CRP",
                "mild bilateral knee effusion",
            ],
            "reasoning": "RA can explain bilateral joint pain and morning stiffness, but the acute rash pattern, recent outdoor exposure, fever, and elevated inflammatory markers make Lyme-related arthritis more likely.",
            "consistency_note": "The bull's-eye rash and recent Connecticut hiking exposure strongly point away from primary RA.",
        },
        {
            "name": "Systemic Lupus Erythematosus (SLE)",
            "confidence": 0.05,
            "evidence": [
                "bilateral joint pain",
                "low-grade fever",
                "skin rash",
                "fatigue",
                "elevated inflammatory markers",
            ],
            "reasoning": "SLE can present with rash, fatigue, fever, and joint symptoms, but the bull's-eye morphology, localized thigh rash, and outdoor exposure history are not characteristic of SLE.",
            "consistency_note": "No typical lupus features are documented, such as malar rash, photosensitivity, oral ulcers, renal findings, or cytopenias.",
        },
    ],
    "pipeline_meta": {
        "stage_timings": [
            {"stage": "preprocess", "duration_ms": 120.5},
            {"stage": "ner", "duration_ms": 1840.2},
            {"stage": "aggregate", "duration_ms": 45.1},
            {"stage": "embed", "duration_ms": 2210.8},
            {"stage": "categorize", "duration_ms": 6820.0},
            {"stage": "candidates", "duration_ms": 9450.3},
            {"stage": "reasoning", "duration_ms": 12030.7},
            {"stage": "ranking", "duration_ms": 30.2},
            {"stage": "consistency", "duration_ms": 6980.5},
        ],
        "total_duration_ms": 39528.3,
        "entity_count_raw": 105,
        "entity_count_normalized": 55,
    },
}

# Minimal second/third cases — same shape, lighter content.
PNEUMONIA = {
    "entities_raw": [
        {
            "text": "productive cough",
            "label": "SYMPTOM",
            "start": 0,
            "end": 16,
            "source": "gliner-biomed",
            "negated": False,
        },
        {"text": "dyspnea", "label": "SYMPTOM", "start": 20, "end": 27, "source": "gliner-biomed", "negated": False},
        {
            "text": "pleuritic chest pain",
            "label": "SYMPTOM",
            "start": 30,
            "end": 50,
            "source": "gliner-biomed",
            "negated": False,
        },
        {"text": "hemoptysis", "label": "SYMPTOM", "start": 60, "end": 70, "source": "gliner-biomed", "negated": True},
        {
            "text": "right lower lobe consolidation",
            "label": "FINDING",
            "start": 80,
            "end": 110,
            "source": "gliner-biomed",
            "negated": False,
        },
    ],
    "entities_normalized": [
        {
            "canonical": "Productive cough",
            "label": "SYMPTOM",
            "variants": ["productive cough", "cough with sputum"],
            "mention_count": 2,
            "negated": False,
        },
        {
            "canonical": "Dyspnea",
            "label": "SYMPTOM",
            "variants": ["dyspnea", "shortness of breath"],
            "mention_count": 2,
            "negated": False,
        },
        {
            "canonical": "RLL consolidation",
            "label": "FINDING",
            "variants": ["right lower lobe consolidation"],
            "mention_count": 1,
            "negated": False,
        },
        {
            "canonical": "Hemoptysis",
            "label": "SYMPTOM",
            "variants": ["hemoptysis"],
            "mention_count": 1,
            "negated": True,
        },
    ],
    "categorized_entities": {
        "symptoms": ["Productive cough", "Dyspnea", "Pleuritic chest pain", "Fever"],
        "diseases": ["Community-acquired pneumonia"],
        "findings": ["RLL consolidation", "Neutrophilia", "Elevated CRP", "Elevated procalcitonin"],
        "procedures": ["Chest X-ray", "Blood cultures"],
        "other": ["Type 2 diabetes", "Hypertension"],
    },
    "diagnoses": [
        {
            "name": "Community-Acquired Pneumonia",
            "confidence": 0.85,
            "evidence": ["productive cough", "RLL consolidation", "fever", "neutrophilia", "elevated procalcitonin"],
            "reasoning": "Focal consolidation on imaging with productive cough, fever, and markedly elevated inflammatory markers is classic for bacterial CAP.",
            "consistency_note": None,
        },
        {
            "name": "Pulmonary Embolism",
            "confidence": 0.08,
            "evidence": ["pleuritic chest pain", "dyspnea", "tachycardia"],
            "reasoning": "PE can cause pleuritic pain and dyspnea, but the consolidation, purulent sputum, and procalcitonin favor infection.",
            "consistency_note": "No hypoxia out of proportion and no risk factors documented.",
        },
    ],
    "pipeline_meta": {
        "stage_timings": [
            {"stage": "preprocess", "duration_ms": 110.0},
            {"stage": "ner", "duration_ms": 1700.0},
            {"stage": "aggregate", "duration_ms": 40.0},
            {"stage": "embed", "duration_ms": 2100.0},
            {"stage": "categorize", "duration_ms": 6500.0},
            {"stage": "candidates", "duration_ms": 9000.0},
            {"stage": "reasoning", "duration_ms": 11000.0},
            {"stage": "ranking", "duration_ms": 28.0},
            {"stage": "consistency", "duration_ms": 6500.0},
        ],
        "total_duration_ms": 36978.0,
        "entity_count_raw": 88,
        "entity_count_normalized": 47,
    },
}

MI = {
    "entities_raw": [
        {
            "text": "crushing substernal chest pain",
            "label": "SYMPTOM",
            "start": 0,
            "end": 30,
            "source": "gliner-biomed",
            "negated": False,
        },
        {
            "text": "diaphoresis",
            "label": "SYMPTOM",
            "start": 40,
            "end": 51,
            "source": "gliner-biomed",
            "negated": False,
        },
        {
            "text": "ST elevation",
            "label": "FINDING",
            "start": 60,
            "end": 72,
            "source": "gliner-biomed",
            "negated": False,
        },
        {
            "text": "shortness of breath",
            "label": "SYMPTOM",
            "start": 80,
            "end": 99,
            "source": "gliner-biomed",
            "negated": True,
        },
    ],
    "entities_normalized": [
        {
            "canonical": "Substernal chest pain",
            "label": "SYMPTOM",
            "variants": ["crushing substernal chest pain"],
            "mention_count": 1,
            "negated": False,
        },
        {
            "canonical": "Diaphoresis",
            "label": "SYMPTOM",
            "variants": ["diaphoresis"],
            "mention_count": 1,
            "negated": False,
        },
        {
            "canonical": "ST elevation",
            "label": "FINDING",
            "variants": ["ST elevation"],
            "mention_count": 1,
            "negated": False,
        },
    ],
    "categorized_entities": {
        "symptoms": ["Substernal chest pain", "Diaphoresis", "Nausea", "Radiation to left arm"],
        "diseases": ["STEMI"],
        "findings": ["ST elevation (II/III/aVF)", "Elevated troponin", "Elevated CK-MB"],
        "procedures": ["ECG", "Troponin assay"],
        "other": ["Hypertension", "Hyperlipidemia"],
    },
    "diagnoses": [
        {
            "name": "Inferior STEMI",
            "confidence": 0.92,
            "evidence": [
                "ST elevation (II/III/aVF)",
                "elevated troponin",
                "crushing chest pain",
                "radiation to jaw/arm",
                "diaphoresis",
            ],
            "reasoning": "ST elevation in the inferior leads with reciprocal changes plus markedly elevated troponin is diagnostic of acute inferior STEMI.",
            "consistency_note": None,
        },
        {
            "name": "Aortic Dissection",
            "confidence": 0.04,
            "evidence": ["sudden-onset chest pain"],
            "reasoning": "Considered for sudden severe chest pain, but ST elevation with biomarker rise points to coronary occlusion.",
            "consistency_note": "No tearing/radiating-to-back pain or pulse deficit documented.",
        },
    ],
    "pipeline_meta": {
        "stage_timings": [
            {"stage": "preprocess", "duration_ms": 95.0},
            {"stage": "ner", "duration_ms": 1500.0},
            {"stage": "aggregate", "duration_ms": 35.0},
            {"stage": "embed", "duration_ms": 1900.0},
            {"stage": "categorize", "duration_ms": 6000.0},
            {"stage": "candidates", "duration_ms": 8500.0},
            {"stage": "reasoning", "duration_ms": 10000.0},
            {"stage": "ranking", "duration_ms": 25.0},
            {"stage": "consistency", "duration_ms": 6000.0},
        ],
        "total_duration_ms": 34055.0,
        "entity_count_raw": 64,
        "entity_count_normalized": 38,
    },
}

CASES = {"lyme": LYME, "pneumonia": PNEUMONIA, "mi": MI}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    await asyncio.sleep(FAKE_LATENCY_S)
    return CASES[_matric(req.text)]


@app.get("/health")
async def health():
    return {"status": "ok", "orchestrator_ready": True}


@app.get("/sys/health")
async def sys_health():
    # All green by default. Flip any "status" to "down" / "degraded" to test
    # the header pills against failure states.
    return {
        "status": "ok",
        "services": {
            "orchestrator": {"status": "ok", "latency_ms": 0},
            "ner": {"status": "ok", "latency_ms": 12.3, "http_status": 200},
            "embedding": {"status": "ok", "latency_ms": 8.1, "http_status": 200},
            "llm": {"status": "ok", "latency_ms": 41.7, "http_status": 200},
        },
    }


if __name__ == "__main__":
    import uvicorn

    print("MOCK orchestrator on http://localhost:8003 — dev only, not production.")
    uvicorn.run(app, host="0.0.0.0", port=8003)
