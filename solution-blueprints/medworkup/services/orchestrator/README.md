<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# MedWorkUp Orchestrator – Phase 1

A production-ready orchestrator that turns raw clinical SOAP notes into a structured differential diagnosis.

```
POST /analyze
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Step 1  Preprocessing    (local)   SOAP detect, negation │
│  Step 2  NER ×2           (HTTP)    parallel gliner + MedCAT │
│  Step 3  Aggregation      (local)   span merge + dedup    │
│  Step 4  Embeddings       (HTTP)    cosine cluster        │
│  Step 5  Categorize       (LLM)     symptoms/diseases/findings │
│  Step 6  Candidates       (LLM)     differential list     │
│  Step 7  Reasoning        (LLM)     evidence-grounded CoT │
│  Step 8  Ranking          (local)   sort + filter         │
│  Step 9  Consistency      (LLM)     cross-check + adjust  │
└──────────────────────────────────────────────────────────┘
       │
       ▼
  AnalyzeResponse JSON
```

---

## Project Structure

```
orchestrator/
├── .env.orchestrator.example
├── example_request_response.json
├── orchestrator-compose-snippet.yml   ← add to existing docker-compose
└── services/
    └── orchestrator/
        ├── Dockerfile
        ├── requirements.txt
        └── app/
            ├── main.py            ← FastAPI entry point
            ├── config.py          ← all env-var settings
            ├── schemas.py         ← shared Pydantic models
            ├── clients/
            │   ├── base.py        ← async HTTP + retry
            │   ├── ner_client.py
            │   ├── embedding_client.py   ← with in-process cache
            │   └── llm_client.py        ← JSON-safe chat helper
            ├── prompts/
            │   ├── categorization.py
            │   ├── candidates.py
            │   ├── reasoning.py         ← constrained CoT
            │   └── consistency.py
            ├── pipeline/
            │   ├── orchestrator.py      ← wires all steps
            │   ├── step1_preprocess.py
            │   ├── step2_ner.py
            │   ├── step3_aggregate.py
            │   ├── step4_embeddings.py
            │   ├── step5_categorize.py
            │   ├── step6_candidates.py
            │   ├── step7_reasoning.py
            │   ├── step8_ranking.py
            │   └── step9_consistency.py
            └── utils/
                ├── timing.py
                └── logging.py
```

---

## Prerequisites

The three services from Phase 0 must be running:

| Service | Port |
|---|---|
| MedGemma (LLM) | 8000 |
| NER Service | 8001 |
| Embedding Service | 8002 |
| MedCAT Service | 8004 |

---

## Quick Start

### 1. Add env vars

Append `.env.orchestrator.example` contents to your existing `.env`:

```bash
cat .env.orchestrator.example >> .env
```

### 2. Add the orchestrator to docker-compose

Copy the service block from `orchestrator-compose-snippet.yml` into your `docker-compose.yml` under `services:`.

### 3. Build and start

```bash
docker compose up --build orchestrator
```

Or start everything together:

```bash
docker compose up --build
```

### 4. Test it

```bash
curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient presents with SOB and CP. BP 158/96. ECG shows ST elevation. History of HTN and DM. Rule out PE."
  }' | python3 -m json.tool
```

Check pipeline stage info:

```bash
curl http://localhost:8003/pipeline | python3 -m json.tool
```

---

## Configuration

All settings are env-var driven via `config.py`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8000` | MedGemma endpoint |
| `NER_BASE_URL` | `http://localhost:8001` | NER service |
| `EMBEDDING_BASE_URL` | `http://localhost:8002` | Embedding service |
| `ENTITY_SIMILARITY_THRESHOLD` | `0.85` | Cosine sim for clustering |
| `DIAGNOSIS_MIN_CONFIDENCE` | `0.05` | Drop diagnoses below this |
| `EXPAND_ABBREVIATIONS` | `true` | Expand clinical abbreviations |
| `DEBUG_PIPELINE` | `false` | Log intermediate stage outputs |
| `LLM_TEMPERATURE` | `0.1` | LLM sampling temperature |
| `LOG_LEVEL` | `info` | `debug`/`info`/`warning`/`error` |

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| One NER model down | Continues with the other model's results |
| LLM returns invalid JSON | Falls back to safe defaults (raw entity list) |
| Embedding service down | Entities not clustered; each stays independent |
| All downstream services down | Returns 503 with structured error |

---

## Extending for Phase 2

**SNOMED/ICD mapping**: In `step4_embeddings.py`, replace the cosine clustering loop with a UMLS API call. The interface (`List[AggregatedEntity]` → `List[NormalizedEntity]`) doesn't change.

**Knowledge graph**: Add a `step4b_kg_enrichment.py` between steps 4 and 5 that looks up canonical entities in a graph store and attaches related concepts.

**Caching across requests**: Replace the in-process `EmbeddingClient._cache` dict with Redis.

**API gateway**: Add nginx upstream blocks pointing to `:8001`, `:8002`, `:8003`; no service code changes required.
