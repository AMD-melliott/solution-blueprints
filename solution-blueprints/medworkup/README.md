<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# MedWorkUp — Clinical Diagnostic Assistant

End-to-end clinical NLP pipeline with a live web UI.

```
Browser
  └── http://localhost:8080 (nginx UI service)
        ├── /                → static index.html
        ├── /analyze         → orchestrator POST /analyze
        ├── /health          → orchestrator GET  /health
        ├── /sys/health      → orchestrator GET  /sys/health
        └── /pipeline        → orchestrator GET  /pipeline

orchestrator:8003
  ├── NER:       http://ner-service:8001
  ├── Embedding: http://embedding-service:8002
  ├── MedCAT:    http://medcat-service:8004
  └── LLM:       http://medgemma:8000
```

## Services

| Service | Port (host) | Description |
|---|---|---|
| `ui` | **8080** | nginx — static UI + reverse proxy. **Primary entry point.** |
| `orchestrator` | 8003 | FastAPI NLP pipeline. Also accessible directly for dev/debug. |
| `ner-service` | 8001 | GLiNER biomedical NER (ONNX or HuggingFace auto-download) |
| `embedding-service` | 8002 | SapBERT ONNX embedding |
| `medcat-service` | 8004 | MedCAT NER+L REST API — concept recognition & UMLS/SNOMED linking |
| `medgemma` | 8000 | MedGemma 27B via AMD AIM |

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker ≥ 24 + Compose v2 | `docker compose version` |
| AMD ROCm drivers + `/dev/kfd` `/dev/dri` | Required for MedGemma |
| HuggingFace token (`HUGGING_FACE_HUB_TOKEN`), from an account with MedGemma license access granted | Required for MedGemma model download |
| KCL licence key (`UMLS_API_KEY`) | Required for MedCAT model download — register at https://medcat.sites.er.kcl.ac.uk |
| KCL personal details | `MEDCAT_FIRST_NAME`, `MEDCAT_LAST_NAME`, `MEDCAT_EMAIL`, `MEDCAT_AFFILIATION`, `MEDCAT_USE_CASE` — submitted to KCL download form |

> **MedGemma is a gated model.** A HuggingFace token alone will not download it — the account
> that issued the token must first accept the license at
> https://huggingface.co/google/medgemma-27b-it (see [Third-Party Components](#third-party-components)
> for the license terms). Without this step the MedGemma container will crash-loop on a
> HuggingFace `401`/`403` error even with a valid token.

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env — set all required fields:
#   HUGGING_FACE_HUB_TOKEN  — HuggingFace token for MedGemma
#   UMLS_API_KEY            — KCL licence key for MedCAT model download
#   MEDCAT_FIRST_NAME       — \
#   MEDCAT_LAST_NAME        —  | Personal details required by KCL download form
#   MEDCAT_EMAIL            —  |
#   MEDCAT_AFFILIATION      —  |
#   MEDCAT_USE_CASE         — /
#   MEDCAT_MODEL            — model to download (default: snomed_mimic)

# 2. Download the MedCAT model, then build and start everything
bash scripts/download_medcat_model.sh
docker compose up --build -d

# 3. Open the UI
open http://localhost:8080
```

## Kubernetes Deployment

This repository ships with a Helm chart for Kubernetes deployment. See
`docs/DEPLOYMENT.md` for the full step-by-step guide including configuration,
monitoring, and troubleshooting.

Quick overview:

```bash
# 1. Build and push the four custom images
export REGISTRY="your.registry.com/medworkup"
export TAG="v1.0.0"

docker build -t $REGISTRY/orchestrator:$TAG services/orchestrator/
docker build -t $REGISTRY/ner:$TAG          services/ner/
docker build -t $REGISTRY/embedding:$TAG    services/embedding/
docker build -t $REGISTRY/ui:$TAG           services/ui/

docker push $REGISTRY/orchestrator:$TAG $REGISTRY/ner:$TAG \
            $REGISTRY/embedding:$TAG    $REGISTRY/ui:$TAG

# 2. Create my-values.yaml with your registry, credentials, and GPU model
#    (see docs/DEPLOYMENT.md Step 3 for the full annotated example)

# 3. Deploy
kubectl create namespace medworkup
helm dependency build .
helm template medworkup . \
  -f my-values.yaml \
  | kubectl apply -f - -n medworkup
```

## Model Loading Modes

The NER and Embedding services support two model loading modes, selected at startup via environment variables.

### HuggingFace mode (default)

Leave `NER_ONNX_PATH` and `EMBEDDING_ONNX_PATH` unset. Models are downloaded from HuggingFace on first startup and cached in the `hf-cache` Docker volume. Subsequent restarts load from cache instantly — no re-download.

```bash
# .env — HF mode (default, no artifact setup required)
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
NER_ONNX_PATH=
EMBEDDING_ONNX_PATH=
HF_DEVICE=cpu   # or cuda or auto
```

Cache location inside the container: `/models/hf_cache` (mapped to the `hf-cache` named volume).

### ONNX mode

Set the path to a pre-downloaded artifact directory. The service loads from disk — no network access required.

```bash
# .env — ONNX mode
NER_ONNX_PATH=/models/gliner_onnx          # must contain gliner_config.json
EMBEDDING_ONNX_PATH=/models/sapbert_onnx   # must contain model.onnx
```

Mount the artifact directory in `docker-compose.yml` (commented-out examples are included in the file).

### Startup logs

You will always see which mode is active:

```
Model loading mode: ONNX
Loading model from /models/gliner_onnx
```

or

```
Model loading mode: HUGGINGFACE
Downloading model from HuggingFace: Ihor/gliner-biomed-bi-large-v1.0
```

or (on subsequent starts):

```
Model loading mode: HUGGINGFACE
Loading cached model: Ihor/gliner-biomed-bi-large-v1.0
```

## MedCAT Model Setup

Download the MedCAT model before starting the stack — the download script reads
credentials directly from `.env`.

```bash
# Pre-download (uses MEDCAT_MODEL from .env)
bash scripts/download_medcat_model.sh

# Or override the model at the CLI:
bash scripts/download_medcat_model.sh umls_full
```

Available models: `umls_small`, `umls_full`, `snomed_int`, `snomed_uk`, `snomed_mimic` (default).

The script places the model pack under `artifacts/medcat_models/` and
automatically writes `MEDCAT_MODEL_PACK_PATH` into `.env`.

## Common Operations

```bash
bash scripts/download_medcat_model.sh                                # download MedCAT model artifacts
docker compose logs -f                                                # tail all service logs
docker compose logs -f orchestrator                                   # tail pipeline logs only
docker compose logs -f medcat-service                                 # tail MedCAT service logs
curl -sf http://localhost:8003/sys/health | python3 -m json.tool      # show system health JSON
curl -sf http://localhost:8004/api/health/live | python3 -m json.tool # show MedCAT service info + liveness
docker compose down                                                   # stop everything
docker compose down --remove-orphans && docker image prune -f         # remove containers + dangling images
```

See [Direct API Access](#direct-api-access) below for `/analyze` and MedCAT smoke-test requests.

## Direct API Access

The orchestrator is directly accessible on port 8003:

```bash
# Interactive Swagger UI
open http://localhost:8003/docs

# System health
curl http://localhost:8003/sys/health | python3 -m json.tool

# Run the pipeline
curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with fever and productive cough. SpO2 94%. CXR shows consolidation."}' \
  | python3 -m json.tool
```

The MedCAT service is directly accessible on port 8004:

```bash
# Service info and model details
curl http://localhost:8004/api/info | python3 -m json.tool

# Single document annotation
curl -X POST http://localhost:8004/api/process \
  -H "Content-Type: application/json" \
  -d '{"content":{"text":"Patient presents with fever and productive cough. SpO2 94%."}}' \
  | python3 -m json.tool

# Bulk annotation
curl -X POST http://localhost:8004/api/process_bulk \
  -H "Content-Type: application/json" \
  -d '{"content":[{"text":"Diagnosed with leukemia."},{"text":"History of type 2 diabetes."}]}' \
  | python3 -m json.tool
```

## Configuration

All settings are environment-variable driven via `.env`. This single file is
shared by both docker compose and the `scripts/` tooling.

| Variable | Default | Description |
|---|---|---|
| `HUGGING_FACE_HUB_TOKEN` | — | **Required** for MedGemma |
| `UMLS_API_KEY` | — | **Required** for `scripts/download_medcat_model.sh` |
| `MEDCAT_MODEL` | `snomed_mimic` | Model to download (`umls_small` / `umls_full` / `snomed_int` / `snomed_uk` / `snomed_mimic`) |
| `UI_PORT` | `8080` | Host port for the web UI |
| `LLM_TIMEOUT` | `120` | Seconds to wait for MedGemma response |
| `ENTITY_SIMILARITY_THRESHOLD` | `0.85` | Cosine similarity for entity clustering |
| `DIAGNOSIS_MIN_CONFIDENCE` | `0.05` | Filter diagnoses below this confidence |
| `OMP_NUM_THREADS` | `4` | CPU threads for ONNX services |
| `DEBUG_PIPELINE` | `false` | Log intermediate pipeline outputs |
| `MEDCAT_PORT` | `8004` | Host port for the MedCAT service |
| `MEDCAT_MODEL_PACK_PATH` | — | Container path to model pack `.zip` (set automatically by `scripts/download_medcat_model.sh`) |
| `MEDCAT_CDB_PATH` | — | Container path to concept database (fallback if no model pack) |
| `MEDCAT_VOCAB_PATH` | — | Container path to vocabulary (fallback if no model pack) |
| `MEDCAT_META_PATH_LIST` | — | Colon-separated MetaCAT model paths (optional) |
| `MEDCAT_MODEL_NAME` | `MedCAT` | Informative model name reported by `/api/info` |
| `MEDCAT_BULK_NPROC` | `8` | Worker threads for `/api/process_bulk` |
| `MEDCAT_SERVER_WORKERS` | `1` | Gunicorn workers (increase for parallel `/api/process` requests) |
| `MEDCAT_WORKER_TIMEOUT` | `300` | Gunicorn worker timeout in seconds |
| `MEDCAT_SHM_SIZE` | `8g` | Shared memory for MedCAT container (≥8g required when `MEDCAT_BULK_NPROC > 1`) |
| `MEDCAT_ENABLE_METRICS` | `false` | Expose Prometheus metrics at `/metrics` |
| `MEDCAT_ENABLE_DEMO_UI` | `false` | Enable MedCAT built-in demo web UI |
| `MEDCAT_DEID_MODE` | `false` | Enable de-identification mode (requires a DeID model pack) |

## Project Structure

```
medical-experiments/
├── Chart.yaml                           ← Helm chart metadata
├── values.yaml                          ← Helm configuration defaults
├── templates/                           ← Kubernetes manifest templates
├── docs/
│   ├── README.md                        ← Architecture overview
│   └── DEPLOYMENT.md                    ← Docker Compose + Helm deployment guide
├── docker-compose.yml
├── .env                                 ← single config file (gitignored)
├── .env.example                         ← committed template
├── scripts/
│   └── download_medcat_model.sh
├── artifacts/
│   └── medcat_models/
└── services/
    ├── ui/
    │   ├── Dockerfile             nginx:1.27-alpine
    │   ├── nginx.conf             reverse proxy config
    │   ├── static/                static UI fallback
    │   ├── frontend/              React + TypeScript SPA (Vite)
    │   └── README.md
    ├── orchestrator/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   ├── README.md
    │   └── app/
    │       ├── main.py            FastAPI + CORS + /sys/health
    │       ├── config.py
    │       ├── schemas.py
    │       ├── clients/           ner / embedding / medcat / llm HTTP clients
    │       ├── pipeline/          steps 1-9 + orchestrator
    │       ├── prompts/           LLM prompt templates
    │       └── utils/
    ├── ner/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   ├── README.md
    │   └── app/
    └── embedding/
        ├── Dockerfile
        ├── requirements.txt
        ├── README.md
        └── app/
```

> **Note:** `medcat-service` uses the pre-built Docker Hub image
> `cogstacksystems/medcat-service:latest` and does not have a local `services/`
> directory. All configuration is via environment variables and the
> `artifacts/medcat_models/` volume mount.

## Networking

All services run on a shared Docker bridge network `hc-net`. Inter-service
calls use service-name hostnames (e.g. `http://medcat-service:8004`). Only the
`ui` service needs to be reachable from the host — all other ports are exposed
for development/debugging convenience but not required.

## Troubleshooting

**ONNX services fail to start**
```bash
# Check artifact paths exist and contain model.onnx
docker compose logs ner-service
```

**MedGemma slow or timing out**
```bash
# Increase LLM_TIMEOUT in .env (default 120s)
# Check GPU is accessible
docker compose logs medgemma
```

**MedCAT service fails to start or is unhealthy**
```bash
# Most common cause: missing or misconfigured model artifacts
# Run the download script if you haven't already:
bash scripts/download_medcat_model.sh

docker compose logs -f medcat-service

# Verify your model pack path resolves correctly inside the container
docker compose exec medcat-service ls /cat/models

# Check that MEDCAT_MODEL_PACK_PATH or MEDCAT_CDB_PATH+MEDCAT_VOCAB_PATH
# are set in .env and point to files that exist under artifacts/medcat_models/

# If the service OOMs, reduce MEDCAT_BULK_NPROC to 1 and MEDCAT_SHM_SIZE to 1g
```

**MedCAT /api/process returns empty annotations**
```bash
# The model loaded successfully but found no recognised entities.
# Verify your model pack supports the terminology in your text.
# Check /api/info to confirm which model and ontology are loaded.
curl -sf http://localhost:8004/api/health/live | python3 -m json.tool
```

**UI shows red service dots**
```bash
# Check which service is down
curl http://localhost:8003/sys/health
# Check logs of the failing service
docker compose logs -f
```
