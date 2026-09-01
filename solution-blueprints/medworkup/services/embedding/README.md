<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Embedding Service — SapBERT Biomedical Embeddings

A FastAPI service that produces dense vector embeddings for biomedical entity
mentions using [SapBERT](https://github.com/cambridgeltl/SapBERT). Used by the
orchestrator to compute cosine-similarity clusters for entity normalization.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/embed` | POST | Compute embeddings for a list of entity strings |
| `/health` | GET | Liveness probe |

## Model Loading Modes

### HuggingFace Mode (default)

Leave `EMBEDDING_ONNX_PATH` unset. Model auto-downloads on first startup.

```bash
EMBEDDING_ONNX_PATH=
EMBEDDING_HF_MODEL_ID=cambridgeltl/SapBERT-from-PubMedBERT-fulltext
HF_DEVICE=cpu
```

### ONNX Mode

Set `EMBEDDING_ONNX_PATH` to a directory containing a `model.onnx` artifact.

```bash
EMBEDDING_ONNX_PATH=/models/sapbert_onnx
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_ONNX_PATH` | `""` | Path to ONNX artifact directory. Empty = HuggingFace mode. |
| `EMBEDDING_HF_MODEL_ID` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | HuggingFace model ID |
| `HF_DEVICE` | `cpu` | Inference device: `cpu`, `cuda`, or `auto` |
| `HF_HOME` | `/models/hf_cache` | HuggingFace cache directory |
| `APP_PORT` | `8002` | Listening port |
| `OMP_NUM_THREADS` | `4` | CPU threads for ONNX runtime |

## Local Development

```bash
cd services/embedding
pip install -r requirements.txt
APP_PORT=8002 uvicorn app.main:app --host 0.0.0.0 --port 8002
```
