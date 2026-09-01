<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# NER Service — GLiNER Biomedical NER

A FastAPI service that performs biomedical Named Entity Recognition using
[GLiNER](https://github.com/urchade/GLiNER). Supports two model loading modes:
ONNX (fast, offline) and HuggingFace (auto-download on first start).

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ner` | POST | Recognize biomedical entities in text |
| `/health` | GET | Liveness probe |

## Model Loading Modes

### HuggingFace Mode (default)

Leave `NER_ONNX_PATH` unset. The model is downloaded from HuggingFace Hub on
first startup and cached in `HF_HOME` (default: `/models/hf_cache`).

```bash
NER_ONNX_PATH=
NER_HF_MODEL_ID=Ihor/gliner-biomed-large-v1.0
HF_DEVICE=cpu                    # cpu | cuda | auto
HUGGING_FACE_HUB_TOKEN=hf_...   # only needed for private models
```

### ONNX Mode

Set `NER_ONNX_PATH` to a directory containing a `gliner_config.json` artifact
(produced by `model.save_pretrained()`).

```bash
NER_ONNX_PATH=/models/gliner_onnx
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NER_ONNX_PATH` | `""` | Path to ONNX artifact directory. Empty = HuggingFace mode. |
| `NER_HF_MODEL_ID` | `Ihor/gliner-biomed-large-v1.0` | HuggingFace model ID |
| `HF_DEVICE` | `cpu` | Inference device: `cpu`, `cuda`, or `auto` |
| `HF_HOME` | `/models/hf_cache` | HuggingFace cache directory |
| `APP_PORT` | `8001` | Listening port |
| `OMP_NUM_THREADS` | `4` | CPU threads for ONNX runtime |

## Local Development

```bash
cd services/ner
pip install -r requirements.txt
APP_PORT=8001 uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Startup Logs

```
Model loading mode: ONNX
Loading model from /models/gliner_onnx
```
or
```
Model loading mode: HUGGINGFACE
Downloading model from HuggingFace: Ihor/gliner-biomed-large-v1.0
```
