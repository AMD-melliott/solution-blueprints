<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Fraud Detection — GNN + XGBoost Ensemble

GraphSAGE extracts per-transaction embeddings from the card-graph; XGBoost trains on tabular features concatenated with those embeddings.

## Architecture

```
train_transaction.csv + train_identity.csv
         │
         ▼
  graph_builder          → artifacts/graph.pt
  (temporal split, preprocessing, PyG graph)
         │
         ▼
  gnn_train              → artifacts/model.pt
  (GraphSAGE, best val ROC-AUC checkpoint, embeddings stored in bundle)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
  extract_embeddings                   gnn_eval + gnn_visualize
  → artifacts/embeddings.npz           → artifacts/gnn_metrics.json
         │                                  → artifacts/plots/gnn_*.png
         ▼
  xgb_train                            ← reads graph.x (tabular) + embeddings
  → artifacts/xgb_model.joblib         → hstack([tabular, GNN_emb]) → XGBoost
         │
         ▼
  xgb_eval + xgb_visualize
  → artifacts/xgb_metrics.json
  → artifacts/plots/xgb_*.png

Inference:
  POST /score  →  featurize → GNN isolated → embedding → XGBoost
```

## Run locally with make

Place data files in `./data/`:
```
data/
  train_transaction.csv
  train_identity.csv
```

```bash
cd fraud_gnn_xgb

# Full pipeline (graph → GNN → XGBoost → eval → plots)
make pipeline

# Individual steps
make graph          # build PyG graph
make gnn-train      # train GNN
make embeddings     # extract embeddings to .npz
make xgb-train      # train XGBoost on tabular + GNN embeddings
make gnn-eval       # GNN test-set metrics  → artifacts/gnn_metrics.json
make xgb-eval       # XGBoost test metrics  → artifacts/xgb_metrics.json
make gnn-viz        # GNN plots             → artifacts/plots/gnn_*.png
make xgb-viz        # XGBoost plots         → artifacts/plots/xgb_*.png

# Start inference API (dev)
make serve

# Start inference API (production)
make serve-prod
```

Custom config or Python interpreter:
```bash
make pipeline CONFIG=my_config.yaml
make serve PYTHON=python3.11
```

## Run with Docker

Artifacts are shared via a host directory (`./artifacts`). Run containers sequentially.

### 1. Build images

```bash
docker build -t fraud-gnn-train  -f Dockerfile.gnn-train  .
docker build -t fraud-xgb-train  -f Dockerfile.xgb-train  .
docker build -t fraud-infer      -f Dockerfile.infer       .
```

### 2. GNN training

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/artifacts:/app/artifacts \
  fraud-gnn-train
```

Skip re-building the graph if `artifacts/graph.pt` already exists:
```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/artifacts:/app/artifacts \
  -e SKIP_GRAPH=1 \
  fraud-gnn-train
```

### 3. XGBoost training

Requires `artifacts/model.pt` from step 2.
Uses `amd_xgboost` (ROCm GPU-accelerated) — pass GPU devices as with GNN training.

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  -v $(pwd)/artifacts:/app/artifacts \
  fraud-xgb-train
```

### 4. Inference service

Requires `artifacts/model.pt` (GNN) + `artifacts/xgb_model.joblib` (XGBoost).

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  -v $(pwd)/artifacts:/app/artifacts:ro \
  -p 8000:8000 \
  fraud-infer
```

CPU-only (no GPU pass-through):
```bash
docker run --rm \
  -v $(pwd)/artifacts:/app/artifacts:ro \
  -p 8000:8000 \
  fraud-infer
```

## Inference API

**Health check:**
```bash
curl http://localhost:8000/health
```

**Score a transaction (XGBoost ensemble — default, best accuracy):**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 150.0, "ProductCD": "W", "card4": "visa", "P_emaildomain": "gmail.com"}'
```

Response:
```json
{
  "fraud_probability": 0.034,
  "decision": "APPROVE",
  "model_version": "142",
  "model_type": "xgb_ensemble"
}
```

**Scoring modes** (`?mode=` query param):

| Mode | Description |
|---|---|
| `xgb_ensemble` | GNN embedding + tabular → XGBoost (**default**, best) |
| `gnn_only` | GNN isolated forward pass |
| `graph` | GNN with rolling neighbor mini-graph |

```bash
curl -X POST "http://localhost:8000/score?mode=gnn_only" \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 150.0}'
```

**Interactive docs:** http://localhost:8000/docs

## Artifact layout

```
artifacts/
├── graph.pt               # PyG graph bundle (preprocessing + graph structure)
├── model.pt               # GNN weights + per-node embeddings
├── embeddings.npz         # standalone embeddings (train/test split)
├── xgb_model.joblib       # trained XGBoost classifier
├── xgb_features.json      # feature name list (tabular + gnn_emb_*)
├── gnn_metrics.json       # GNN test-set metrics
├── xgb_metrics.json       # XGBoost test-set metrics
├── gnn_predictions.csv    # per-transaction GNN scores
├── xgb_predictions.csv    # per-transaction XGBoost scores
└── plots/
    ├── gnn_training_curves.png
    ├── gnn_roc_pr_curves.png
    ├── gnn_confusion_matrix.png
    ├── gnn_score_dist.png
    ├── xgb_feature_importance.png
    ├── xgb_roc_pr_curves.png
    ├── xgb_confusion_matrix.png
    └── xgb_score_dist.png
```

## Config highlights (`config.yaml`)

| Key | Default | Description |
|---|---|---|
| `split.test_fraction` | 0.20 | Temporal test fraction |
| `model.embed_dim` | 64 | GNN embedding size |
| `training.max_wall_seconds` | 3600 | GNN wall-clock training budget |
| `xgb.use_gnn_embeddings` | true | False = tabular-only XGBoost |
| `xgb.early_stopping_rounds` | 30 | XGBoost early stopping |
| `eval.threshold` | 0.5 | Decision threshold |
