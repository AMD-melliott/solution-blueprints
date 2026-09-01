<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

    # Training Guide — GNN + XGBoost Fraud Detection

## Input Data Format

Two CSV files joined on `TransactionID` (IEEE-CIS Fraud Detection schema):

| File | Description |
|------|-------------|
| `data/train_transaction.csv` | ~590k rows — amounts, card fields, D/C/V/M columns |
| `data/train_identity.csv` | ~144k rows — device, network, identity fields (left-joined) |

**Key columns:**

| Column | Type | Description |
|--------|------|-------------|
| `TransactionID` | int | Join key; dropped as a feature |
| `isFraud` | 0/1 | Target label (~3.5% fraud rate) |
| `TransactionDT` | int | Seconds offset — used for temporal train/test split |
| `TransactionAmt` | float | Transaction amount |
| `card1`–`card6` | int/float | Anonymized card attributes |
| `addr1`, `addr2` | float | Billing/shipping zip |
| `P_emaildomain`, `R_emaildomain` | str | Purchaser/recipient email domains |
| `M1`–`M9` | str | Match flags (`T`/`F`; M4 is multi-class) |
| `D1`–`D15` | float | Timedelta features |
| `C1`–`C14` | float | Counting features |
| `V1`–`V339` | float | Vesta-engineered features |
| `id_01`–`id_38` | mixed | Identity/device attributes (from identity CSV) |
| `DeviceType`, `DeviceInfo` | str | Device type and fingerprint |

Place both files under `data/` relative to the working directory (configurable via `config.yaml → data.data_dir`).

---

## Quick Start — Docker (recommended)

Run both training stages from the project root (`fintech-fraud-detection/`). Data must be placed under `data/` beforehand.

### Stage 1: GNN training

```bash
# Build
docker build -f train/Dockerfile.gnn-train -t fraud-gnn-train train/

# Run (CPU)
docker run --rm \
  -v "$(pwd)/data":/app/data:ro \
  -v "$(pwd)/artifacts":/app/artifacts \
  fraud-gnn-train

# Run (AMD GPU / ROCm)
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  -v "$(pwd)/data":/app/data:ro \
  -v "$(pwd)/artifacts":/app/artifacts \
  fraud-gnn-train
```

Produces `artifacts/graph.pt`, `artifacts/model.pt`, `artifacts/gnn.onnx`.

### Stage 2: XGBoost training

```bash
# Build
docker build -f train/Dockerfile.xgb-train -t fraud-xgb-train train/

# Run (CPU)
docker run --rm \
  -v "$(pwd)/artifacts":/app/artifacts \
  fraud-xgb-train

# Run (AMD GPU / ROCm)
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  -v "$(pwd)/artifacts":/app/artifacts \
  fraud-xgb-train
```

Reads `artifacts/model.pt` from Stage 1 and produces `artifacts/xgb_model.joblib`, `artifacts/xgb_preprocessor.joblib`, `artifacts/xgb_features.json`, `artifacts/xgb.onnx`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKIP_GRAPH` | `0` | Set to `1` to reuse an existing `graph.pt` |
| `SKIP_TRAIN` | `0` | Set to `1` to reuse an existing `model.pt` |
| `SKIP_VIZ` | `1` | Set to `0` to generate diagnostic plots |

---

## Retraining Without Docker

> **Prerequisites:** Python 3.10+, PyTorch, PyTorch Geometric, XGBoost, scikit-learn, joblib, onnxmltools, onnxruntime, tqdm, yaml, matplotlib, seaborn.

Run from the project root (`fintech-fraud-detection/`):

```bash
# 1. Build the transaction graph (preprocessing + feature engineering + edge construction)
python -m train.pipeline.graph_builder -c train/config.yaml

# 2. Train the GNN encoder (GraphSAGE → node embeddings) and export ONNX
python -m train.pipeline.gnn_train -c train/config.yaml

# 3. Train XGBoost on tabular features + GNN embeddings, export ONNX
python -m train.pipeline.xgb_train -c train/config.yaml

# 4. Evaluate GNN on the temporal test split
python -m train.pipeline.gnn_eval -c train/config.yaml

# 5. Evaluate XGBoost on the temporal test split
python -m train.pipeline.xgb_eval -c train/config.yaml
```

Steps 1→2 can be piped directly (skips writing `graph.pt` to disk):

```bash
python -m train.pipeline.graph_builder -c train/config.yaml -o - \
  | python -m train.pipeline.gnn_train -c train/config.yaml -i -
```

**Artifacts produced** (in `artifacts/` by default):

| File | Content |
|------|---------|
| `artifacts/graph.pt` | PyG graph bundle (features, edges, train/test split) |
| `artifacts/model.pt` | GNN weights + node embeddings + preprocessing stats |
| `artifacts/gnn.onnx` | GNN encoder in ONNX (all three inference modes) |
| `artifacts/xgb_model.joblib` | Trained XGBoost classifier |
| `artifacts/xgb_preprocessor.joblib` | Featurization state for standalone XGBoost service |
| `artifacts/xgb_features.json` | Ordered feature name list |
| `artifacts/xgb.onnx` | XGBoost model in ONNX |
| `artifacts/xgb_metrics.json` | Test-set metrics (ROC-AUC, AUC-PR, F1, …) |

### Optional steps

```bash
# Generate diagnostic plots (GNN)
python -m train.pipeline.gnn_visualize -c train/config.yaml

# Generate diagnostic plots (XGBoost)
python -m train.pipeline.xgb_visualize -c train/config.yaml

# Verify featurization parity between training and serving paths
python -m train.pipeline.featurize_parity -c train/config.yaml  # exit 0 = OK, exit 1 = drift
```

---

## Model Architecture

```
Input: raw transaction dict
    ↓  graph_builder.preprocess
Node features x  [N, ~350 feats]  — z-scored, clipped at ±10
Edge index       [2, E]           — directed earlier→later, per shared card/uid

┌─────────────────────────────────────────────────────────────┐
│  GraphSAGE Encoder  (model.py: GraphSAGEEncoder)            │
│                                                             │
│  conv1: SAGEConv(n_feats  → 160)                           │
│  LayerNorm(160) → ReLU → Dropout(0.2)                      │
│                                                             │
│  conv2: SAGEConv(160 → 160)                                │
│  LayerNorm(160) → ReLU → Dropout(0.2)                      │
│                                                             │
│  conv3: SAGEConv(160 → 64)   ← node embedding (embed_dim)  │
└─────────────────────────────────────────────────────────────┘
    ↓  embedding  [N, 64]
┌─────────────────────────────────────────────────────────────┐
│  MLP Classifier Head  (model.py: FraudGNN.classifier)       │
│                                                             │
│  Linear(64 → 48) → ReLU → Dropout(0.2) → Linear(48 → 2)   │
└─────────────────────────────────────────────────────────────┘
    ↓  logits [N, 2]  →  softmax  →  fraud_prob [N]

GNN embeddings [N, 64]
    ↓  concatenated with tabular features [N, ~350]
    ↓  → XGBoost input [N, ~414 feats]

┌─────────────────────────────────────────────────────────────┐
│  XGBoost Classifier                                         │
│  n_estimators=1000  max_depth=6  tree_method=hist           │
│  scale_pos_weight ≈ neg/pos  (≈27 for 3.5% fraud rate)     │
└─────────────────────────────────────────────────────────────┘
    ↓  fraud_probability ∈ [0, 1]
```

---

## Training Hyperparameters

### GNN (gnn_train.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Loss** | `CrossEntropyLoss` with class weights | weights = total / (2 · class_count); corrects the ~27× class imbalance |
| **Optimizer** | `AdamW` | lr = **3e-3**, weight_decay = **1e-4** |
| **Scheduler** | `CosineAnnealingWarmRestarts` | T_0 = **2500** epochs, T_mult = **2** |
| **Min LR** | `1e-6` | floor of the cosine schedule |
| **Grad clip** | `1.0` | `clip_grad_norm_` applied every step |
| **Batch size** | Full graph (no mini-batching) | all N nodes in one forward pass |
| **Train / val split** | 85 / 15% | temporal tail used as validation |
| **Stop criterion** | 500 epochs **or** 3600 s wall time | best val ROC-AUC checkpoint saved |
| **Hidden dim** | 160 | SAGEConv layers 1 & 2 |
| **Embedding dim** | 64 | SAGEConv layer 3 = node embedding |
| **Head hidden** | 48 | MLP classifier |
| **Dropout** | 0.2 | applied after each encoder layer and in MLP |

### XGBoost (xgb_train.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Loss** | `logloss` (binary cross-entropy) | `eval_metric = aucpr` for early stopping |
| **n_estimators** | 1000 | upper bound; early stopping typically cuts this |
| **Learning rate** | 0.05 | shrinkage per boosting round |
| **max_depth** | 6 | per-tree depth |
| **subsample** | 0.8 | row sampling per tree |
| **colsample_bytree** | 0.8 | column sampling per tree |
| **scale_pos_weight** | neg / pos | automatic; ≈27 for the reference dataset |
| **Early stopping** | 30 rounds | on AUC-PR of the temporal validation tail |
| **Batch size** | Full dataset | tree-based, no mini-batching |
| **Device** | auto (CUDA / ROCm / CPU) | XGBoost ≥ 2.0 uses `device="cuda"` for GPU |

---

## Featurization Parity

`featurize.py` (this file) is the **single source of truth** for serving-time featurization.
It must be kept byte-identical to the copies in the GNN and XGBoost inference services:

```
train/pipeline/featurize.py                 (training repo — authoritative)
backend/gnn/featurize.py                   (GNN service)
backend/xgb/featurize.py                   (XGBoost service)
```

Run the parity check after any retraining to confirm no drift:

```bash
python -m train.pipeline.featurize_parity -c train/config.yaml   # exit 0 = OK, exit 1 = drift
```
