<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Fintech Fraud Detection Deployment Guide

Solution Blueprints are provided as Helm Charts. The recommended approach to deploy them is to pipe the output of `helm template` to `kubectl apply -f -`.
We do not recommend `helm install`, which by default uses a Secret to keep track of the related resources.
This does not work well with Enterprise clusters that often have limitations on the kinds of resources that regular users are allowed to create.

This blueprint is designed to run on **AMD Instinct** GPUs.

## Hugging Face token

InitContainers download model artifacts and data from Hugging Face at pod startup. The chart expects a Kubernetes Secret named `hf-token` (key: `hf-token`).

1. Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Read access).
2. Ensure you can access the Hugging Face repositories downloaded at pod startup (configured in `values.yaml`):
   - [dklestov-cloud/gnn-fraud](https://huggingface.co/dklestov-cloud/gnn-fraud) (model) — GNN and XGBoost ONNX artifacts (`artifacts.gnn.hfRepo`, `artifacts.xgb.hfRepo`)
   - [dklestov-cloud/gnn-fraud-transactions](https://huggingface.co/datasets/dklestov-cloud/gnn-fraud-transactions) (dataset) — demo transaction CSVs (`artifacts.transactions.hfRepo`)
3. Create the Secret in your target namespace before deploying:

```bash
namespace="my-namespace"
kubectl create secret generic hf-token \
  --from-literal=hf-token="hf_YOUR_TOKEN_HERE" \
  -n $namespace
```

## Deploy

Define the `name` (deployment name) and the `namespace` (Kubernetes namespace), then pipe the output of `helm template` to `kubectl apply -f -`:

```bash
name="my-deployment"
namespace="my-namespace"

helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-fintech-fraud-detection \
  | kubectl apply -f - -n $namespace
```

You can create a namespace with `kubectl create namespace $namespace`.

> **Known issue (Helm 4.2.1+):** Helm 4.2.1 and newer leak `Pulled:`/`Digest:` metadata to stdout ([helm#32215](https://github.com/helm/helm/issues/32215)), which breaks the piped `helm template … | kubectl apply -f -`. Until the fix ships, either use Helm **3.16 – 4.2.0**, or split the pull and template steps, e.g.:
>
> ```bash
> helm pull oci://registry-1.docker.io/amdenterpriseai/aimsb-fintech-fraud-detection --untar
> helm template $name ./aimsb-fintech-fraud-detection \
>   # …same flags as the piped command above…
>   | kubectl apply -f - -n $namespace
> ```

## Connecting

Wait until all pods report `Running` and `Ready` (initContainers download artifacts on first start):

```bash
kubectl get pods -n $namespace
```

## Connect to the UI

### Option 1: Port forwarding

Port-forward to the UI service (port 8080):

```bash
kubectl port-forward svc/$name-aimsb-fintech-fraud-detection-middleware 8080:8080 -n $namespace
```

Then open [http://localhost:8080](http://localhost:8080) and click **▶ Start** on the Live tab.

### Option 2: HTTPRoute (Gateway Access)

If your cluster has a Gateway API compatible gateway (e.g., Kubernetes Gateway, Istio, etc.), you can enable HTTPRoute creation to route traffic through the gateway.

**Prerequisites:**
- A Gateway named `https` must exist in the `envoy-gateway-system` namespace (or configure a different gateway).
- The Gateway must be properly configured with listeners.

**Enabling HTTPRoute:**

Use `--set http_route.enabled=true` in the `helm template` command to enable HTTPRoute creation:

```bash
name="my-deployment"
namespace="my-namespace"

helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-fintech-fraud-detection \
  --set http_route.enabled=true \
  | kubectl apply -f - -n $namespace
```

**Obtaining the URL:**

The URL to access the blueprint via HTTPRoute is formed by the release name and the hostname of the gateway. Use this command to produce the URL by querying the hostname from the cluster:

```bash
echo "https://aimsb-fintech-fraud-detection-$name$(kubectl get gtw https -n envoy-gateway-system -o jsonpath='{.spec.listeners[?(@.name=="https")].hostname}' | tr -d '*')/"
```

## Clean up

When you are finished, remove the deployed resources using the same deployment command, with `kubectl delete` instead of `kubectl apply`:

```bash
name="my-deployment"
namespace="my-namespace"

helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-fintech-fraud-detection \
  | kubectl delete -f - -n $namespace
```

## Model Training

The GNN and XGBoost models were trained on the publicly available [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
dataset (~590 k transactions from the Kaggle competition).
This section describes how both models were built so that users can reproduce the process on their own transaction data.

### Data and Splitting

The dataset consists of two files — transaction records and identity records — joined on a transaction ID.
Records are sorted by `TransactionDT` (a time-offset field) and split 80 % / 20 % into train and test sets in chronological order.
All preprocessing statistics (medians, label encodings, normalisation parameters) are fit exclusively on the training portion to prevent data leakage.

### Transaction Graph Construction

Transactions are represented as nodes in a directed graph.
Edges connect transactions that share the same entity key,
flowing only from earlier to later in time so that message passing cannot aggregate future information.
Two types of entity keys are used:

- **card1** — the payment card identifier
- **uid** — a composite identity key (`card1 + addr1 + D1`) that approximates a device fingerprint

Each node's feature vector is built from the available tabular fields after preprocessing: numeric columns are median-imputed and z-score normalised (clipped to ±10);
categorical columns are label-encoded on train values;
binary match columns (M1–M9) are mapped to 1/0/−1;
high-cardinality identifiers are replaced by their train-frequency and log-count encodings;
card-level amount z-scores and per-product fraud rate scores are added as engineered features.

### GNN Architecture

The GNN is a three-layer GraphSAGE encoder followed by a two-layer MLP classification head:

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1   = nn.LayerNorm(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2   = nn.LayerNorm(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)

class FraudGNN(nn.Module):
    def __init__(self, in_channels, hidden=160, embed_dim=64, head_hidden=48, dropout=0.2):
        super().__init__()
        self.encoder    = GraphSAGEEncoder(in_channels, hidden, embed_dim, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_hidden, 2),
        )

    def forward(self, x, edge_index):
        emb = self.encoder(x, edge_index)
        return self.classifier(emb), emb
```

LayerNorm is used instead of BatchNorm so the model behaves identically at batch size 1 (single-transaction inference) and in batch mode,
and exports cleanly to ONNX without running-statistics issues.

**Training details:** full-batch training with AdamW (lr = 3 × 10⁻³, weight decay = 10⁻⁴) and cosine annealing warm restarts.
Loss is CrossEntropyLoss with inverse-frequency class weights to handle the ~3.5 % fraud rate.
A 15 % temporal validation tail is carved from the training split;
the best checkpoint by val ROC-AUC is kept.
Training runs for up to 500 epochs or 3 600 s wall time, whichever comes first.

After training, the encoder's 64-dimensional output (node embeddings) is extracted for every transaction and passed to the XGBoost stage.

### XGBoost Model

XGBoost is trained on a concatenation of the normalised tabular features and the 64-dimensional GNN embeddings.
Class imbalance is handled by setting `scale_pos_weight = n_negatives / n_positives`.
Early stopping (30 rounds, eval metric: AUC-PR) uses the same 15 % temporal validation tail;
the held-out test set is never seen during fitting.
Key hyperparameters: `n_estimators = 1 000`, `max_depth = 6`, `learning_rate = 0.05`, `subsample = colsample_bytree = 0.8`.

### ONNX Export

Both models are exported to ONNX immediately after training:

- **`gnn.onnx`** — takes `node_features [N, n_feats]` and `edge_index [2, E]` as inputs. Passing an empty `[2, 0]` edge tensor gives isolated (no-graph) inference for a single transaction;
- passing a populated edge index enables rolling-window or batch-graph inference.
- ONNX opset 18 is used to ensure the scatter aggregation ops in SAGEConv export without manual fallback.
- **`xgb.onnx`** — takes `features [N, n_tabular + 64]` (tabular features concatenated with GNN embeddings) and outputs class probabilities.

These two files, together with the serialised preprocessing state (`xgb_preprocessor.joblib`), are the only artifacts needed at inference time. They are stored on HuggingFace and downloaded by the BentoML services at pod startup.

### Training on Your Own Data

To train on a different transaction dataset, the main things to adapt are:

1. **Graph edge keys** — configure which columns identify shared-entity relationships (equivalent to `card1` / `uid` in the IEEE-CIS case).
2. **Feature columns** — the preprocessing pipeline is column-name-driven; adding or removing columns flows through automatically.
3. **Class balance** — `scale_pos_weight` and the GNN class weights are computed from the training split at run time, so no manual adjustment is needed for a different fraud rate.
