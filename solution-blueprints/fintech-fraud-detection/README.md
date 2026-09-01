<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Fintech Fraud Detection

Real-time financial transaction fraud detection demo running on AMD GPUs in Kubernetes. A Graph Neural Network and XGBoost ensemble scores each transaction against a deterministic rules engine and classifies it as legitimate, requiring review, or fraudulent. A streaming React frontend replays the IEEE-CIS dataset at configurable speed, showing live fraud decisions and financial impact metrics.

For local development see section below. In-depth information about the Solution Blueprint, including deployment details can be found in the docs folder:

- [docs/README.md](docs/README.md) — architecture, components, system requirements, configuration options
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — full deployment guide, all Helm values, HTTPRoute setup, models

## Starting the service

### Prerequisites

- **Helm 3.x** and **kubectl** configured to reach the target cluster
- **Kubernetes cluster** with at least 2 AMD GPUs (ROCm 7.1.1+) and the AMD GPU device plugin installed
- A default **StorageClass** that can provision `ReadWriteOnce` volumes (9 Gi total across three PVCs)
- A **Hugging Face token** (Read access) — see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#hugging-face-token) for details

### Services

| Service | Port | Description |
|---|---|---|
| **GNN Service** | 3000 | BentoML Graph Neural Network inference on AMD GPU |
| **XGBoost Service** | 3001 | BentoML XGBoost inference on AMD GPU |
| **Backend** | 8000 | FastAPI orchestrator — rules engine + GNN/XGBoost scoring |
| **Middleware + UI** | 8080 | FastAPI SSE stream + React frontend |

The Middleware + UI service handles the demo replay loop:
- Downloads the IEEE-CIS transaction dataset via an initContainer at pod startup
- Loads up to `maxDemoRows` rows into memory
- Iterates rows at configurable speed, calling the Backend `/score` endpoint per transaction
- Broadcasts scored results to the browser via Server-Sent Events (SSE)
- Serves the React frontend as static files

### Deploy

Create the Hugging Face secret before deploying — see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#hugging-face-token). Then from this chart directory:

```bash
name="my-deployment"
namespace="my-namespace"

helm dependency build

helm template $name . \
  | kubectl apply -f - -n $namespace
```

Wait for all pods to become ready (initContainers download artifacts on first start):

```bash
kubectl get pods -n $namespace
```

### Connect to the UI

```bash
kubectl port-forward svc/$name-aimsb-fintech-fraud-detection-middleware 8080:8080 -n $namespace
```

Open [http://localhost:8080](http://localhost:8080) and click **▶ Start** on the Live tab.

## Terms of Use

Copyright © Advanced Micro Devices, Inc., or its affiliates. Licensed under the MIT License.
