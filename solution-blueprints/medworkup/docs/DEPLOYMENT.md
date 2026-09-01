<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# MedWorkUp — Clinical Diagnostic Assistant Deployment Guide

Solution Blueprints are provided as Helm Charts.

The recommended approach to deploy them is to pipe the output of `helm template` to `kubectl apply -f -`.
We do not recommend `helm install`, which by default uses a Secret to keep track of the related resources.
This does not work well with Enterprise clusters that often have limitations on the kinds of resources that
regular users are allowed to create.

## Prerequisites

| Requirement | Notes |
|---|---|
| `helm` ≥ 3.12 | `helm version` |
| `kubectl` configured | `kubectl get nodes` |
| Kubernetes cluster with AMD GPU node | Required for MedGemma |
| HuggingFace token, from an account with MedGemma license access granted | Required for MedGemma model download |
| UMLS API key | License for MedCAT — register at [https://uts.nlm.nih.gov/uts/profile](https://uts.nlm.nih.gov/uts/profile) |

## Credentials

> **Never commit credentials to version control.** Create the Kubernetes Secrets below before deploying.

The chart references pre-existing secrets by name — it does not create secrets from `values.yaml`.

### HuggingFace token (MedGemma model download)

MedGemma is a **gated model** on HuggingFace. A HuggingFace token by itself is not enough —
your HuggingFace account must also have accepted the model's license before the token can
download it:

1. Log in to HuggingFace and open the [MedGemma model page](https://huggingface.co/google/medgemma-27b-it)
2. Review and accept the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms)
   to request access. Access is typically granted automatically.
3. Generate a token from that same account (Settings → Access Tokens) and use it below.

If the license has not been accepted, the MedGemma pod will fail to download the model and
crash-loop with a `401`/`403` error from HuggingFace — a valid-looking token is not sufficient
on its own.

The token is passed through a single Kubernetes Secret referenced by name in `values.yaml`.
Create the secret before deploying:

```bash
kubectl create secret generic hf-token \
  --from-literal=hf-token=<YOUR_HUGGINGFACE_TOKEN> \
  -n $namespace
```

The secret name (`hf-token`) matches the defaults in `values.yaml`. It's referenced twice —
once for NER/Embedding, once for the `aimchart-llm` subchart that deploys MedGemma:

```yaml
secrets:
  hfTokenSecret: hf-token   # used by the NER and Embedding services

medgemma:
  env_vars:
    HF_TOKEN:
      name: hf-token        # used by the aimchart-llm subchart
      key: hf-token
```

If you use a different secret name, override both `secrets.hfTokenSecret` and
`medgemma.env_vars.HF_TOKEN.name`/`.key` to match.

### UMLS API key and MedCAT registration details

Register at [https://uts.nlm.nih.gov/uts/profile](https://uts.nlm.nih.gov/uts/profile) to obtain your UMLS API key, then create the secret:

```bash
kubectl create secret generic medcat-credentials \
  --from-literal=UMLS_API_KEY=<YOUR_UMLS_API_KEY> \
  --from-literal=MEDCAT_FIRST_NAME=<FIRST_NAME> \
  --from-literal=MEDCAT_LAST_NAME=<LAST_NAME> \
  --from-literal=MEDCAT_EMAIL=<EMAIL> \
  --from-literal=MEDCAT_AFFILIATION=<INSTITUTION> \
  --from-literal=MEDCAT_USE_CASE=<USE_CASE> \
  -n $namespace
```

The secret name (`medcat-credentials`) matches the default in `values.yaml`:

```yaml
secrets:
  medcatCredentialsSecret: medcat-credentials
```

Override `secrets.medcatCredentialsSecret` if you use a different secret name.

### External LLM endpoint (optional)

To skip deploying the bundled MedGemma subchart and point the orchestrator at an
already-running OpenAI-compatible endpoint instead, set `medgemma.enabled: false`
and `medgemma.existingService` to its base URL. If that endpoint requires
authentication, set one of:

```yaml
medgemma:
  enabled: false
  existingService: "https://your-llm-endpoint.example.com"
  apiKey: "<YOUR_API_KEY>"        # plain value, or:
  apiKeySecretRef:                # a Secret reference (preferred)
    name: my-llm-api-key
    key: api-key
```

The orchestrator sends this as an `Authorization: Bearer <token>` header on every
request to the LLM. Leave both unset for unauthenticated endpoints (e.g. the
bundled subchart).

## Persistent volumes

The chart creates up to two PersistentVolumeClaims of its own:

| PVC | Default size | Used by | Access mode |
|---|---|---|---|
| `{name}-hf-cache` | 20 Gi | NER, Embedding | ReadWriteOnce |
| `{name}-model-artifacts` | 20 Gi | MedCAT | ReadWriteOnce |

> **Multi-node caveat (NER + Embedding):** The shared `hf-cache` PVC is still ReadWriteOnce, so NER and Embedding must land on the same node. On single-node clusters or clusters where both pods are guaranteed to co-schedule this is not an issue. If you need to spread them across nodes, set an RWX-capable StorageClass:
>
> ```yaml
> storage:
>   hfCache:
>     storageClassName: "your-rwx-storage-class"
> ```

The `hf-cache` PVC is gated behind `storage.hfCache.enabled`, while `model-artifacts` is gated behind `medcat.enabled`; both persist across redeployments to avoid re-downloading models. Delete them explicitly during clean-up if a full reset is needed (see [Clean Up](#clean-up)).

MedGemma's own model cache is managed separately by the `aimchart-llm` subchart, as an ephemeral volume claim template (`medgemma.storage.ephemeral`, default 80 Gi on the `mlstorage` StorageClass) rather than a static PVC — it doesn't appear in the table above, and is torn down and recreated per-pod like the rest of the subchart's storage.

## Deploy

With the credential secrets in place:

```bash
name="medworkup"
namespace="default"

helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-medworkup \
  -f values.yaml \
  | kubectl apply -f - -n $namespace
```

> **Known issue (Helm 4.2.1+):** Helm 4.2.1 and newer leak `Pulled:`/`Digest:` metadata to stdout ([helm#32215](https://github.com/helm/helm/issues/32215)), which breaks the piped `helm template … | kubectl apply -f -`. Until the fix ships, either use Helm **3.16 – 4.2.0**, or split the pull and template steps, e.g.:
>
> ```bash
> helm pull oci://registry-1.docker.io/amdenterpriseai/aimsb-medworkup --untar
> helm template $name ./aimsb-medworkup \
>   # …same flags as the piped command above…
>   | kubectl apply -f - -n $namespace
> ```

> On first run, MedCAT (~2–5 GB) and MedGemma (~54 GB) models are downloaded automatically.
> Allow 15–30 minutes for MedCAT to become ready, and 30–90 minutes for MedGemma.

To pull the application images from a private registry, provide a pre-existing Kubernetes image pull secret:

```bash
helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-medworkup \
  -f values.yaml \
  --set imagePullSecrets[0].name=$secret_name \
  | kubectl apply -f - -n $namespace
```

The same secret is applied to the orchestrator, NER, embedding, UI, and MedCAT pods. Configure the MedGemma subchart separately with `--set medgemma.imagePullSecrets[0].name=$secret_name` when its image also requires private-registry credentials.

## Connecting

### Option 1: Port Forwarding

```bash
kubectl port-forward services/$name-aimsb-medworkup-ui 8080:8080 -n $namespace
```

The UI is available at http://localhost:8080.

### Option 2: HTTPRoute (Gateway Access)

If your cluster has a Gateway API compatible gateway (e.g., kgateway, Istio), enable HTTPRoute creation:

```bash
name="medworkup"
namespace="default"

helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-medworkup \
  -f values.yaml \
  --set http_route.enabled=true \
  | kubectl apply -f - -n $namespace
```

Obtain the URL:

```bash
echo "https://$name-aimsb-medworkup-ui$(kubectl get gtw -A -o jsonpath='{.items[*].spec.listeners[?(@.name=="https")].hostname}' | tr -d \* | awk '{print $1}')/"
```

## Clean Up

```bash
helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-medworkup \
  -f values.yaml \
  | kubectl delete -f - -n $namespace
```

The MedCAT model PersistentVolumeClaim persists after deletion to avoid re-downloading on redeployment. Delete it explicitly for a full clean-up:

```bash
kubectl delete pvc -n $namespace --all
```
