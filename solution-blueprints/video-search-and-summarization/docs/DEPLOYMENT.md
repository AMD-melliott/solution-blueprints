<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Video Search and Summarization Deployment Guide

Solution Blueprints are provided as Helm Charts.

The recommended approach to deploy them is to pipe the output of `helm template` to `kubectl apply -f -`.
We don't recommend `helm install`, which by default uses a Secret to keep track of the related resources.
This does not work well with Enterprise clusters that often have limitations on the kinds of resources that
regular users are allowed to create.

This blueprint is designed to run on **AMD Instinct** GPUs.

To deploy the blueprint, run the following command:

```bash
name="my-deployment"
namespace="my-namespace"
helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-video-search-and-summarization \
  | kubectl apply -f - -n $namespace
```

> **Known issue (Helm 4.2.1+):** Helm 4.2.1 and newer leak `Pulled:`/`Digest:` metadata to stdout ([helm#32215](https://github.com/helm/helm/issues/32215)), which breaks the piped `helm template … | kubectl apply -f -`. Until the fix ships, either use Helm **3.16 – 4.2.0**, or split the pull and template steps, e.g.:
>
> ```bash
> helm pull oci://registry-1.docker.io/amdenterpriseai/aimsb-video-search-and-summarization --untar
> helm template $name ./aimsb-video-search-and-summarization \
>   # …same flags as the piped command above…
>   | kubectl apply -f - -n $namespace
> ```

## Using an existing deployment or external LLM

By default, the LLM is deployed by the helm chart. If you already have a compatible AIM deployed, you can use that instead and reuse resources.

To use an existing deployment or external LLM for summaries and Q&A, set the value `llm.existingService` to that endpoint. Then, any other values you pass in the `llm` mapping are simply ignored, and your existing service is used instead. You should use the Kubernetes Service name, or if the service is in a different namespace, you can use the long form `<SERVICENAME>.<NAMESPACE>.svc.cluster.local:<SERVICEPORT>`. If needed, you can pass a whole URL.

Full example command:

```bash
name="my-deployment"
namespace="my-namespace"
servicename="aim-llm-my-model-123456"
helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-video-search-and-summarization \
  --set llm.existingService=$servicename \
  | kubectl apply -f - -n $namespace
```

### API Key and Model Configuration for External LLMs

You can independently configure API authentication and an explicit model name for the LLM backend:

- `llm.apiKey`, `llm.model`

If a model name is not provided, the chart queries the backend `/v1/models` endpoint and uses the first available model.

Example command:

```bash
name="my-deployment"
namespace="my-namespace"
llm_api_url="https://llm-api.example.com"
llm_api_key="<LLM_API_KEY>"
llm_model="Qwen/Qwen3-32B"

helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-video-search-and-summarization \
  --set llm.existingService=$llm_api_url \
  --set llm.apiKey=$llm_api_key \
  --set llm.model=$llm_model \
  | kubectl apply -f - -n $namespace
```

## Default AIM image and GPU compatibility

By default, the chart deploys these images:

- `llm.image=amdenterpriseai/aim-qwen-qwen3-32b:0.11.1` (summaries and grounded answers)
- `vlm.image=vllm/vllm-openai-rocm:v0.22.0` (vision-language captioning, `Qwen/Qwen3-VL-8B-Instruct`)
- `embedText.image` / `embedImage.image` `=vllm/vllm-openai-rocm:v0.22.0` (BGE-M3 text and CLIP image embeddings)

On newer GPUs, these images may not be the best match and can fail to start or run sub-optimally.
To choose a newer AIM or a different LLM, override `llm.image`; the vision-language and embedding images can be overridden similarly. See the [catalog of available AIMs](https://enterprise-ai.docs.amd.com/en/latest/aims/catalog/models.html) for options.

Example:

```bash
name="my-deployment"
namespace="my-namespace"
helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-video-search-and-summarization \
  --set llm.image=amdenterpriseai/aim-qwen-qwen3-32b:<NEWER_TAG> \
  | kubectl apply -f - -n $namespace
```

## Connecting

### Option 1: Port Forwarding

To connect to the UI, port-forward port 8090. The UI will then be available at <http://localhost:8090>.

```bash
kubectl port-forward services/$name-aimsb-video-search-and-summarization 8090:80 -n $namespace
```

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
helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-video-search-and-summarization \
  --set http_route.enabled=true \
  | kubectl apply -f - -n $namespace
```

**Obtaining the URL:**

The URL to access the blueprint via HTTPRoute is formed by the service name and the hostname of the gateway. Use this command to produce the URL by querying the hostname from the cluster:

```bash
echo "https://$name-aimsb-video-search-and-summarization$(kubectl get gtw -n envoy-gateway-system https -o jsonpath='{.spec.listeners[?(@.name=="https")].hostname}' | tr -d \*)/"
```

## Using Video Search and Summarization

Upload a video from the UI. It is automatically split into segments, captioned by the vision-language model, and indexed (caption text and keyframe images) for retrieval. Once indexing completes you can:

- **Ask** questions in natural language and get answers cited to the exact segment and timestamp.
- **Analyze** the video to produce a timestamped summary (overview, timeline, key observations).
- **Track** objects to draw boxes and IDs on a tracking-overlay copy of the video and make the tracked objects queryable in Ask.

The vision-language model and LLM can take several minutes to load on first start; the app remains available while they come up.

## Clean Up

When you are finished, remove the deployed resources:

```bash
helm template $name oci://registry-1.docker.io/amdenterpriseai/aimsb-video-search-and-summarization \
  | kubectl delete -f - -n $namespace
```
