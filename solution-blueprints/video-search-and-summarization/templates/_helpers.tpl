# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

{{/*
Create a default fully qualified app name.
*/}}
{{- define "aimsb-video-search-and-summarization.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart name and version as used by the chart label.
*/}}
{{- define "aimsb-video-search-and-summarization.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "aimsb-video-search-and-summarization.labels" -}}
helm.sh/chart: {{ include "aimsb-video-search-and-summarization.chart" . }}
{{ include "aimsb-video-search-and-summarization.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels used by the main app deployment and service.
*/}}
{{- define "aimsb-video-search-and-summarization.selectorLabels" -}}
app: {{ include "aimsb-video-search-and-summarization.fullname" . }}
{{- end }}

{{/*
Names of the in-chart auxiliary services (no aimchart exists for these yet).
*/}}
{{- define "aimsb-video-search-and-summarization.vlm.name" -}}{{ include "aimsb-video-search-and-summarization.fullname" . }}-vlm{{- end }}
{{- define "aimsb-video-search-and-summarization.track.name" -}}{{ include "aimsb-video-search-and-summarization.fullname" . }}-track{{- end }}

{{/*
Embedding service base URLs, resolved through the aimchart-embedding subchart's url helper so
the service names stay in sync with the rendered subchart deployments. The app appends
/v1/embeddings (and /health for readiness waits).
*/}}
{{- define "aimsb-video-search-and-summarization.embedText.url" -}}
{{- $ctx := dict "Values" (merge (dict) .Values.embedText) "Release" .Release "Chart" (dict "Name" "embed-text") -}}
{{- include "aim-embedding.url" $ctx -}}
{{- end }}
{{- define "aimsb-video-search-and-summarization.embedImage.url" -}}
{{- $ctx := dict "Values" (merge (dict) .Values.embedImage) "Release" .Release "Chart" (dict "Name" "embed-image") -}}
{{- include "aim-embedding.url" $ctx -}}
{{- end }}

{{/*
Environment variables for the application container.
Wires the LLM (aimchart-llm) and the vector DB (aimchart-chromadb) via their URL
template functions, and the in-chart VLM / embedding / tracking services by service name.
*/}}
{{- define "aimsb-video-search-and-summarization.container.env" -}}
- name: LLM_BASE_URL
  {{ $llm := dict "Values" (merge (dict) .Values.llm) "Release" .Release "Chart" (dict "Name" "llm") }}
  value: {{ include "aimchart-llm.url" $llm | quote }}
{{- if .Values.llm.model }}
- name: LLM_MODEL
  value: {{ .Values.llm.model | quote }}
{{- end }}
{{- if .Values.llm.apiKeySecretRef }}
- name: OPENAI_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.llm.apiKeySecretRef.name }}
      key: {{ .Values.llm.apiKeySecretRef.key }}
{{- else }}
- name: OPENAI_API_KEY
  value: {{ default "EMPTY" .Values.llm.apiKey | quote }}
{{- end }}
- name: CHROMA_URL
  {{ $cdb := dict "Values" (merge (dict) .Values.chromadb) "Release" .Release "Chart" (dict "Name" "chromadb") }}
  value: {{ include "aim-chromadb.url" $cdb | quote }}
- name: VLM_BASE_URL
  value: "http://{{ include "aimsb-video-search-and-summarization.vlm.name" . }}:{{ .Values.vlm.port }}/v1"
- name: VLM_MODEL
  value: {{ .Values.vlm.model | quote }}
- name: EMBED_TEXT_URL
  value: {{ include "aimsb-video-search-and-summarization.embedText.url" . | quote }}
- name: EMBED_TEXT_MODEL
  value: {{ .Values.embedText.model | quote }}
- name: EMBED_IMAGE_URL
  value: {{ include "aimsb-video-search-and-summarization.embedImage.url" . | quote }}
- name: EMBED_IMAGE_MODEL
  value: {{ .Values.embedImage.model | quote }}
- name: TRACK_URL
  value: "http://{{ include "aimsb-video-search-and-summarization.track.name" . }}:{{ .Values.track.port }}"
- name: DATA_DIR
  value: "/workload/data"
- name: UI_DIR
  value: "/workload/mount/src/ui"
{{- range $key, $value := .Values.env_vars }}
- name: {{ $key }}
  value: {{ tpl $value $ | quote }}
{{- end }}
{{- end }}

{{/*
Container resources for the app.
*/}}
{{- define "aimsb-video-search-and-summarization.container.resources" -}}
{{- if .Values.resources }}
{{- toYaml .Values.resources }}
{{- else }}
{}
{{- end }}
{{- end }}

{{/*
Volume mounts: shared memory, ephemeral working dir, and the source code from the ConfigMap.
*/}}
{{- define "aimsb-video-search-and-summarization.container.volumeMounts" -}}
- name: dshm
  mountPath: /dev/shm
- name: ephemeral-storage
  mountPath: /workload
{{- range $path, $_ := .Files.Glob "src/**" }}
- name: workload-mount
  mountPath: /workload/mount/{{ $path }}
  subPath: {{ $path | replace "/" "_" }}
{{- end }}
{{- end }}

{{/*
Volumes backing the mounts above.
*/}}
{{- define "aimsb-video-search-and-summarization.container.volumes" -}}
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: {{ .Values.storage.dshm.sizeLimit }}
- name: ephemeral-storage
  emptyDir:
    sizeLimit: {{ .Values.storage.ephemeral.quantity }}
- name: workload-mount
  configMap:
    name: {{ include "aimsb-video-search-and-summarization.fullname" . }}
{{- end }}

{{/*
Entrypoint: install ffmpeg + Python deps from the mounted source, then run the app.
*/}}
{{- define "aimsb-video-search-and-summarization.entrypoint" -}}
set -euo pipefail
echo "Installing ffmpeg and Python dependencies..."
apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg > /dev/null
pip install --no-cache-dir -r /workload/mount/src/requirements.txt
mkdir -p /workload/data
echo "Starting Uvicorn server..."
cd /workload/mount/src
uvicorn app:app --host 0.0.0.0 --port {{ .Values.deployment.ports.http }} --root-path /
{{- end }}
