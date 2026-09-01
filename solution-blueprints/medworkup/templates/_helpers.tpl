{{/*
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "medworkup.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
Truncates at 63 chars because some Kubernetes name fields are limited to 63 chars.
*/}}
{{- define "medworkup.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- if ne .Release.Name "release-name" -}}
{{- printf "%s-%s" .Release.Name (include "medworkup.name" .) | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- include "medworkup.name" . | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels applied to all resources.
*/}}
{{- define "medworkup.labels" -}}
app.kubernetes.io/name: {{ include "medworkup.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Image pull secrets block
*/}}
{{- define "medworkup.imagePullSecrets" -}}
{{- if .Values.imagePullSecrets }}
imagePullSecrets:
{{- toYaml .Values.imagePullSecrets | nindent 2 }}
{{- end }}
{{- end -}}

{{/*
Full image reference for a custom service.
Usage: include "medworkup.image" (dict "root" . "svc" .Values.images.ner)
*/}}
{{- define "medworkup.image" -}}
{{- $registry := .root.Values.images.registry -}}
{{- $repo := .svc.repository -}}
{{- $tag := tpl (.svc.tag | default "latest") .root -}}
{{- if $registry -}}
{{- printf "%s%s:%s" $registry $repo $tag -}}
{{- else -}}
{{- printf "%s:%s" $repo $tag -}}
{{- end -}}
{{- end -}}

{{/*
Orchestrator in-cluster URL.
*/}}
{{- define "medworkup.orchestratorUrl" -}}
http://{{ include "medworkup.fullname" . }}-orchestrator:{{ .Values.ports.orchestrator }}
{{- end -}}

{{/*
NER service in-cluster URL.
*/}}
{{- define "medworkup.nerUrl" -}}
http://{{ include "medworkup.fullname" . }}-ner:{{ .Values.ports.ner }}
{{- end -}}

{{/*
Embedding service in-cluster URL.
*/}}
{{- define "medworkup.embeddingUrl" -}}
http://{{ include "medworkup.fullname" . }}-embedding:{{ .Values.ports.embedding }}
{{- end -}}

{{/*
MedCAT service in-cluster URL.
*/}}
{{- define "medworkup.medcatUrl" -}}
http://{{ include "medworkup.fullname" . }}-medcat:{{ .Values.ports.medcat }}
{{- end -}}
