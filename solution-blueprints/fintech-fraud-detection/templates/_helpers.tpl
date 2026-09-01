# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

{{/*
Base name: <release>-aimsb-fintech-fraud-detection
*/}}
{{- define "fintech.complexName" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}-aimsb-fintech-fraud-detection
{{- end -}}

{{/*
Image pull secrets block
*/}}
{{- define "fintech.imagePullSecrets" -}}
{{- if .Values.imagePullSecrets }}
imagePullSecrets:
  {{- range .Values.imagePullSecrets }}
  - name: {{ .name | quote }}
  {{- end }}
{{- end }}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "fintech.labels" -}}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{/*
In-cluster service URLs
*/}}
{{- define "fintech.backendUrl" -}}
http://{{ include "fintech.complexName" . }}-backend:{{ .Values.ports.backend }}
{{- end -}}

{{- define "fintech.middlewareUrl" -}}
http://{{ include "fintech.complexName" . }}-middleware:{{ .Values.ports.middleware }}
{{- end -}}

{{- define "fintech.gnnUrl" -}}
http://{{ include "fintech.complexName" . }}-gnn:{{ .Values.ports.gnn }}
{{- end -}}

{{- define "fintech.xgbUrl" -}}
http://{{ include "fintech.complexName" . }}-xgb:{{ .Values.ports.xgb }}
{{- end -}}


{{- define "release.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "release.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- if ne .Release.Name "release-name" -}}
{{- include "release.name" . }}-{{ .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- include "release.name" . | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}
