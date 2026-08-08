{{/* Return the chart-qualified base resource name. */}}
{{- define "flakegraph.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Resolve the FlakeGraph Spark driver/executor image. */}}
{{- define "flakegraph.sparkImage" -}}
{{- if .Values.spark.image.digest -}}
{{- printf "%s@%s" .Values.spark.image.repository .Values.spark.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.spark.image.repository .Values.spark.image.tag -}}
{{- end -}}
{{- end -}}

{{/* Select an explicitly named or chart-managed service account. */}}
{{- define "flakegraph.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "flakegraph.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* Select a release-scoped or explicitly external Spark service account. */}}
{{- define "flakegraph.sparkServiceAccountName" -}}
{{- $name := coalesce .Values.spark.serviceAccount.name .Values.spark.serviceAccountName -}}
{{- if .Values.spark.serviceAccount.create -}}
{{- default (printf "%s-spark" (include "flakegraph.fullname" . | trunc 57 | trimSuffix "-")) $name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- default "default" $name -}}
{{- end -}}
{{- end -}}

{{/* Keep chart-owned Spark RBAC distinct across releases. */}}
{{- define "flakegraph.sparkRbacName" -}}
{{- printf "%s-spark" (include "flakegraph.fullname" . | trunc 57 | trimSuffix "-") -}}
{{- end -}}

{{/* Resolve either an immutable digest or a conventional repository tag. */}}
{{- define "flakegraph.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) -}}
{{- end -}}
{{- end -}}

{{/* Resolve the immutable local model-server image when model serving is enabled. */}}
{{- define "flakegraph.modelServingImage" -}}
{{- if .Values.modelServing.image.digest -}}
{{- printf "%s@%s" .Values.modelServing.image.repository .Values.modelServing.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.modelServing.image.repository .Values.modelServing.image.tag -}}
{{- end -}}
{{- end -}}

{{/* Return the stable in-cluster vLLM Service name. */}}
{{- define "flakegraph.modelServingName" -}}
{{- printf "%s-vllm" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Return the OpenAI-compatible endpoint consumed by FlakeGraph workers. */}}
{{- define "flakegraph.modelServingEndpoint" -}}
{{- printf "http://%s:%v/v1" (include "flakegraph.modelServingName" .) .Values.modelServing.service.port -}}
{{- end -}}

{{/* Return the Service name that keeps inference traffic on the caller's node. */}}
{{- define "flakegraph.localModelServingName" -}}
{{- printf "%s-local" (include "flakegraph.modelServingName" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Return the node-local OpenAI-compatible endpoint used by colocated workers. */}}
{{- define "flakegraph.localModelServingEndpoint" -}}
{{- printf "http://%s:%v/v1" (include "flakegraph.localModelServingName" .) .Values.modelServing.service.port -}}
{{- end -}}

{{/* Use an existing config map when configuration is managed externally. */}}
{{- define "flakegraph.configMapName" -}}
{{- default (printf "%s-config" (include "flakegraph.fullname" .)) .Values.config.existingConfigMap -}}
{{- end -}}

{{/* Use an existing ontology ConfigMap or the chart-owned content object. */}}
{{- define "flakegraph.ontologyConfigMapName" -}}
{{- default (printf "%s-ontology" (include "flakegraph.fullname" .)) .Values.ontology.existingConfigMap -}}
{{- end -}}

{{/* Cluster-scoped scheduling names include the namespace to avoid release collisions. */}}
{{- define "flakegraph.priorityClassPrefix" -}}
{{- printf "%s-%s" .Release.Namespace (include "flakegraph.fullname" .) | trunc 54 | trimSuffix "-" -}}
{{- end -}}

{{- define "flakegraph.workerPriorityClassName" -}}
{{- printf "%s-worker" (include "flakegraph.priorityClassPrefix" .) -}}
{{- end -}}

{{- define "flakegraph.sparkPriorityClassName" -}}
{{- printf "%s-spark" (include "flakegraph.priorityClassPrefix" .) -}}
{{- end -}}

{{- define "flakegraph.modelPriorityClassName" -}}
{{- printf "%s-model" (include "flakegraph.priorityClassPrefix" .) -}}
{{- end -}}

{{/* Common ownership labels used by every namespaced object. */}}
{{- define "flakegraph.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
{{- end -}}
