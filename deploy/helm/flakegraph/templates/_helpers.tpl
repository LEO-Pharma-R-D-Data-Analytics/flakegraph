{{/* Return the chart-qualified base resource name. */}}
{{- define "flakegraph.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
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

{{/* Return the stable in-cluster vLLM Service name. */}}
{{- define "flakegraph.modelServingName" -}}
{{- printf "%s-vllm" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Return the OpenAI-compatible endpoint consumed by FlakeGraph workers. */}}
{{- define "flakegraph.modelServingEndpoint" -}}
{{- printf "http://%s:%v/v1" (include "flakegraph.modelServingName" .) .Values.modelServing.service.port -}}
{{- end -}}

{{/* Describe the attention shape that sets KV cost per token, as sidecar JSON. */}}
{{- define "flakegraph.modelServingGeometry" -}}
{{- printf "{\"kv_heads\":%v,\"head_dim\":%v,\"attention_layers\":%v,\"recurrent_layers\":%v,\"recurrent_state_bytes_per_layer\":%v,\"recurrent_state_slots_per_sequence\":%v,\"kv_cache_dtype\":\"%s\",\"weights_bytes\":%v}" .Values.modelServing.sizing.kvHeads .Values.modelServing.sizing.headDim .Values.modelServing.sizing.attentionLayers .Values.modelServing.sizing.recurrentLayers .Values.modelServing.sizing.recurrentStateBytesPerLayer .Values.modelServing.sizing.recurrentStateSlotsPerSequence .Values.modelServing.server.kvCacheDtype (mulf .Values.modelServing.sizing.weightsGiB 1073741824 | int64) -}}
{{- end -}}

{{/* Describe the memory an engine may spend, as sidecar JSON. */}}
{{- define "flakegraph.modelServingDeviceBudget" -}}
{{- printf "{\"device_memory_bytes\":%v,\"gpu_memory_utilization\":%v,\"overhead_bytes\":%v}" (mulf .Values.modelServing.sizing.deviceMemoryGiB 1073741824 | int64) .Values.modelServing.server.gpuMemoryUtilization (mulf .Values.modelServing.sizing.overheadGiB 1073741824 | int64) -}}
{{- end -}}

{{/* Return the one OpenAI-compatible URL every consumer resolves. */}}
{{- define "flakegraph.gatewayName" -}}
{{- printf "%s-litellm" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "flakegraph.gatewayEndpoint" -}}
{{- printf "http://%s:%v/v1" (include "flakegraph.gatewayName" .) .Values.gateway.litellm.service.port -}}
{{- end -}}

{{/* Return the Envoy listener the gateway forwards inference to. */}}
{{- define "flakegraph.inferenceRouterName" -}}
{{- printf "%s-inference-router" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "flakegraph.inferenceRouterEndpoint" -}}
{{- printf "http://%s:%v/v1" (include "flakegraph.inferenceRouterName" .) .Values.gateway.placement.service.port -}}
{{- end -}}

{{/* Resolve an immutable digest or a tag for any component image block. */}}
{{- define "flakegraph.componentImage" -}}
{{- if .digest -}}
{{- printf "%s@%s" .repository .digest -}}
{{- else -}}
{{- printf "%s:%s" .repository .tag -}}
{{- end -}}
{{- end -}}

{{/* The single sign-in gate in front of every routed service. */}}
{{- define "flakegraph.authProxyName" -}}
{{- printf "%s-auth" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* The control plane is a deployed component, not a script on a node. */}}
{{- define "flakegraph.controlPlaneName" -}}
{{- printf "%s-app" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Return the OCR shim Service and the endpoint the pipeline parses through. */}}
{{- define "flakegraph.ocrShimName" -}}
{{- printf "%s-ocr" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "flakegraph.ocrShimEndpoint" -}}
{{- printf "http://%s:%v" (include "flakegraph.ocrShimName" .) .Values.documentParsing.shim.service.port -}}
{{- end -}}

{{- define "flakegraph.mineruName" -}}
{{- printf "%s-mineru" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* The coordinates every FlakeGraph consumer needs to reach the shared planes.

     Defined once because the bootstrap Job validates the very profile the
     workers run: configured differently, it either passes a profile the workers
     cannot execute or fails one they can. Both have drifted from each other
     before, in both directions. */}}
{{- define "flakegraph.consumerEnv" -}}
{{- if .Values.gateway.enabled }}
- name: KG_LLM_ENDPOINT
  value: {{ include "flakegraph.gatewayEndpoint" . | quote }}
- name: KG_LLM_MODEL
  value: {{ .Values.gateway.litellm.aliases.batch | quote }}
- name: KG_LLM_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.gateway.litellm.virtualKeySecret.name }}
      key: {{ .Values.gateway.litellm.virtualKeySecret.batchKey }}
{{- end }}
{{- if .Values.documentParsing.enabled }}
- name: KG_MINERU_API_URL
  value: {{ include "flakegraph.ocrShimEndpoint" . | quote }}
- name: KG_MINERU_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.documentParsing.shim.keySecret.name }}
      key: {{ .Values.documentParsing.shim.keySecret.batchKey }}
{{- end }}
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

{{- define "flakegraph.servingPriorityClassName" -}}
{{- printf "%s-serving" (include "flakegraph.priorityClassPrefix" .) -}}
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

{{/* The labels a selector matches on, and nothing else.

     Selectors are immutable once applied, and a pod template that carries
     helm.sh/chart rolls every pod on every chart bump. Both are avoided by
     keeping this triple apart from the ownership labels above. Takes a dict:
     "root" is the top-level context, "component" the pool being selected. */}}
{{- define "flakegraph.selectorLabels" -}}
app.kubernetes.io/name: {{ .root.Chart.Name }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/* A CPU-utilisation HorizontalPodAutoscaler for one workload.

     Takes a dict: "root" is the top-level context, "name" the workload and the
     HPA, "kind" the workload's kind, "component" its selector component, and
     "autoscaling" the values block with min/max replicas and the CPU target. */}}
{{- define "flakegraph.cpuHpa" -}}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ .name }}
  labels:
    {{- include "flakegraph.labels" .root | nindent 4 }}
    app.kubernetes.io/component: {{ .component }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: {{ .kind }}
    name: {{ .name }}
  minReplicas: {{ .autoscaling.minReplicas }}
  maxReplicas: {{ .autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .autoscaling.targetCpuUtilizationPercentage }}
{{- end -}}

{{/* The gate's middlewares, as the annotation every gated Ingress carries.

     Order matters, and it is the reverse of the intuition: middlewares run
     outermost first, so the errors middleware has to wrap the gate in order to
     see its refusal and turn it into a redirect. Listed the other way round the
     gate still refuses, and the browser is handed a bare 401. Defined once so
     every host behind the gate is gated the same way. */}}
{{- define "flakegraph.authProxyMiddlewares" -}}
{{- printf "%s-%s-errors@kubernetescrd,%s-%s-auth@kubernetescrd" .Release.Namespace (include "flakegraph.authProxyName" .) .Release.Namespace (include "flakegraph.authProxyName" .) -}}
{{- end -}}

{{/* Names produced by the kube-prometheus-stack release monitoring addresses. */}}
{{- define "flakegraph.grafanaServiceName" -}}
{{- printf "%s-grafana" .Values.monitoring.release | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* The ConfigMap of SQL the CloudNativePG exporter runs as metrics. */}}
{{- define "flakegraph.postgresQueriesName" -}}
{{- printf "%s-postgres-queries" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Refuse to render monitoring objects on a cluster that cannot hold them.

     Without the operator's CRDs a ServiceMonitor is not a degraded object, it
     is an install error at apply time, reported per object and after everything
     else has been applied. Failing here names the cause once, first. */}}
{{- define "flakegraph.monitoringRequired" -}}
{{- if not (.Capabilities.APIVersions.Has "monitoring.coreos.com/v1/ServiceMonitor") -}}
{{- fail "monitoring.enabled needs the Prometheus operator CRDs (monitoring.coreos.com/v1). Install the kube-prometheus-stack release from deploy/spark/install-cluster.sh into this namespace first, or pass --api-versions monitoring.coreos.com/v1/ServiceMonitor,monitoring.coreos.com/v1/PodMonitor,monitoring.coreos.com/v1/PrometheusRule when rendering offline." -}}
{{- end -}}
{{- end -}}
