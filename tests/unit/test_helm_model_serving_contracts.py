"""Protect the Kubernetes-managed local model-serving deployment contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

_CHART = Path("deploy/helm/flakegraph")
_VALUES = _CHART / "values.yaml"
_SCHEMA = _CHART / "values.schema.json"
_MODEL_TEMPLATE = _CHART / "templates/model-serving.yaml"
_WORKER_TEMPLATE = _CHART / "templates/workers.yaml"
_AUTOSCALING_TEMPLATE = _CHART / "templates/queue-autoscaling.yaml"
_DATABASE_BOOTSTRAP_TEMPLATE = _CHART / "templates/database-bootstrap-job.yaml"
_NOTES_TEMPLATE = _CHART / "templates/NOTES.txt"
_CLOUDNATIVEPG_TEMPLATE = _CHART / "templates/cloudnativepg.yaml"
_PRIORITY_TEMPLATE = _CHART / "templates/priorityclasses.yaml"
_SPARK_RBAC_TEMPLATE = _CHART / "templates/spark-rbac.yaml"
_PUBLIC_EXAMPLE = Path("deploy/examples/k3s-spark-values.yaml")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def test_model_serving_defaults_are_pinned_and_resource_bounded() -> None:
    """Keep opt-in inference reproducible instead of following moving artifacts."""

    values = _load_yaml(_VALUES)["modelServing"]

    assert values["enabled"] is False
    assert values["runtime"] == "vllm"
    assert values["image"]["tag"] == "26.05.post1-py3"
    assert (
        values["image"]["digest"]
        == "sha256:94e21552f644e0c1627464ba89d2f7a4ce7442e196f72afa0bb5d7fba23cbb03"
    )
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", values["image"]["digest"])
    assert values["model"]["name"] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert _COMMIT.fullmatch(values["model"]["revision"])
    assert values["server"]["maxNumSeqs"] == 4
    assert values["server"]["maxNumBatchedTokens"] == 32768
    assert values["server"]["gpuMemoryUtilization"] == 0.5
    assert values["huggingFaceTokenSecret"]["name"] == ""
    assert values["persistence"]["enabled"] is True
    assert values["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert values["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert values["service"]["internalTrafficPolicy"] == "Cluster"


def test_model_serving_template_owns_the_complete_model_lifecycle() -> None:
    """Require storage, pinned model loading, probes, spreading, and service discovery."""

    template = _MODEL_TEMPLATE.read_text(encoding="utf-8")

    required_fragments = [
        "kind: StatefulSet",
        "kind: Service",
        "volumeClaimTemplates:",
        "requiredDuringSchedulingIgnoredDuringExecution:",
        "--revision",
        "--max-num-batched-tokens",
        "--speculative-config",
        "VLLM_MARLIN_USE_ATOMIC_ADD",
        "huggingFaceTokenSecret",
        "fastsafetensors",
        "startupProbe:",
        "readinessProbe:",
        "livenessProbe:",
        "internalTrafficPolicy:",
        'include "flakegraph.localModelServingName"',
    ]
    for fragment in required_fragments:
        assert fragment in template


def test_spark_executor_spreading_degrades_gracefully() -> None:
    """Prefer fleet-wide placement without deadlocking finalization on node loss."""

    template = (_CHART / "templates/spark-rbac.yaml").read_text(encoding="utf-8")

    assert "preferredDuringSchedulingIgnoredDuringExecution:" in template
    assert "weight: 100" in template
    assert "podAffinityTerm:" in template
    assert "spark-role: executor" in template
    assert ".Values.spark.executorPodSecurityContext" in template
    assert ".Values.spark.executorContainerSecurityContext" in template
    values = _load_yaml(_VALUES)
    assert values["spark"]["executorPodSecurityContext"]["runAsUser"] == 185
    assert values["spark"]["executorContainerSecurityContext"]["allowPrivilegeEscalation"] is False


def test_workers_receive_only_the_chart_managed_llm_endpoint() -> None:
    """Wire chat to vLLM while preserving the independent embedding provider."""

    template = _WORKER_TEMPLATE.read_text(encoding="utf-8")

    assert "if $root.Values.modelServing.enabled" in template
    assert "KG_LLM_ENDPOINT" in template
    assert "KG_LLM_MODEL" in template
    assert "KG_EMBED_ENDPOINT" not in template
    assert "KG_EMBED_MODEL" not in template
    assert 'include "flakegraph.modelServingEndpoint"' in template
    assert 'include "flakegraph.localModelServingEndpoint"' in template
    assert "if $pool.localModelServing" in template
    assert "topologySpreadConstraints:" in template


def test_local_model_workers_are_colocated_and_invalid_profiles_fail_rendering() -> None:
    """Make a node-local inference endpoint unreachable only by invalid chart input."""

    template = _WORKER_TEMPLATE.read_text(encoding="utf-8")

    assert "localModelServing requires modelServing.enabled=true" in template
    assert "affinity.podAffinity is managed by localModelServing" in template
    assert "requiredDuringSchedulingIgnoredDuringExecution:" in template
    assert "app.kubernetes.io/component: model-serving" in template
    assert "topologyKey: {{ $root.Values.modelServing.topologyKey }}" in template


def test_provider_secret_import_is_an_explicit_credential_allowlist() -> None:
    """Prevent unrelated Secret keys from silently replacing reviewed provider settings."""

    values = _load_yaml(_VALUES)
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    workers = _WORKER_TEMPLATE.read_text(encoding="utf-8")
    bootstrap = _DATABASE_BOOTSTRAP_TEMPLATE.read_text(encoding="utf-8")

    assert values["providerSecret"]["env"] == [
        {"name": "KG_LLM_API_KEY", "key": "KG_LLM_API_KEY", "optional": True},
        {"name": "KG_EMBED_API_KEY", "key": "KG_EMBED_API_KEY", "optional": True},
        {
            "name": "KG_SNOWFLAKE_PASSWORD",
            "key": "KG_SNOWFLAKE_PASSWORD",
            "optional": True,
        },
        {
            "name": "KG_SNOWFLAKE_OAUTH_TOKEN",
            "key": "KG_SNOWFLAKE_OAUTH_TOKEN",
            "optional": True,
        },
    ]
    assert "env" in schema["properties"]["providerSecret"]["required"]
    assert "envFrom:" not in workers
    assert "envFrom:" not in bootstrap
    assert "range $mapping := $root.Values.providerSecret.env" in workers
    assert "range $mapping := .Values.providerSecret.env" in bootstrap


def test_ontology_is_a_portable_chart_managed_deployment_input() -> None:
    """Mount ontology content without baking repository datasets into worker images."""

    values = _load_yaml(_VALUES)
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    configmap = (_CHART / "templates/configmap.yaml").read_text(encoding="utf-8")
    workers = _WORKER_TEMPLATE.read_text(encoding="utf-8")
    bootstrap = _DATABASE_BOOTSTRAP_TEMPLATE.read_text(encoding="utf-8")

    assert values["ontology"] == {
        "existingConfigMap": "",
        "key": "ontology.yaml",
        "content": "",
    }
    assert "ontology" in schema["required"]
    assert 'include "flakegraph.ontologyConfigMapName"' in configmap
    assert "KG_ONTOLOGY_PROFILE" in workers
    assert "KG_ONTOLOGY_PROFILE" in bootstrap
    assert "checksum/ontology" in workers


def test_worker_stage_names_follow_the_distributed_dag() -> None:
    """Keep typed Helm values aligned with role-specific worker pools."""

    values = _load_yaml(_VALUES)
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    stage_enum = schema["$defs"]["worker"]["properties"]["stages"]["items"]["enum"]
    chart_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(_CHART.rglob("*")) if path.is_file()
    )

    assert values["workers"]["prepare"]["stages"] == ["prepare_document"]
    assert values["workers"]["extract"]["stages"] == [
        "extract_document_context",
        "extract_entity_window",
        "compact_entity_inventory",
        "extract_relation_window",
        "compact_document",
    ]
    assert values["workers"]["finalize"]["stages"] == ["finalize_graph"]
    assert stage_enum == [
        "prepare_document",
        "extract_document_context",
        "extract_entity_window",
        "compact_entity_inventory",
        "extract_relation_window",
        "compact_document",
        "finalize_graph",
    ]
    assert "extract_window" not in chart_text


def test_mutable_worker_image_tags_are_refreshed_on_rollout() -> None:
    """Prevent node caches from retaining an older binary after a chart upgrade."""

    values = _load_yaml(_VALUES)

    assert values["image"]["pullPolicy"] == "Always"
    assert values["spark"]["image"]["pullPolicy"] == "Always"
    assert values["modelServing"]["image"]["pullPolicy"] == "IfNotPresent"
    assert values["modelServing"]["image"]["digest"].startswith("sha256:")


def test_workers_seed_preloaded_models_into_their_writable_cache() -> None:
    """Prevent the read-only runtime cache mount from hiding baked checkpoints."""

    template = _WORKER_TEMPLATE.read_text(encoding="utf-8")

    assert "initContainers:" in template
    assert "name: seed-provider-cache" in template
    # The init container runs as an unprivileged image user. Recursive copy is
    # sufficient; archive mode would also preserve timestamps/ownership and can
    # fail against an EmptyDir prepared by Kubernetes securityContext settings.
    assert "cp -R" in template
    assert "cp -a" not in template
    assert "/home/kgprocessor/.cache/." in template
    assert "/home/spark/.cache/." in template
    assert "mountPath: /home/kgprocessor/.cache" in template
    assert "value: /cache/huggingface" in template
    assert "value: /cache/sentence_transformers" in template
    seed_resources = template.split("name: seed-provider-cache", maxsplit=1)[1].split(
        "securityContext:", maxsplit=1
    )[0]
    assert "memory: 1Gi" in seed_resources


def test_spark_workers_reserve_memory_for_python_provider_processes() -> None:
    """Keep executor heap and non-heap memory explicit across Helm boundaries."""

    values = _load_yaml(_VALUES)
    template = _WORKER_TEMPLATE.read_text(encoding="utf-8")
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))

    assert values["spark"]["executorMemory"] == "8g"
    assert values["spark"]["executorMemoryOverhead"] == "8g"
    assert "KG_DISTRIBUTED_SPARK_EXECUTOR_MEMORY_OVERHEAD" in template
    assert "executorMemoryOverhead" in schema["properties"]["spark"]["required"]


def test_schema_and_public_fleet_example_expose_local_model_serving() -> None:
    """Keep strict values validation and the public multi-node example synchronized."""

    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    example = _load_yaml(_PUBLIC_EXAMPLE)

    assert "modelServing" in schema["required"]
    assert schema["properties"]["modelServing"]["additionalProperties"] is False
    assert example["modelServing"] == {
        "enabled": True,
        "replicas": 4,
        "nodeSelector": {"flakegraph.io/node-class": "nvidia-spark"},
    }
    prepare = example["workers"]["prepare"]
    assert prepare["replicas"] == 16
    assert prepare["autoscaling"]["maxReplicas"] == 16
    assert prepare["topologySpreadConstraints"][0]["maxSkew"] == 1
    assert prepare["topologySpreadConstraints"][0]["matchLabelKeys"] == ["pod-template-hash"]
    extract = example["workers"]["extract"]
    assert extract["replicas"] == 8
    assert extract["autoscaling"]["maxReplicas"] == 8
    assert extract["localModelServing"] is True
    assert extract["nodeSelector"] == {"flakegraph.io/node-class": "nvidia-spark"}
    assert extract["topologySpreadConstraints"][0]["maxSkew"] == 1
    assert extract["topologySpreadConstraints"][0]["matchLabelKeys"] == ["pod-template-hash"]


def test_worker_pools_autoscale_from_dependency_aware_postgres_demand() -> None:
    """Release drained worker resources without scaling away active task leases."""

    values = _load_yaml(_VALUES)
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    autoscaling = _AUTOSCALING_TEMPLATE.read_text(encoding="utf-8")
    workers = _WORKER_TEMPLATE.read_text(encoding="utf-8")
    postgres = Path("src/kg_processor/adapters/distributed/postgres.py").read_text(encoding="utf-8")

    assert values["autoscaling"]["enabled"] is True
    assert values["autoscaling"]["pollingIntervalSeconds"] == 5
    assert values["autoscaling"]["cooldownPeriodSeconds"] == 30
    assert values["workers"]["prepare"]["autoscaling"]["minReplicas"] == 0
    assert values["workers"]["extract"]["autoscaling"]["maxReplicas"] == 32
    assert values["workers"]["finalize"]["autoscaling"]["maxReplicas"] == 1
    assert values["distributed"]["leaseSeconds"] == 300
    assert "autoscaling" in schema["required"]
    assert "autoscaling" in schema["$defs"]["worker"]["required"]
    assert "kind: ScaledObject" in autoscaling
    assert "type: postgresql" in autoscaling
    assert "FROM flakegraph_worker_demand" in autoscaling
    assert "SUM(LEAST(" in autoscaling
    assert "connectionFromEnv: KG_DISTRIBUTED_DATABASE_URL" in autoscaling
    assert "passwordFromEnv: KG_DISTRIBUTED_DATABASE_PASSWORD" in autoscaling
    assert "restoreToOriginalReplicaCount: true" in autoscaling
    assert 'lookup "apps/v1" "Deployment"' in workers
    assert "replicas: {{ $pool.autoscaling.minReplicas }}" in workers
    assert "if not $existingDeployment" in workers
    assert "CREATE OR REPLACE VIEW flakegraph_worker_demand" in postgres
    assert "task.status = 'running'" in postgres
    assert "remaining_dependencies" in postgres
    assert "FROM flakegraph_publication AS publication" in postgres
    assert "SELECT 'finalize_graph' AS stage" in postgres


def test_database_schema_is_bootstrapped_before_a_helm_release_is_ready() -> None:
    """Make KEDA's worker-demand view deterministic on fresh installs and upgrades."""

    values = _load_yaml(_VALUES)
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    template = _DATABASE_BOOTSTRAP_TEMPLATE.read_text(encoding="utf-8")

    assert values["database"]["bootstrap"]["enabled"] is True
    assert values["database"]["bootstrap"]["activeDeadlineSeconds"] >= 600
    assert "bootstrap" in schema["properties"]["database"]["required"]
    assert "kind: Job" in template
    assert "helm.sh/hook: post-install,post-upgrade" in template
    assert "helm.sh/hook-delete-policy: before-hook-creation,hook-succeeded" in template
    assert "distributed" in template
    assert "init" in template
    assert "preflight --deployment-worker" in template
    assert "KG_DISTRIBUTED_DATABASE_URL" in template


def test_operator_notes_describe_the_selected_database_bootstrap_mode() -> None:
    """Helm output must not claim a disabled bootstrap hook completed."""

    notes = _NOTES_TEMPLATE.read_text(encoding="utf-8")

    assert "if .Values.database.bootstrap.enabled" in notes
    assert "The database bootstrap hook completed" in notes
    assert "Database bootstrap was disabled" in notes


def test_bundled_database_has_capacity_for_documented_fleet_workers() -> None:
    """Reserve database sessions for workers, autoscaling, and operations."""

    values = _load_yaml(_VALUES)
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    template = _CLOUDNATIVEPG_TEMPLATE.read_text(encoding="utf-8")
    cloud_native_pg = values["database"]["cloudNativePG"]

    assert cloud_native_pg["maxConnections"] >= 300
    assert (
        "maxConnections"
        in schema["properties"]["database"]["properties"]["cloudNativePG"]["required"]
    )
    assert "max_connections:" in template


def test_scheduling_priorities_preserve_models_and_release_workers_for_spark() -> None:
    """Make the extraction-to-finalization handoff resilient to autoscaler delay."""

    values = _load_yaml(_VALUES)
    priorities = values["scheduling"]["priorityClasses"]
    priority_template = _PRIORITY_TEMPLATE.read_text(encoding="utf-8")
    worker_template = _WORKER_TEMPLATE.read_text(encoding="utf-8")
    spark_template = _SPARK_RBAC_TEMPLATE.read_text(encoding="utf-8")
    model_template = _MODEL_TEMPLATE.read_text(encoding="utf-8")

    assert priorities["enabled"] is True
    assert priorities["workerValue"] < priorities["sparkValue"] < priorities["modelValue"]
    assert priority_template.count("kind: PriorityClass") == 3
    assert 'include "flakegraph.workerPriorityClassName"' in worker_template
    assert 'include "flakegraph.sparkPriorityClassName"' in spark_template
    assert 'include "flakegraph.modelPriorityClassName"' in model_template
    assert values["terminationGracePeriodSeconds"] > 3600


def test_public_fleet_example_enables_object_backed_spark_finalization() -> None:
    """Keep the scalable data plane visible in the generic deployment example."""

    example = _load_yaml(_PUBLIC_EXAMPLE)

    assert example["artifactStorage"] == {
        "uri": "s3://flakegraph-artifacts",
        "endpointUrl": "https://object-storage.example.com",
        "existingSecret": "flakegraph-artifacts",
    }
    assert example["spark"]["enabled"] is True
    assert example["spark"]["executorInstances"] == 4
    assert example["spark"]["executorMemoryOverhead"] == "8g"
    assert example["spark"]["nodeSelector"] == {"flakegraph.io/node-class": "nvidia-spark"}


def test_spark_role_allows_native_executor_creation_and_cleanup() -> None:
    """Permit Spark to remove executor collections when a driver terminates."""

    template = _SPARK_RBAC_TEMPLATE.read_text(encoding="utf-8")

    assert 'resources: ["pods", "services", "configmaps", "persistentvolumeclaims"]' in template
    assert '"create"' in template
    assert '"delete"' in template
    assert '"deletecollection"' in template


def test_object_storage_cannot_silently_select_an_unavailable_spark_runtime() -> None:
    """Reject a fleet configuration that would fail only after extraction ends."""

    template = _WORKER_TEMPLATE.read_text(encoding="utf-8")

    assert "and .Values.artifactStorage.uri (not .Values.spark.enabled)" in template
    assert "spark.enabled must be true when artifactStorage.uri is configured" in template


def test_model_serving_documentation_is_consolidated_into_the_fleet_guide() -> None:
    """Keep model operations in the generic fleet guide without a parallel host guide."""

    fleet_guide = Path("docs/kubernetes-fleet.md").read_text(encoding="utf-8")
    top_level_docs = sorted(path.name for path in Path("docs").glob("*.md"))

    assert top_level_docs == [
        "algorithm.md",
        "architecture.md",
        "kubernetes-fleet.md",
        "snowflake-setup.md",
    ]
    assert "## Local Model Serving" in fleet_guide
    assert "KEDA 2.20.1" in fleet_guide
    assert "kind: ScaledObject" in _AUTOSCALING_TEMPLATE.read_text(encoding="utf-8")
    assert "autoscaling:" in fleet_guide
    assert "No separate inference router is required" in fleet_guide
    assert "--set-file config.content=deploy/private/fleet-config.yaml" in fleet_guide


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one repository-owned YAML mapping for contract assertions."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
