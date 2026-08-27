"""Protect the Kubernetes-managed serving and document-parsing contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from kg_processor.serving.sizing import (
    BYTES_PER_GIB,
    DeviceBudget,
    ModelGeometry,
    compute_sizing,
)

_CHART = Path("deploy/helm/flakegraph")
_VALUES = _CHART / "values.yaml"
_SCHEMA = _CHART / "values.schema.json"
_MODEL_TEMPLATE = _CHART / "templates/model-serving.yaml"
_NETWORK_POLICY_TEMPLATE = _CHART / "templates/model-serving-networkpolicy.yaml"
_LITELLM_TEMPLATE = _CHART / "templates/gateway-litellm.yaml"
_PLACEMENT_TEMPLATE = _CHART / "templates/gateway-placement.yaml"
_DOCUMENT_PARSING_TEMPLATE = _CHART / "templates/document-parsing.yaml"
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
    assert values["model"]["name"] == "unsloth/Qwen3.8-27B-NVFP4"
    assert _COMMIT.fullmatch(values["model"]["revision"])
    assert values["server"]["maxNumBatchedTokens"] == 32768
    assert values["huggingFaceTokenSecret"]["name"] == ""
    assert values["persistence"]["enabled"] is True
    assert values["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert values["resources"]["limits"]["nvidia.com/gpu"] == "1"
    # Speculative decoding conflicts with asynchronous scheduling and forfeits
    # much of the reusable prefix, so it is opt-in behind a benchmark.
    assert values["server"]["speculativeTokens"] == 0
    # The checkpoint declares its own scheme, and asserting a different one is a
    # startup failure. The reference build is compressed-tensors, not modelopt.
    assert values["server"]["quantization"] == "compressed-tensors"


def test_the_shipped_sequence_limit_is_one_the_sizing_formula_supports() -> None:
    """Ship a default that keeps KV-pressure eviction of batch work impossible.

    The chart and the sidecar must agree on this or the pods refuse to start,
    so the contract is checked against the formula rather than a magic number.
    """

    values = _load_yaml(_VALUES)["modelServing"]
    sizing = values["sizing"]

    verdict = compute_sizing(
        ModelGeometry(
            kv_heads=sizing["kvHeads"],
            head_dim=sizing["headDim"],
            attention_layers=sizing["attentionLayers"],
            kv_cache_dtype=values["server"]["kvCacheDtype"],
            weights_bytes=int(sizing["weightsGiB"] * BYTES_PER_GIB),
        ),
        DeviceBudget(
            device_memory_bytes=int(sizing["deviceMemoryGiB"] * BYTES_PER_GIB),
            gpu_memory_utilization=values["server"]["gpuMemoryUtilization"],
            overhead_bytes=int(sizing["overheadGiB"] * BYTES_PER_GIB),
        ),
        sizing["expectedContextTokens"],
        values["server"]["maxNumSeqs"],
    )

    assert verdict.sequence_limit_binds_first, verdict.detail


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
        "VLLM_MARLIN_USE_ATOMIC_ADD",
        "huggingFaceTokenSecret",
        "fastsafetensors",
        "startupProbe:",
        "readinessProbe:",
        "livenessProbe:",
    ]
    for fragment in required_fragments:
        assert fragment in template


def test_priority_scheduling_cannot_be_configured_away() -> None:
    """Render the flag unconditionally; without it every stamp is silently ignored.

    There is no values key that omits it and no branch that guards it, because
    an engine serving FIFO looks exactly like an engine honouring priority until
    someone measures the wait.
    """

    template = _MODEL_TEMPLATE.read_text(encoding="utf-8")
    before, _, after = template.partition("- --scheduling-policy")

    assert after.startswith("\n            - priority")
    # Nothing between the engine's own argument list and the flag may branch,
    # so no values file can leave it out.
    assert "{{- if" not in before.rsplit("- serve", maxsplit=1)[1]


def test_the_engine_is_reachable_only_through_the_enforcement_floor() -> None:
    """Bind the engine to loopback so no route to inference skips the sidecar."""

    template = _MODEL_TEMPLATE.read_text(encoding="utf-8")
    policy = _NETWORK_POLICY_TEMPLATE.read_text(encoding="utf-8")

    assert "- --host\n            - 127.0.0.1" in template
    assert "command: [flakegraph, serving, sidecar]" in template
    assert "FLAKEGRAPH_SIDECAR_KEYS_FILE" in template
    assert "FLAKEGRAPH_SIDECAR_BANDS" in template
    # The sidecar refuses to start on a limit the KV budget cannot support, so
    # the numbers it checks must actually reach it.
    assert "FLAKEGRAPH_SIDECAR_GEOMETRY" in template
    assert "FLAKEGRAPH_SIDECAR_DEVICE_BUDGET" in template
    assert "FLAKEGRAPH_SIDECAR_MAX_NUM_SEQS" in template
    # Probes belong to the floor: a healthy engine behind an unhealthy sidecar
    # is not a servable replica.
    assert template.index("startupProbe:") < template.index("name: vllm")
    assert "kind: NetworkPolicy" in policy
    assert "app.kubernetes.io/component: inference-router" in policy
    assert "enginePort" not in policy


def test_the_engine_pod_accepts_a_site_supplied_trust_store() -> None:
    """Let an operator give the engine the CA bundle its egress requires.

    The pod downloads its own weights. Where TLS is terminated on the way out,
    Hub metadata still resolves while the weight transfer fails verification —
    which presents as a stalled download, not an error, so the escape hatch has
    to exist rather than being discovered under time pressure.
    """

    values = _load_yaml(_VALUES)["modelServing"]
    template = _MODEL_TEMPLATE.read_text(encoding="utf-8")

    for field in ("extraEnv", "extraVolumes", "extraVolumeMounts"):
        assert values[field] == []
        assert f".Values.modelServing.{field}" in template
    # The mounts belong to the engine, which is the container that downloads.
    engine = template.split("name: vllm", maxsplit=1)[1]
    assert ".Values.modelServing.extraVolumeMounts" in engine
    assert ".Values.modelServing.extraEnv" in engine


def test_the_gateway_expresses_priority_as_which_alias_a_key_may_call() -> None:
    """Keep the class-to-band mapping out of the request body entirely."""

    template = _LITELLM_TEMPLATE.read_text(encoding="utf-8")
    values = _load_yaml(_VALUES)["gateway"]

    assert values["enabled"] is True
    assert set(values["litellm"]["aliases"]) == {"interactive", "dev", "batch"}
    # Each alias must present a *different* upstream key. Sharing one would make
    # every class the same band while still looking correctly configured.
    for field in ("interactiveKey", "devKey", "batchKey"):
        assert f"upstreamKeySecret.{field}" in template
    upstream_keys = values["litellm"]["upstreamKeySecret"]
    assert len({upstream_keys[field] for field in ("interactiveKey", "devKey", "batchKey")}) == 3
    assert len(set(values["litellm"]["aliases"].values())) == 3
    # Body injection would be stripped by the sidecar and would collide with
    # LiteLLM's own reserved field name besides.
    assert "priority:" not in template
    # Pointed at FlakeGraph's own schema, LiteLLM's migration tool baselines
    # instead of migrating and silently creates none of its tables.
    assert values["litellm"]["databaseUrlSuffix"].endswith("schema=litellm")
    assert "$(FLAKEGRAPH_DATABASE_URI)" in template
    assert "proxy_batch_write_at" in template


def test_placement_routes_on_the_pickers_choice_and_needs_no_crds() -> None:
    """Keep the chart installable where no Inference Extension CRDs exist."""

    template = _PLACEMENT_TEMPLATE.read_text(encoding="utf-8")
    chart_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(_CHART.rglob("*")) if path.is_file()
    )

    assert "type: ORIGINAL_DST" in template
    assert "http_header_name: x-gateway-destination-endpoint" in template
    assert "envoy.filters.http.ext_proc" in template
    assert "--endpoint-selector" in template
    assert "prefix-cache-scorer" in template
    # An InferencePool would make a CRD a precondition for installing at all.
    assert "InferencePool" not in chart_text
    assert "kind: Role" in template
    assert "resources: [pods]" in template


def test_document_parsing_holds_work_rather_than_letting_it_fail() -> None:
    """Give the shim a resolvable pool and the capacity it must not exceed."""

    template = _DOCUMENT_PARSING_TEMPLATE.read_text(encoding="utf-8")
    values = _load_yaml(_VALUES)["documentParsing"]

    assert values["enabled"] is True
    # MinerU's own default of three is too low to keep a node busy.
    assert values["mineru"]["maxConcurrentRequests"] > 3
    assert "command: [flakegraph, serving, ocr-shim]" in template
    assert "FLAKEGRAPH_OCR_SHIM_UPSTREAM_CAPACITY" in template
    assert "FLAKEGRAPH_OCR_SHIM_DATABASE_URL" in template
    assert "MINERU_API_MAX_CONCURRENT_REQUESTS" in template
    assert "- 0.0.0.0" in template
    # Headless: a load-balanced ClusterIP would hide the per-replica load that
    # admission control has to count.
    assert "clusterIP: None" in template
    # The parsing pool runs the same image as everything else. MinerU is already
    # installed in it and exposes an entry point, so a second image would only
    # add another artefact to keep on the right architecture.
    assert "command: [mineru-api]" in template
    # Every path the parser writes to has to be one of the writable mounts; the
    # image's defaults sit inside the read-only layer.
    assert "MINERU_API_OUTPUT_ROOT" in template
    for variable in ("MINERU_API_OUTPUT_ROOT", "XDG_CACHE_HOME", "HF_HOME", "HOME"):
        value = template.split(f"name: {variable}", maxsplit=1)[1].split("value:", maxsplit=1)[1]
        assert value.strip().splitlines()[0].strip().startswith(("/tmp", "/models")), variable
    assert "mineru" not in values or "image" not in values["mineru"]
    assert template.count('include "flakegraph.image"') == 2


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


def test_the_pipeline_is_a_metered_consumer_like_any_other() -> None:
    """Send batch traffic through the same gateway, holding a batch virtual key.

    The exception that used to exist here — workers addressing an engine
    directly — made "what is consuming this fleet" a question with a partial
    answer, and pinned a node's workers to the GPU beside them so an idle pool
    left that GPU idle too.
    """

    template = _WORKER_TEMPLATE.read_text(encoding="utf-8")
    helpers = (_CHART / "templates/_helpers.tpl").read_text(encoding="utf-8")
    consumer_env = helpers.split('define "flakegraph.consumerEnv"', maxsplit=1)[1].split(
        "{{- end -}}", maxsplit=1
    )[0]

    assert "KG_LLM_ENDPOINT" in consumer_env
    assert "KG_LLM_MODEL" in consumer_env
    assert "KG_LLM_API_KEY" in consumer_env
    assert 'include "flakegraph.gatewayEndpoint"' in consumer_env
    assert ".Values.gateway.litellm.aliases.batch" in consumer_env
    assert "KG_MINERU_API_URL" in consumer_env
    assert 'include "flakegraph.consumerEnv"' in template
    # Embeddings stay an independently configured provider.
    assert "KG_EMBED_ENDPOINT" not in template
    assert "KG_EMBED_MODEL" not in template
    assert "topologySpreadConstraints:" in template


def test_the_llm_credential_is_declared_exactly_once_per_container() -> None:
    """Never let two sources define KG_LLM_API_KEY in one pod spec.

    Kubernetes resolves a duplicated variable by ordering, not by intent, and an
    optional key that is absent can blank a value another source just set. With
    the gateway on, its virtual key is the credential and the provider Secret's
    mapping must stand aside.
    """

    template = _WORKER_TEMPLATE.read_text(encoding="utf-8")

    assert 'eq $mapping.name "KG_LLM_API_KEY"' in template
    guard = template.split("range $mapping := $root.Values.providerSecret.env", maxsplit=1)[1]
    assert "$root.Values.gateway.enabled" in guard.split("- name: {{ $mapping.name }}")[0]


def test_the_validator_is_configured_exactly_like_what_it_validates() -> None:
    """Emit one env block for workers and for the Job that preflights them.

    The bootstrap Job runs preflight against the same profile the workers run.
    Configured separately, the two have drifted in both directions: the Job has
    passed a profile the workers could not execute, and failed one they could.
    """

    workers = _WORKER_TEMPLATE.read_text(encoding="utf-8")
    bootstrap = _DATABASE_BOOTSTRAP_TEMPLATE.read_text(encoding="utf-8")
    helpers = (_CHART / "templates/_helpers.tpl").read_text(encoding="utf-8")

    assert 'define "flakegraph.consumerEnv"' in helpers
    for template in (workers, bootstrap):
        assert 'include "flakegraph.consumerEnv"' in template
        # Neither may hand-roll the variables the shared block owns.
        assert "- name: KG_LLM_ENDPOINT" not in template
        assert "- name: KG_MINERU_API_URL" not in template
    # The Job must not be left pointing at an engine the consumers no longer use.
    assert 'include "flakegraph.modelServingEndpoint"' not in bootstrap


def test_no_consumer_can_still_address_an_engine_directly() -> None:
    """Leave one endpoint key, so the placement layer stays replaceable."""

    chart_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(_CHART.rglob("*")) if path.is_file()
    )
    example = _PUBLIC_EXAMPLE.read_text(encoding="utf-8")

    assert "localModelServing" not in chart_text
    assert "localModelServing" not in example
    assert "internalTrafficPolicy" not in chart_text


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
    assert "gateway" in schema["required"]
    assert "documentParsing" in schema["required"]
    assert schema["properties"]["modelServing"]["additionalProperties"] is False
    assert example["modelServing"] == {
        "enabled": True,
        "replicas": 4,
        "nodeSelector": {"flakegraph.io/node-class": "nvidia-spark"},
    }
    assert example["gateway"]["placement"]["replicas"] == 2
    assert example["documentParsing"]["mineru"]["replicas"] == 4
    prepare = example["workers"]["prepare"]
    assert prepare["replicas"] == 16
    assert prepare["autoscaling"]["maxReplicas"] == 16
    assert prepare["topologySpreadConstraints"][0]["maxSkew"] == 1
    assert prepare["topologySpreadConstraints"][0]["matchLabelKeys"] == ["pod-template-hash"]
    extract = example["workers"]["extract"]
    assert extract["replicas"] == 16
    assert extract["autoscaling"]["maxReplicas"] == 16
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
    assert "CREATE VIEW flakegraph_worker_demand" in postgres
    assert "task.status = 'running'" in postgres
    assert "remaining_dependencies" in postgres
    assert "FROM flakegraph_publication AS publication" in postgres
    assert "SELECT 'finalize_graph' AS stage" in postgres


def test_each_pool_scales_on_each_priority_band_independently() -> None:
    """Let a small urgent run raise a drained pool without a bulk backlog.

    With one combined trigger, demand from an urgent run is indistinguishable
    from demand from a queue of a hundred thousand bulk tasks.
    """

    autoscaling = _AUTOSCALING_TEMPLATE.read_text(encoding="utf-8")
    postgres = Path("src/kg_processor/adapters/distributed/postgres.py").read_text(encoding="utf-8")

    assert 'range $band := list "interactive" "bulk"' in autoscaling
    assert "WHERE priority_band = '{{ $band }}'" in autoscaling
    # The cap stays inside each band, so one band cannot spend the other's.
    assert "SUM(LEAST(" in autoscaling
    assert "AS priority_band" in postgres
    assert "DROP VIEW IF EXISTS flakegraph_worker_demand" in postgres


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
    # The serving plane outranks the workers deliberately: it is the path every
    # consumer takes, so leaving it at the default priority lets a queue scale-up
    # preempt the gateway those same workers are trying to reach.
    assert (
        priorities["workerValue"]
        < priorities["sparkValue"]
        < priorities["servingValue"]
        < priorities["modelValue"]
    )
    assert priority_template.count("kind: PriorityClass") == 4
    for template_path in (
        _LITELLM_TEMPLATE,
        _PLACEMENT_TEMPLATE,
        _DOCUMENT_PARSING_TEMPLATE,
    ):
        assert (
            'include "flakegraph.servingPriorityClassName"'
            in template_path.read_text(encoding="utf-8")
        )
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

    # The risk this guards is model operations drifting into a second guide that
    # then disagrees with this one, not the docs directory gaining a file. Pin
    # the guide's presence and forbid a parallel host/model guide by name; a
    # document on another subject is free to exist.
    assert "kubernetes-fleet.md" in top_level_docs
    forbidden = [
        name
        for name in top_level_docs
        if name != "kubernetes-fleet.md"
        and any(word in name for word in ("model", "serving", "vllm", "host", "gpu"))
    ]
    assert not forbidden, f"model-serving documentation must stay in the fleet guide: {forbidden}"
    assert "## Serving Plane" in fleet_guide
    assert "KEDA 2.20.1" in fleet_guide
    assert "kind: ScaledObject" in _AUTOSCALING_TEMPLATE.read_text(encoding="utf-8")
    assert "autoscaling:" in fleet_guide
    assert "--set-file config.content=deploy/private/fleet-config.yaml" in fleet_guide
    # The guide has to state the trap, because both conventions live in one repo.
    prose = " ".join(fleet_guide.split())
    assert "lower is served first" in prose
    assert "priority DESC" in prose


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load one repository-owned YAML mapping for contract assertions."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
