"""Protect the Kubernetes-managed serving and document-parsing contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from helm import FULLNAME as _FULLNAME
from helm import args as _args
from helm import container as _container
from helm import env as _env
from helm import fails as _fails
from helm import load_yaml as _load_yaml
from helm import notes as _notes
from helm import one as _one
from helm import pod as _pod
from helm import render as _render
from helm import schema as _schema
from helm import values as _values

from kg_processor.adapters.distributed.postgres import _SCHEMA_STATEMENTS
from kg_processor.serving.sizing import (
    BYTES_PER_GIB,
    DeviceBudget,
    ModelGeometry,
    compute_sizing,
)

_PUBLIC_EXAMPLE = Path("deploy/examples/k3s-spark-values.yaml")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

# The serving plane is opt-in, and its shipped draft model is a path that
# something outside the chart has to supply.
_SERVING = (
    "modelServing.enabled=true",
    "modelServing.server.draftModelSeed.providedExternally=true",
)
_SPARK = (*_SERVING, "spark.enabled=true")
_VLLM = f"{_FULLNAME}-vllm"
_ROUTER = f"{_FULLNAME}-inference-router"
_WORKER_POOLS = ("prepare", "extract", "finalize")
_CONSUMER_ENV = (
    "KG_LLM_ENDPOINT",
    "KG_LLM_MODEL",
    "KG_LLM_API_KEY",
    "KG_MINERU_API_URL",
    "KG_MINERU_API_KEY",
)


def test_model_serving_defaults_are_pinned_and_resource_bounded() -> None:
    """Keep opt-in inference reproducible instead of following moving artifacts."""

    values = _values()["modelServing"]

    assert values["enabled"] is False
    assert values["runtime"] == "vllm"
    # vLLM mainline. NVIDIA's fork stopped at 0.21 and never registered
    # DFlash2DraftModel, so the drafter configured below cannot load on it.
    assert values["image"]["repository"] == "vllm/vllm-openai"
    assert values["image"]["tag"] == "v0.28.0"
    assert (
        values["image"]["digest"]
        == "sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce"
    )
    assert values["model"]["name"] == "unsloth/Qwen3.8-27B-NVFP4"
    assert _COMMIT.fullmatch(values["model"]["revision"])
    assert values["server"]["maxNumBatchedTokens"] == 32768
    assert values["huggingFaceTokenSecret"]["name"] == ""
    assert values["persistence"]["enabled"] is True
    assert values["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert values["resources"]["limits"]["nvidia.com/gpu"] == "1"
    # Decode is bandwidth-bound here, so the drafter is what breaks the one
    # token per weight read ceiling. DFlash2 proposes a block of eight.
    assert values["server"]["speculativeTokens"] == 8
    assert values["server"]["speculativeMethod"] == "dflash"
    # The checkpoint declares its own scheme, and asserting a different one is a
    # startup failure. The reference build is compressed-tensors, not modelopt.
    assert values["server"]["quantization"] == "compressed-tensors"
    # Served text-only, so the vision tower must neither be profiled nor
    # reachable. Profiling it reserves activation memory for an encoder no
    # request will use, which is memory the KV cache does not get.
    # The checkpoint is a vision-language model, so it serves images. Video is
    # off: nothing consumes it, and it would reserve far more encoder memory.
    assert values["server"]["skipMultimodalProfiling"] is False
    assert values["server"]["limitMultimodalPerPrompt"]["image"] > 0
    assert values["server"]["limitMultimodalPerPrompt"]["video"] == 0


def test_a_local_draft_checkpoint_is_not_pinned_to_a_hub_revision() -> None:
    """Keep the drafter's path and its revision from being set together.

    A filesystem path has no Hub commit behind it. Naming one anyway is not a
    no-op: the engine tries to resolve the revision against a repository that
    does not exist and fails at startup, long after the chart looked correct.
    """

    server = _values()["modelServing"]["server"]

    assert server["speculativeDraftModel"] == "/models/dflash2"
    assert server["speculativeDraftRevision"] == ""


def test_the_weight_budget_counts_the_drafter_that_stays_resident() -> None:
    """Charge the draft model against the KV budget it actually competes with.

    The drafter occupies device memory for the whole life of the process, so a
    weights figure covering only the target checkpoint overstates how many
    sequences fit and walks the engine into KV-pressure preemption.
    """

    values = _values()["modelServing"]

    # The engine reports "Model loading took 24.24 GiB" for the target plus the
    # DFlash2 draft; the target alone is 21.81.
    assert values["sizing"]["weightsGiB"] == 24.24


def test_the_shipped_sequence_limit_is_one_the_sizing_formula_supports() -> None:
    """Ship a default that keeps KV-pressure eviction of batch work impossible.

    The chart and the sidecar must agree on this or the pods refuse to start,
    so the contract is checked against the formula rather than a magic number.
    """

    values = _values()["modelServing"]
    sizing = values["sizing"]

    verdict = compute_sizing(
        ModelGeometry(
            kv_heads=sizing["kvHeads"],
            head_dim=sizing["headDim"],
            attention_layers=sizing["attentionLayers"],
            kv_cache_dtype=values["server"]["kvCacheDtype"],
            weights_bytes=int(sizing["weightsGiB"] * BYTES_PER_GIB),
            recurrent_layers=sizing["recurrentLayers"],
            recurrent_state_bytes_per_layer=sizing["recurrentStateBytesPerLayer"],
            recurrent_state_slots_per_sequence=sizing["recurrentStateSlotsPerSequence"],
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


def test_the_engine_owns_the_complete_model_lifecycle() -> None:
    """Require storage, pinned model loading, probes, spreading, and service discovery."""

    values = _values()["modelServing"]
    rendered = _render(_SERVING)
    engines = _one(rendered, "StatefulSet", _VLLM)
    pod = _pod(engines)
    engine = _container(pod, "vllm")
    args = _args(engine)

    assert (
        _one(rendered, "Service", _VLLM)["spec"]["selector"]
        == engines["spec"]["selector"]["matchLabels"]
    )
    assert engines["spec"]["serviceName"] == f"{_VLLM}-headless"
    assert [claim["metadata"]["name"] for claim in engines["spec"]["volumeClaimTemplates"]] == [
        "models"
    ]
    # One engine per host, and required: two engines cannot share the GPU.
    required = pod["affinity"]["podAntiAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
    assert required[0]["labelSelector"]["matchLabels"] == engines["spec"]["selector"]["matchLabels"]
    assert args["--revision"] == values["model"]["revision"]
    assert args["--max-num-batched-tokens"] == str(values["server"]["maxNumBatchedTokens"])
    assert args["--limit-mm-per-prompt"] == json.dumps(
        values["server"]["limitMultimodalPerPrompt"], separators=(",", ":")
    )
    assert _env(engine)["VLLM_MARLIN_USE_ATOMIC_ADD"]["value"] == "1"
    # Probes belong to the enforcement floor, which the next test pins.
    assert {"startupProbe", "readinessProbe", "livenessProbe"} <= set(_container(pod, "sidecar"))


def test_the_engine_reads_a_private_hub_and_a_named_loader_only_when_told() -> None:
    """A Hub token and a loader are opt-in, and each reaches the engine intact."""

    gated = _render(
        (
            *_SERVING,
            "modelServing.huggingFaceTokenSecret.name=hub-token",
            "modelServing.server.loadFormat=safetensors",
            "modelServing.server.skipMultimodalProfiling=true",
        )
    )
    engine = _container(_pod(_one(gated, "StatefulSet", _VLLM)), "vllm")
    values = _values()["modelServing"]["huggingFaceTokenSecret"]

    assert _env(engine)["HF_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "hub-token",
        "key": values["key"],
        "optional": values["optional"],
    }
    assert _args(engine)["--load-format"] == "safetensors"
    assert "--skip-mm-profiling" in engine["args"]

    default = _container(_pod(_one(_render(_SERVING), "StatefulSet", _VLLM)), "vllm")
    assert "HF_TOKEN" not in _env(default)
    assert "--load-format" not in default["args"]
    assert "--skip-mm-profiling" not in default["args"]


def test_engine_tuning_flags_can_be_configured_away() -> None:
    """Let an empty value omit a flag rather than render an empty argument.

    Every one of these names a backend, a loader, or a scheme that a given vLLM
    build may not carry, and vLLM fails at startup on an unknown name rather
    than falling back. A chart that can only ever assert one is a chart that
    cannot follow the engine across an upgrade — which is exactly the move that
    stranded this deployment on a fork.
    """

    flags = {
        "quantization": "--quantization",
        "attentionBackend": "--attention-backend",
        "moeBackend": "--moe-backend",
        "loadFormat": "--load-format",
    }
    cleared = _render((*_SERVING, *(f"modelServing.server.{key}=" for key in flags)))
    engine = _container(_pod(_one(cleared, "StatefulSet", _VLLM)), "vllm")

    for flag in flags.values():
        assert flag not in engine["args"], flag
    assert "" not in engine["args"]


def test_priority_scheduling_cannot_be_configured_away() -> None:
    """Render the flag unconditionally; without it every stamp is silently ignored.

    There is no values key that omits it, because an engine serving FIFO looks
    exactly like an engine honouring priority until someone measures the wait.
    """

    schema = _schema()
    server = schema["properties"]["modelServing"]["properties"]["server"]

    assert server["additionalProperties"] is False
    assert not any("scheduling" in key.lower() for key in server["properties"])
    for rendered in (
        _render(_SERVING),
        _render((*_SERVING, "modelServing.server.quantization=")),
        _render(_SERVING[1:], values=(_PUBLIC_EXAMPLE,)),
    ):
        engine = _container(_pod(_one(rendered, "StatefulSet", _VLLM)), "vllm")
        assert _args(engine)["--scheduling-policy"] == "priority"


def test_the_engine_is_reachable_only_through_the_enforcement_floor() -> None:
    """Bind the engine to loopback so no route to inference skips the sidecar."""

    values = _values()["modelServing"]
    rendered = _render(_SERVING)
    pod = _pod(_one(rendered, "StatefulSet", _VLLM))
    engine = _container(pod, "vllm")
    sidecar = _container(pod, "sidecar")
    sidecar_env = _env(sidecar)

    assert _args(engine)["--host"] == "127.0.0.1"
    assert _args(engine)["--port"] == str(values["server"]["enginePort"])
    # The engine's own port is never a container port, so no Service can name it.
    assert values["server"]["enginePort"] not in {
        port["containerPort"] for container in pod["containers"] for port in container["ports"]
    }
    assert sidecar["command"] == ["flakegraph", "serving", "sidecar"]
    assert sidecar_env["FLAKEGRAPH_SIDECAR_UPSTREAM"]["value"] == (
        f"http://127.0.0.1:{values['server']['enginePort']}"
    )
    assert sidecar_env["FLAKEGRAPH_SIDECAR_KEYS_FILE"]["value"].endswith(
        values["sidecar"]["keySecret"]["key"]
    )
    assert (
        json.loads(sidecar_env["FLAKEGRAPH_SIDECAR_BANDS"]["value"]) == values["sidecar"]["bands"]
    )
    # The sidecar refuses to start on a limit the KV budget cannot support, so
    # the numbers it checks must actually reach it.
    geometry = json.loads(sidecar_env["FLAKEGRAPH_SIDECAR_GEOMETRY"]["value"])
    assert geometry["kv_heads"] == values["sizing"]["kvHeads"]
    assert geometry["weights_bytes"] == int(values["sizing"]["weightsGiB"] * BYTES_PER_GIB)
    budget = json.loads(sidecar_env["FLAKEGRAPH_SIDECAR_DEVICE_BUDGET"]["value"])
    assert budget["gpu_memory_utilization"] == values["server"]["gpuMemoryUtilization"]
    assert sidecar_env["FLAKEGRAPH_SIDECAR_MAX_NUM_SEQS"]["value"] == str(
        values["server"]["maxNumSeqs"]
    )
    # Probes belong to the floor: a healthy engine behind an unhealthy sidecar
    # is not a servable replica.
    assert {"startupProbe", "readinessProbe", "livenessProbe"} <= set(sidecar)
    assert not {"startupProbe", "readinessProbe", "livenessProbe"} & set(engine)

    policy = _one(rendered, "NetworkPolicy", _VLLM)
    router = {"app.kubernetes.io/component": "inference-router"}
    admitted = {
        port["port"]
        for rule in policy["spec"]["ingress"]
        for peer in rule["from"]
        for port in rule["ports"]
        if router.items() <= peer["podSelector"]["matchLabels"].items()
    }
    assert "http" in admitted
    assert values["server"]["enginePort"] not in admitted


def test_the_engine_pod_accepts_a_site_supplied_trust_store() -> None:
    """Let an operator give the engine the CA bundle its egress requires.

    The pod downloads its own weights. Where TLS is terminated on the way out,
    Hub metadata still resolves while the weight transfer fails verification —
    which presents as a stalled download, not an error, so the escape hatch has
    to exist rather than being discovered under time pressure.
    """

    values = _values()["modelServing"]
    for field in ("extraEnv", "extraVolumes", "extraVolumeMounts"):
        assert values[field] == []

    rendered = _render(
        (
            *_SERVING,
            "modelServing.extraEnv[0].name=SSL_CERT_FILE",
            "modelServing.extraEnv[0].value=/etc/ssl/site/ca.crt",
            "modelServing.extraVolumes[0].name=site-ca",
            "modelServing.extraVolumes[0].secret.secretName=site-ca",
            "modelServing.extraVolumeMounts[0].name=site-ca",
            "modelServing.extraVolumeMounts[0].mountPath=/etc/ssl/site",
        )
    )
    pod = _pod(_one(rendered, "StatefulSet", _VLLM))
    engine = _container(pod, "vllm")

    # The mounts belong to the engine, which is the container that downloads.
    assert _env(engine)["SSL_CERT_FILE"]["value"] == "/etc/ssl/site/ca.crt"
    assert {"name": "site-ca", "mountPath": "/etc/ssl/site"} in engine["volumeMounts"]
    assert {"name": "site-ca", "secret": {"secretName": "site-ca"}} in pod["volumes"]
    sidecar = _container(pod, "sidecar")
    assert "SSL_CERT_FILE" not in _env(sidecar)
    assert "site-ca" not in {mount["name"] for mount in sidecar["volumeMounts"]}


def test_the_gateway_expresses_priority_as_which_alias_a_key_may_call() -> None:
    """Keep the class-to-band mapping out of the request body entirely."""

    values = _values()["gateway"]
    rendered = _render(())
    config = yaml.safe_load(
        _one(rendered, "ConfigMap", f"{_FULLNAME}-litellm")["data"]["config.yaml"]
    )
    gateway = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-litellm")), "litellm")
    env = _env(gateway)

    assert set(values["litellm"]["aliases"]) == {"interactive", "dev", "batch"}
    # Each alias must present a *different* upstream key. Sharing one would make
    # every class the same band while still looking correctly configured.
    upstream = values["litellm"]["upstreamKeySecret"]
    served = {entry["model_name"]: entry["litellm_params"] for entry in config["model_list"]}
    assert set(served) == set(values["litellm"]["aliases"].values())
    assert {params["api_key"] for params in served.values()} == {
        f"os.environ/{upstream[field]}" for field in ("interactiveKey", "devKey", "batchKey")
    }
    assert len({params["api_key"] for params in served.values()}) == 3
    for field in ("interactiveKey", "devKey", "batchKey"):
        assert env[upstream[field]]["valueFrom"]["secretKeyRef"] == {
            "name": upstream["name"],
            "key": upstream[field],
        }
    # Body injection would be stripped by the sidecar and would collide with
    # LiteLLM's own reserved field name besides.
    assert "priority" not in json.dumps(config)
    # Pointed at FlakeGraph's own schema, LiteLLM's migration tool baselines
    # instead of migrating and silently creates none of its tables.
    assert values["litellm"]["databaseUrlSuffix"].endswith("schema=litellm")
    assert env["DATABASE_URL"]["value"] == (
        f"$(FLAKEGRAPH_DATABASE_URI){values['litellm']['databaseUrlSuffix']}"
    )
    assert (
        config["general_settings"]["proxy_batch_write_at"]
        == (values["litellm"]["spendLogging"]["batchWriteIntervalSeconds"])
    )


def test_placement_routes_on_the_pickers_choice_and_needs_no_crds() -> None:
    """Keep the chart installable where no Inference Extension CRDs exist."""

    rendered = _render(_SERVING)
    router = _one(rendered, "ConfigMap", _ROUTER)["data"]
    envoy = yaml.safe_load(router["envoy.yaml"])
    plugins = yaml.safe_load(router["default-plugins.yaml"])
    picker = _container(_pod(_one(rendered, "Deployment", _ROUTER)), "endpoint-picker")

    clusters = {cluster["name"]: cluster for cluster in envoy["static_resources"]["clusters"]}
    assert clusters["original_destination"]["type"] == "ORIGINAL_DST"
    assert clusters["original_destination"]["original_dst_lb_config"] == {
        "use_http_header": True,
        "http_header_name": "x-gateway-destination-endpoint",
    }
    listener = next(
        listener
        for listener in envoy["static_resources"]["listeners"]
        if listener["name"] == "inference"
    )
    http_filters = listener["filter_chains"][0]["filters"][0]["typed_config"]["http_filters"]
    assert "envoy.filters.http.ext_proc" in {http_filter["name"] for http_filter in http_filters}
    assert _args(picker)["--endpoint-selector"].endswith(
        "app.kubernetes.io/component=model-serving"
    )
    assert "prefix-cache-scorer" in {plugin["type"] for plugin in plugins["plugins"]}
    # An InferencePool would make a CRD a precondition for installing at all.
    assert not [doc for doc in rendered if doc["kind"] == "InferencePool"]
    role = _one(rendered, "Role", _ROUTER)
    assert role["rules"] == [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "watch", "list"]}
    ]


def test_the_console_runs_the_fleet_the_way_the_workers_do() -> None:
    """The console is the Node server, and its CLI sees the workers' fleet.

    The CLI the console shells out to submits to the same queue, reads the same
    artifact store, and resolves the same provider placeholders as the workers
    - a run it writes is one they claim, a graph they publish is one it reads.
    Behind the gate it trusts the gate's identity headers; without one it does
    not.
    """

    values = _values()
    gated = _render(
        (
            *_SERVING,
            "spark.enabled=true",
            "artifactStorage.uri=s3://artifacts",
            "artifactStorage.existingSecret=artifacts",
            "ingress.enabled=true",
            "ingress.domain=example.test",
            "ingress.authProxy.enabled=true",
            "controlPlane.networkPolicy.enabled=true",
            "controlPlane.ask.secretName=ask",
        )
    )
    pod = _pod(_one(gated, "Deployment", f"{_FULLNAME}-app"))
    console = _container(pod, "control-plane")
    env = _env(console)
    worker = _env(_worker(gated, "extract"))

    assert console["command"] == ["node", "server.js"]
    assert console["workingDir"] == "/app/react"
    assert env["PORT"]["value"] == str(values["controlPlane"]["service"]["port"])
    assert env["FLAKEGRAPH_CLI"]["value"] == "flakegraph"
    assert env["FLAKEGRAPH_APP_DEFAULT_RUNTIME"]["value"] == "kubernetes"
    assert env["FLAKEGRAPH_TRUST_IDENTITY_HEADERS"]["value"] == "true"
    for name in (
        "KG_DISTRIBUTED_DATABASE_URL",
        "KG_DISTRIBUTED_ARTIFACT_URI",
        "KG_LLM_ENDPOINT",
        "KG_MINERU_API_URL",
    ):
        assert env[name] == worker[name], name
    assert env["DATABASE_URL"]["valueFrom"] == env["KG_DISTRIBUTED_DATABASE_URL"]["valueFrom"]
    assert env["FLAKEGRAPH_ASK_BASE_URL"]["value"] == env["KG_LLM_ENDPOINT"]["value"]
    assert env["FLAKEGRAPH_ASK_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "ask",
        "key": values["controlPlane"]["ask"]["secretKey"],
    }
    assert {
        probe["httpGet"]["path"]
        for probe in (console["startupProbe"], console["readinessProbe"], console["livenessProbe"])
    } == {"/api/health"}
    assert {mount["mountPath"] for mount in console["volumeMounts"]} >= {
        "/app/react/.next/cache",
        "/tmp",
        values["controlPlane"]["persistence"]["stateRoot"],
    }

    ungated = _env(
        _container(_pod(_one(_render(()), "Deployment", f"{_FULLNAME}-app")), "control-plane")
    )
    assert "FLAKEGRAPH_TRUST_IDENTITY_HEADERS" not in ungated
    assert "FLAKEGRAPH_ASK_BASE_URL" not in ungated


def test_the_control_plane_can_see_what_preflight_asks_about() -> None:
    """A pool scaled to zero on purpose is told apart by its ScaledObject."""

    role = _one(_render(()), "Role", f"{_FULLNAME}-app")
    granted = {
        (group, resource)
        for rule in role["rules"]
        for group in rule["apiGroups"]
        for resource in rule["resources"]
        if "list" in rule["verbs"]
    }
    assert ("keda.sh", "scaledobjects") in granted
    assert ("apps", "statefulsets") in granted
    # Recovery may restart the worker pools and nothing else: the write grant
    # names them.
    writes = [
        rule for rule in role["rules"] if "patch" in rule["verbs"] or "update" in rule["verbs"]
    ]
    assert len(writes) == 1
    assert set(writes[0]["resources"]) == {"deployments", "deployments/scale"}
    assert set(writes[0]["resourceNames"]) == {
        f"{_FULLNAME}-prepare",
        f"{_FULLNAME}-extract",
        f"{_FULLNAME}-finalize",
    }


def test_document_parsing_holds_work_rather_than_letting_it_fail() -> None:
    """Give the shim a resolvable pool and the capacity it must not exceed."""

    values = _values()
    parsing = values["documentParsing"]
    rendered = _render(())
    pool = _one(rendered, "StatefulSet", f"{_FULLNAME}-mineru")
    parser = _container(_pod(pool), "mineru")
    shim = _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-ocr")), "ocr-shim")
    shim_env = _env(shim)

    # MinerU's own default of three is too low to keep a node busy.
    assert parsing["mineru"]["maxConcurrentRequests"] > 3
    assert shim["command"] == ["flakegraph", "serving", "ocr-shim"]
    assert shim_env["FLAKEGRAPH_OCR_SHIM_UPSTREAM_CAPACITY"]["value"] == str(
        parsing["mineru"]["maxConcurrentRequests"]
    )
    assert shim_env["FLAKEGRAPH_OCR_SHIM_UPSTREAM_HOST"]["value"] == f"{_FULLNAME}-mineru"
    assert shim_env["FLAKEGRAPH_OCR_SHIM_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": values["database"]["secretName"],
        "key": values["database"]["secretKey"],
    }
    assert _env(parser)["MINERU_API_MAX_CONCURRENT_REQUESTS"]["value"] == str(
        parsing["mineru"]["maxConcurrentRequests"]
    )
    assert _args(parser)["--host"] == "0.0.0.0"
    # Headless: a load-balanced ClusterIP would hide the per-replica load that
    # admission control has to count.
    assert _one(rendered, "Service", f"{_FULLNAME}-mineru")["spec"]["clusterIP"] == "None"
    # The pool authenticates nobody and answers 409 to whatever skips the
    # queue, so the shim is the only pod allowed to reach it.
    policy = _one(rendered, "NetworkPolicy", f"{_FULLNAME}-mineru")
    assert policy["spec"]["policyTypes"] == ["Ingress"]
    assert [
        (peer["podSelector"]["matchLabels"]["app.kubernetes.io/component"], port["port"])
        for rule in policy["spec"]["ingress"]
        for peer in rule["from"]
        for port in rule["ports"]
    ] == [("ocr-shim", "http")]
    # The parsing pool runs the same image as everything else. MinerU is already
    # installed in it and exposes an entry point, so a second image would only
    # add another artefact to keep on the right architecture.
    assert parser["command"] == ["mineru-api"]
    assert "image" not in parsing["mineru"]
    worker = _worker(rendered, "extract")
    assert parser["image"] == worker["image"]
    assert shim["image"] == worker["image"]
    # Every path the parser writes to has to be one of the writable mounts; the
    # image's defaults sit inside the read-only layer.
    writable = tuple(mount["mountPath"] for mount in parser["volumeMounts"])
    for variable in ("MINERU_API_OUTPUT_ROOT", "XDG_CACHE_HOME", "HF_HOME", "HOME"):
        assert _env(parser)[variable]["value"].startswith(writable), variable
    # The worker waits at least as long as the shim holds a document. Giving up
    # sooner leaves the pool parsing for nobody and retries the parse from the
    # start; the workers' grace period then covers the parse it is finishing.
    config = yaml.safe_load(values["config"]["content"])
    assert config["ocr"]["timeout_seconds"] >= parsing["shim"]["requestTimeoutSeconds"]
    assert values["terminationGracePeriodSeconds"] > config["ocr"]["timeout_seconds"]


def test_the_sidecar_can_hold_its_tag_while_the_application_moves() -> None:
    """A release of the application must not have to roll every engine.

    The sidecar is the engine pod's one container built from the application
    image, and it rarely changes. Pinned, the engines stay up through a worker
    release; unpinned, it follows the application tag as before.
    """

    following = _render((*_SERVING, "image.tag=0.3.11"))
    pinned = _render((*_SERVING, "image.tag=0.3.11", "modelServing.sidecar.imageTag=0.3.10"))

    for rendered, tag in ((following, "0.3.11"), (pinned, "0.3.10")):
        pod = _pod(_one(rendered, "StatefulSet", _VLLM))
        assert _container(pod, "sidecar")["image"].endswith(f"/flakegraph:{tag}")
        assert _worker(rendered, "extract")["image"].endswith("/flakegraph:0.3.11")


def test_document_parsing_takes_the_device_without_claiming_it() -> None:
    """On the GPU the parser is handed the device, never scheduled for it.

    The engine on the same host holds the node's one nvidia.com/gpu, so a claim
    would never schedule; the runtime class injects the device for a pod that
    names it. Off the GPU nothing about the pod mentions a device, so the pool
    still runs where no runtime class exists.
    """

    cpu = _pod(_one(_render(()), "StatefulSet", f"{_FULLNAME}-mineru"))
    cuda = _pod(
        _one(
            _render(("documentParsing.mineru.device.mode=cuda",)),
            "StatefulSet",
            f"{_FULLNAME}-mineru",
        )
    )

    assert "runtimeClassName" not in cpu
    assert _env(_container(cpu, "mineru"))["MINERU_DEVICE_MODE"]["value"] == "cpu"
    assert "NVIDIA_VISIBLE_DEVICES" not in _env(_container(cpu, "mineru"))
    assert cuda["runtimeClassName"] == _values()["modelServing"]["runtimeClassName"]
    parser = _env(_container(cuda, "mineru"))
    assert parser["MINERU_DEVICE_MODE"]["value"] == "cuda"
    assert parser["NVIDIA_VISIBLE_DEVICES"]["value"] == "all"
    assert parser["MINERU_VIRTUAL_VRAM_SIZE"]["value"] == str(
        _values()["documentParsing"]["mineru"]["device"]["virtualVramGiB"]
    )
    assert "nvidia.com/gpu" not in _container(cuda, "mineru")["resources"].get("limits", {})


def test_document_parsing_replicas_spread_across_hosts_without_requiring_it() -> None:
    """One parser per host where the fleet allows it, and still schedulable where not.

    The engines are forced apart by their GPU request; a parsing replica holds
    no GPU, so without a constraint four of them can land on one host and leave
    three idle. The constraint is soft on purpose: a pool larger than the fleet
    doubles up rather than pending forever.
    """

    values = _values()["documentParsing"]["mineru"]
    pool = _one(_render(()), "StatefulSet", f"{_FULLNAME}-mineru")

    assert _pod(pool)["topologySpreadConstraints"] == [
        {
            "maxSkew": 1,
            "topologyKey": values["topologyKey"],
            "whenUnsatisfiable": "ScheduleAnyway",
            "labelSelector": {"matchLabels": pool["spec"]["selector"]["matchLabels"]},
        }
    ]
    assert values["topologyKey"] == "kubernetes.io/hostname"


def test_spark_executor_spreading_degrades_gracefully() -> None:
    """Prefer fleet-wide placement without deadlocking finalization on node loss."""

    values = _values()["spark"]
    executor = _executor_template(_render(_SPARK))
    (container,) = executor["spec"]["containers"]

    preferred = executor["spec"]["affinity"]["podAntiAffinity"][
        "preferredDuringSchedulingIgnoredDuringExecution"
    ]
    assert preferred == [
        {
            "weight": 100,
            "podAffinityTerm": {
                "topologyKey": values["topologyKey"],
                "labelSelector": {"matchLabels": {"spark-role": "executor"}},
            },
        }
    ]
    assert (
        "requiredDuringSchedulingIgnoredDuringExecution"
        not in executor["spec"]["affinity"]["podAntiAffinity"]
    )
    assert executor["spec"]["securityContext"] == values["executorPodSecurityContext"]
    assert container["securityContext"] == values["executorContainerSecurityContext"]
    assert values["executorPodSecurityContext"]["runAsUser"] == 185
    assert values["executorContainerSecurityContext"]["allowPrivilegeEscalation"] is False


def test_the_pipeline_is_a_metered_consumer_like_any_other() -> None:
    """Send batch traffic through the same gateway, holding a batch virtual key.

    The exception that used to exist here — workers addressing an engine
    directly — made "what is consuming this fleet" a question with a partial
    answer, and pinned a node's workers to the GPU beside them so an idle pool
    left that GPU idle too.
    """

    values = _values()
    litellm = values["gateway"]["litellm"]
    rendered = _render(
        (
            "workers.extract.topologySpreadConstraints[0].maxSkew=1",
            "workers.extract.topologySpreadConstraints[0].topologyKey=kubernetes.io/hostname",
            "workers.extract.topologySpreadConstraints[0].whenUnsatisfiable=ScheduleAnyway",
        )
    )

    for pool in _WORKER_POOLS:
        env = _env(_worker(rendered, pool))
        assert env["KG_LLM_ENDPOINT"]["value"] == (
            f"http://{_FULLNAME}-litellm:{litellm['service']['port']}/v1"
        )
        assert env["KG_LLM_MODEL"]["value"] == litellm["aliases"]["batch"]
        assert env["KG_LLM_API_KEY"]["valueFrom"]["secretKeyRef"] == {
            "name": litellm["virtualKeySecret"]["name"],
            "key": litellm["virtualKeySecret"]["batchKey"],
        }
        assert env["KG_MINERU_API_URL"]["value"] == (
            f"http://{_FULLNAME}-ocr:{values['documentParsing']['shim']['service']['port']}"
        )
        # Embeddings stay an independently configured provider.
        assert "KG_EMBED_ENDPOINT" not in env
        assert "KG_EMBED_MODEL" not in env
    extract = _pod(_one(rendered, "Deployment", f"{_FULLNAME}-extract"))
    assert extract["topologySpreadConstraints"] == [
        {
            "maxSkew": 1,
            "topologyKey": "kubernetes.io/hostname",
            "whenUnsatisfiable": "ScheduleAnyway",
        }
    ]


def test_the_llm_credential_is_declared_exactly_once_per_container() -> None:
    """Never let two sources define KG_LLM_API_KEY in one pod spec.

    Kubernetes resolves a duplicated variable by ordering, not by intent, and an
    optional key that is absent can blank a value another source just set. With
    the gateway on, its virtual key is the credential and the provider Secret's
    mapping must stand aside.
    """

    litellm = _values()["gateway"]["litellm"]
    with_gateway = _render(("providerSecret.name=provider",))
    without_gateway = _render(("providerSecret.name=provider", "gateway.enabled=false"))

    consumers = [(pool, _worker) for pool in _WORKER_POOLS] + [
        ("bootstrap", lambda r, _: _bootstrap(r))
    ]
    for pool, container in consumers:
        gateway_consumer = container(with_gateway, pool)
        names = [entry["name"] for entry in gateway_consumer["env"]]
        assert len(names) == len(set(names)), pool
        assert (
            _env(gateway_consumer)["KG_LLM_API_KEY"]["valueFrom"]["secretKeyRef"]["name"]
            == (litellm["virtualKeySecret"]["name"])
        )
        provider_consumer = container(without_gateway, pool)
        assert _env(provider_consumer)["KG_LLM_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == (
            "provider"
        )


def test_the_validator_is_configured_exactly_like_what_it_validates() -> None:
    """Emit one env block for workers and for the Job that preflights them.

    The bootstrap Job runs preflight against the same profile the workers run.
    Configured separately, the two have drifted in both directions: the Job has
    passed a profile the workers could not execute, and failed one they could.
    """

    rendered = _render(_SERVING)
    bootstrap = _first_declared(_bootstrap(rendered))

    for pool in _WORKER_POOLS:
        worker = _first_declared(_worker(rendered, pool))
        for name in _CONSUMER_ENV:
            assert worker[name] == bootstrap[name], (pool, name)
    # The Job must not be left pointing at an engine the consumers no longer use.
    for entry in bootstrap.values():
        assert _VLLM not in entry.get("value", "")


def test_no_consumer_can_still_address_an_engine_directly() -> None:
    """Leave one endpoint key, so the placement layer stays replaceable."""

    schema = _schema()
    rendered = _render(_SERVING[1:], values=(_PUBLIC_EXAMPLE,))

    assert schema["additionalProperties"] is False
    assert "localModelServing" not in schema["properties"]
    assert "localModelServing" not in _load_yaml(_PUBLIC_EXAMPLE)
    assert not any(
        "internalTrafficPolicy" in doc["spec"] for doc in rendered if doc["kind"] == "Service"
    )
    consumers = [_worker(rendered, pool) for pool in _WORKER_POOLS] + [_bootstrap(rendered)]
    for consumer in consumers:
        for entry in consumer["env"]:
            assert _VLLM not in entry.get("value", ""), (consumer["name"], entry["name"])


def test_provider_secret_import_is_an_explicit_credential_allowlist() -> None:
    """Prevent unrelated Secret keys from silently replacing reviewed provider settings."""

    values = _values()
    rendered = _render(("providerSecret.name=provider",))

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
        {"name": "AWS_ACCESS_KEY_ID", "key": "AWS_ACCESS_KEY_ID", "optional": True},
        {"name": "AWS_SECRET_ACCESS_KEY", "key": "AWS_SECRET_ACCESS_KEY", "optional": True},
    ]
    # A mapping without `optional` would render null and be applied as a hard
    # requirement, so the item must spell all three keys.
    assert "missing property 'optional'" in _fails(
        ("providerSecret.env[0].name=KG_X", "providerSecret.env[0].key=KG_X")
    )
    allowlisted = {mapping["name"] for mapping in values["providerSecret"]["env"]}
    # The control plane lists sources and preflights with the same credentials
    # the workers will run with; a bucket it cannot see, they cannot read.
    control_plane = _container(
        _pod(_one(rendered, "Deployment", f"{_FULLNAME}-app")), "control-plane"
    )
    consumers = [_worker(rendered, pool) for pool in _WORKER_POOLS] + [
        _bootstrap(rendered),
        control_plane,
    ]
    for consumer in consumers:
        assert "envFrom" not in consumer, consumer["name"]
        imported = {
            entry["name"]
            for entry in consumer["env"]
            if entry.get("valueFrom", {}).get("secretKeyRef", {}).get("name") == "provider"
        }
        assert imported, consumer["name"]
        assert imported <= allowlisted, consumer["name"]


def test_ontology_is_a_portable_chart_managed_deployment_input() -> None:
    """Mount ontology content without baking repository datasets into worker images."""

    values = _values()
    inline = _render(("ontology.content=entities: []",))
    external = _render(("ontology.existingConfigMap=site-ontology",))

    assert values["ontology"] == {
        "existingConfigMap": "",
        "key": "ontology.yaml",
        "content": "",
    }
    assert _one(inline, "ConfigMap", f"{_FULLNAME}-ontology")["data"] == {
        "ontology.yaml": "entities: []\n"
    }
    assert not [doc for doc in external if doc["metadata"]["name"] == f"{_FULLNAME}-ontology"]
    for rendered, configmap in ((inline, f"{_FULLNAME}-ontology"), (external, "site-ontology")):
        for consumer in [_worker(rendered, pool) for pool in _WORKER_POOLS] + [
            _bootstrap(rendered)
        ]:
            assert _env(consumer)["KG_ONTOLOGY_PROFILE"]["value"] == (
                "/etc/flakegraph-ontology/ontology.yaml"
            )
        for pool in _WORKER_POOLS:
            deployment = _one(rendered, "Deployment", f"{_FULLNAME}-{pool}")
            assert "checksum/ontology" in deployment["spec"]["template"]["metadata"]["annotations"]
            volumes = {volume["name"]: volume for volume in _pod(deployment)["volumes"]}
            assert volumes["ontology"]["configMap"]["name"] == configmap


def test_worker_stage_names_follow_the_distributed_dag() -> None:
    """Keep typed Helm values aligned with role-specific worker pools."""

    values = _values()
    schema = _schema()
    stage_enum = schema["$defs"]["worker"]["properties"]["stages"]["items"]["enum"]
    rendered = _render(())

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
    # Every stage is served by exactly one pool, and no pool serves a stage the
    # schema does not know.
    served = [stage for pool in _WORKER_POOLS for stage in _stages(_worker(rendered, pool))]
    assert sorted(served) == sorted(stage_enum)


def test_mutable_worker_image_tags_are_refreshed_on_rollout() -> None:
    """Prevent node caches from retaining an older binary after a chart upgrade."""

    values = _values()

    assert values["image"]["pullPolicy"] == "Always"
    assert values["spark"]["image"]["pullPolicy"] == "Always"
    assert values["modelServing"]["image"]["pullPolicy"] == "IfNotPresent"


def test_workers_seed_preloaded_models_into_their_writable_cache() -> None:
    """Prevent the read-only runtime cache mount from hiding baked checkpoints."""

    rendered = _render(_SPARK)

    for pool in _WORKER_POOLS:
        deployment = _one(rendered, "Deployment", f"{_FULLNAME}-{pool}")
        seed = _container(_pod(deployment), "seed-provider-cache", "initContainers")
        (script,) = seed["args"]
        # The init container runs as an unprivileged image user. Recursive copy
        # is sufficient; archive mode would also preserve timestamps/ownership
        # and can fail against an EmptyDir prepared by Kubernetes securityContext
        # settings.
        assert "cp -R " in script
        assert "cp -a " not in script
        assert seed["resources"]["limits"]["memory"] == "1Gi"
        assert seed["volumeMounts"] == [{"name": "cache", "mountPath": "/cache"}]
        worker = _worker(rendered, pool)
        mounts = {mount["name"]: mount["mountPath"] for mount in worker["volumeMounts"]}
        if pool == "finalize":
            # The Spark image's baked cache belongs to executor UID 185, so the
            # driver reads its copy through a neutral mount, not another home.
            assert "/home/spark/.cache/." in script
            assert mounts["cache"] == "/cache"
            assert _env(worker)["HF_HOME"]["value"] == "/cache/huggingface"
            assert _env(worker)["SENTENCE_TRANSFORMERS_HOME"]["value"] == (
                "/cache/sentence_transformers"
            )
        else:
            assert "/home/kgprocessor/.cache/." in script
            assert mounts["cache"] == "/home/kgprocessor/.cache"


def test_spark_workers_reserve_memory_for_python_provider_processes() -> None:
    """Keep executor heap and non-heap memory explicit across Helm boundaries."""

    finalize = _env(_worker(_render(_SPARK), "finalize"))

    assert finalize["KG_DISTRIBUTED_SPARK_EXECUTOR_MEMORY"]["value"] == "8g"
    assert finalize["KG_DISTRIBUTED_SPARK_EXECUTOR_MEMORY_OVERHEAD"]["value"] == "8g"


def test_the_public_fleet_example_renders_against_the_charts_own_schema() -> None:
    """The example is the documented path onto a fleet, so it has to install."""

    rendered = _render(_SERVING[1:], values=(_PUBLIC_EXAMPLE,))

    engines = _one(rendered, "StatefulSet", _VLLM)
    assert engines["spec"]["replicas"] == 4
    env = _env(_worker(rendered, "finalize"))
    assert env["KG_DISTRIBUTED_FINALIZATION_ENGINE"]["value"] == "spark"
    assert env["KG_DISTRIBUTED_ARTIFACT_ACCESS_KEY_ID"]["valueFrom"]["secretKeyRef"] == {
        "name": "flakegraph-artifacts",
        "key": "access-key-id",
    }


def test_worker_pools_autoscale_from_dependency_aware_postgres_demand() -> None:
    """Release drained worker resources without scaling away active task leases."""

    values = _values()
    external_database = _render(())
    bundled_database = _render(("database.cloudNativePG.enabled=true",))
    demand_view = next(
        statement
        for statement in _SCHEMA_STATEMENTS
        if "VIEW flakegraph_worker_demand" in statement
    )

    assert values["autoscaling"]["pollingIntervalSeconds"] == 5
    assert values["autoscaling"]["cooldownPeriodSeconds"] == 30
    assert values["workers"]["prepare"]["autoscaling"]["minReplicas"] == 0
    assert values["workers"]["extract"]["autoscaling"]["maxReplicas"] == 32
    assert values["workers"]["finalize"]["autoscaling"]["maxReplicas"] == 1
    assert values["distributed"]["leaseSeconds"] == 300
    for pool in _WORKER_POOLS:
        scaler = _one(external_database, "ScaledObject", f"{_FULLNAME}-{pool}")
        pool_values = values["workers"][pool]["autoscaling"]
        assert scaler["spec"]["scaleTargetRef"]["name"] == f"{_FULLNAME}-{pool}"
        assert scaler["spec"]["minReplicaCount"] == pool_values["minReplicas"]
        assert scaler["spec"]["maxReplicaCount"] == pool_values["maxReplicas"]
        assert scaler["spec"]["advanced"]["restoreToOriginalReplicaCount"] is True
        for trigger in scaler["spec"]["triggers"]:
            assert trigger["type"] == "postgresql"
            assert trigger["metadata"]["connectionFromEnv"] == "KG_DISTRIBUTED_DATABASE_URL"
            query = " ".join(trigger["metadata"]["query"].split())
            assert "FROM flakegraph_worker_demand" in query
            # Bounded per pool, so a million queued rows ask for maxReplicas, not
            # a million workers.
            assert "SUM(LEAST(" in query
            assert str(pool_values["maxReplicas"] * pool_values["targetTasksPerReplica"]) in query
            for stage in _stages(_worker(external_database, pool)):
                assert f"'{stage}'" in query
        # KEDA runs outside the namespace, so the bundled database is addressed
        # by its fully qualified service and a password rather than the URI.
        bundled = _one(bundled_database, "ScaledObject", f"{_FULLNAME}-{pool}")
        for trigger in bundled["spec"]["triggers"]:
            assert trigger["metadata"]["passwordFromEnv"] == "KG_DISTRIBUTED_DATABASE_PASSWORD"
            assert "connectionFromEnv" not in trigger["metadata"]
        # Offline there is no existing Deployment to look up, so a fresh install
        # seeds each pool at its declared minimum rather than Kubernetes' one.
        deployment = _one(external_database, "Deployment", f"{_FULLNAME}-{pool}")
        assert deployment["spec"]["replicas"] == pool_values["minReplicas"]
    # The view the scaler reads exposes exactly the columns the query names.
    for column in ("stage", "desired_workers", "priority_band"):
        assert column in demand_view


def test_each_pool_scales_on_each_priority_band_independently() -> None:
    """Let a small urgent run raise a drained pool without a bulk backlog.

    With one combined trigger, demand from an urgent run is indistinguishable
    from demand from a queue of a hundred thousand bulk tasks.
    """

    rendered = _render(())

    for pool in _WORKER_POOLS:
        triggers = _one(rendered, "ScaledObject", f"{_FULLNAME}-{pool}")["spec"]["triggers"]
        assert [trigger["name"] for trigger in triggers] == [
            f"{pool}-interactive",
            f"{pool}-bulk",
        ]
        for band, trigger in zip(("interactive", "bulk"), triggers, strict=True):
            query = " ".join(trigger["metadata"]["query"].split())
            assert f"WHERE priority_band = '{band}'" in query
            # The cap stays inside each band, so one band cannot spend the other's.
            assert query.index("SUM(LEAST(") < query.index("WHERE priority_band")


def test_database_schema_is_bootstrapped_before_a_helm_release_is_ready() -> None:
    """Make KEDA's worker-demand view deterministic on fresh installs and upgrades."""

    values = _values()
    rendered = _render(())
    job = _one(rendered, "Job", f"{_FULLNAME}-database-bootstrap")
    initialize = _bootstrap(rendered)
    (script,) = initialize["args"]

    assert values["database"]["bootstrap"]["activeDeadlineSeconds"] >= 600
    assert job["metadata"]["annotations"]["helm.sh/hook"] == "post-install,post-upgrade"
    assert job["metadata"]["annotations"]["helm.sh/hook-delete-policy"] == (
        "before-hook-creation,hook-succeeded"
    )
    assert (
        job["spec"]["activeDeadlineSeconds"]
        == values["database"]["bootstrap"]["activeDeadlineSeconds"]
    )
    commands = [line.strip() for line in script.splitlines() if not line.strip().startswith("#")]
    assert commands[0].startswith("flakegraph preflight --deployment-worker")
    assert commands[1].startswith("exec flakegraph distributed init")
    assert _env(initialize)["KG_DISTRIBUTED_DATABASE_URL"]["valueFrom"]["secretKeyRef"] == {
        "name": values["database"]["secretName"],
        "key": values["database"]["secretKey"],
    }


def test_operator_notes_describe_the_selected_database_bootstrap_mode() -> None:
    """Helm output must not claim a disabled bootstrap hook completed."""

    with_hook = _notes(())
    without_hook = _notes(("database.bootstrap.enabled=false",))

    assert "The database bootstrap hook completed" in with_hook
    assert "Database bootstrap was disabled" not in with_hook
    assert "Database bootstrap was disabled" in without_hook
    assert "The database bootstrap hook completed" not in without_hook


def test_bundled_database_has_capacity_for_documented_fleet_workers() -> None:
    """Reserve database sessions for workers, autoscaling, and operations."""

    cloud_native_pg = _values()["database"]["cloudNativePG"]
    cluster = _one(
        _render(("database.cloudNativePG.enabled=true",)),
        "Cluster",
        cloud_native_pg["clusterName"],
    )

    assert cloud_native_pg["maxConnections"] >= 300
    assert cluster["spec"]["postgresql"]["parameters"]["max_connections"] == str(
        cloud_native_pg["maxConnections"]
    )


def test_scheduling_priorities_preserve_models_and_release_workers_for_spark() -> None:
    """Make the extraction-to-finalization handoff resilient to autoscaler delay."""

    values = _values()
    priorities = values["scheduling"]["priorityClasses"]
    rendered = _render(_SPARK)
    prefix = f"fleet-{_FULLNAME}"

    # The serving plane outranks the workers deliberately: it is the path every
    # consumer takes, so leaving it at the default priority lets a queue scale-up
    # preempt the gateway those same workers are trying to reach.
    assert (
        priorities["workerValue"]
        < priorities["sparkValue"]
        < priorities["servingValue"]
        < priorities["modelValue"]
    )
    # Helm prints a value of a million as 1e+06, which the API server reads
    # back as the integer; compare numerically rather than by spelling.
    classes = {
        doc["metadata"]["name"]: float(doc["value"])
        for doc in rendered
        if doc["kind"] == "PriorityClass"
    }
    assert classes == {
        f"{prefix}-worker": priorities["workerValue"],
        f"{prefix}-spark": priorities["sparkValue"],
        f"{prefix}-serving": priorities["servingValue"],
        f"{prefix}-model": priorities["modelValue"],
    }
    placed = {
        name: _pod(_one(rendered, kind, name))["priorityClassName"]
        for kind, name in (
            ("Deployment", f"{_FULLNAME}-litellm"),
            ("Deployment", _ROUTER),
            ("Deployment", f"{_FULLNAME}-ocr"),
            ("StatefulSet", f"{_FULLNAME}-mineru"),
            ("Deployment", f"{_FULLNAME}-prepare"),
            ("Deployment", f"{_FULLNAME}-extract"),
            ("Deployment", f"{_FULLNAME}-finalize"),
            ("StatefulSet", _VLLM),
        )
    }
    assert placed == {
        f"{_FULLNAME}-litellm": f"{prefix}-serving",
        _ROUTER: f"{prefix}-serving",
        f"{_FULLNAME}-ocr": f"{prefix}-serving",
        f"{_FULLNAME}-mineru": f"{prefix}-serving",
        f"{_FULLNAME}-prepare": f"{prefix}-worker",
        f"{_FULLNAME}-extract": f"{prefix}-worker",
        f"{_FULLNAME}-finalize": f"{prefix}-spark",
        _VLLM: f"{prefix}-model",
    }
    assert _executor_template(rendered)["spec"]["priorityClassName"] == f"{prefix}-spark"
    assert values["terminationGracePeriodSeconds"] > 3600


def test_spark_role_allows_native_executor_creation_and_cleanup() -> None:
    """Permit Spark to remove executor collections when a driver terminates."""

    (rule,) = _one(_render(_SPARK), "Role", f"{_FULLNAME}-spark")["rules"]

    assert {"pods", "services", "configmaps", "persistentvolumeclaims"} <= set(rule["resources"])
    assert {"create", "delete", "deletecollection"} <= set(rule["verbs"])


def test_object_storage_cannot_silently_select_an_unavailable_spark_runtime() -> None:
    """Reject a fleet configuration that would fail only after extraction ends."""

    refusal = _fails(("artifactStorage.uri=s3://artifacts",))

    assert "spark.enabled must be true when artifactStorage.uri is configured" in refusal
    _render(("artifactStorage.uri=s3://artifacts", "spark.enabled=true"))


def test_a_draft_model_path_must_have_somewhere_to_come_from() -> None:
    """Naming a path is not obtaining the file at it.

    A replica scheduled onto a node where nobody placed the draft model starts,
    fails to load, and crash-loops with an error about an invalid repository id -
    which reads like a typo rather than a missing artifact. The chart refuses to
    render that, and can seed the volume itself.
    """

    values = _values()["modelServing"]["server"]
    seed = values["draftModelSeed"]

    # The shipped default names a path, so it must also say where it comes from.
    assert values["speculativeDraftModel"].startswith("/")
    assert seed["image"] == ""
    assert seed["providedExternally"] is False

    # Refuses to render a path nothing supplies...
    refusal = _fails(("modelServing.enabled=true",))
    assert "modelServing.server.speculativeDraftModel is the path" in refusal
    assert "draftModelSeed.image" in refusal
    assert "draftModelSeed.providedExternally=true" in refusal

    # ...and seeds the volume when an image carries the model.
    seeded = _render(
        ("modelServing.enabled=true", "modelServing.server.draftModelSeed.image=example/dflash2:1")
    )
    pod = _pod(_one(seeded, "StatefulSet", _VLLM))
    seeder = _container(pod, "seed-draft-model", "initContainers")
    (script,) = seeder["args"]
    commands = [line.strip() for line in script.splitlines() if not line.strip().startswith("#")]
    assert seeder["image"] == "example/dflash2:1"
    assert seeder["volumeMounts"] == [{"name": "models", "mountPath": "/models"}]
    assert f'target="{values["speculativeDraftModel"]}"' in commands
    # The copy lands beside the target and is renamed last, so an interrupted
    # copy is never mistaken for a complete one.
    assert 'cp -r "/dflash2" "$target.partial"' in commands
    assert commands[-1] == 'mv "$target.partial" "$target"'
    assert "initContainers" not in _pod(_one(_render(_SERVING), "StatefulSet", _VLLM))


def test_the_draft_model_seed_is_constrained_by_the_schema() -> None:
    """An operator mistyping this should hear about it at install, not at load."""

    schema = _schema()
    seed = schema["properties"]["modelServing"]["properties"]["server"]["properties"][
        "draftModelSeed"
    ]

    assert seed["additionalProperties"] is False
    assert set(seed["properties"]) == {
        "image",
        "pullPolicy",
        "sourcePath",
        "providedExternally",
    }


def test_cache_events_are_published_on_a_topic_the_picker_subscribes_to() -> None:
    """Publishing on the default empty topic reaches nobody.

    The picker subscribes with the ZMQ prefix filter "kv@" and parses the model
    name out of ``kv@<pod>@<model>``. vLLM's own default topic is the empty
    string, which that filter discards - so a deployment can look configured,
    publish continuously, and have every event dropped before it is read.
    """

    values = _values()["modelServing"]
    engine = _container(_pod(_one(_render(_SERVING), "StatefulSet", _VLLM)), "vllm")
    events = json.loads(_args(engine)["--kv-events-config"])

    assert events["enable_kv_cache_events"] is True
    assert events["topic"] == f"kv@$(POD_IP):{values['kvEvents']['port']}@{values['model']['name']}"
    # The topic interpolates POD_IP, which Kubernetes expands into args only when
    # the same container declares it.
    assert _env(engine)["POD_IP"]["valueFrom"]["fieldRef"]["fieldPath"] == "status.podIP"


def test_exact_prefix_routing_names_its_producer_and_matches_block_size() -> None:
    """Both halves are silent when wrong, so both are pinned here.

    A prefix-cache-scorer with no producer named gets the approximate producer
    auto-created underneath it, and reports confident scores about a cache it has
    never seen. A block size that disagrees with the engine's hashes differently
    sized blocks, which agree about nothing.
    """

    values = _values()["modelServing"]
    router = _one(_render(_SERVING), "ConfigMap", _ROUTER)["data"]
    config = yaml.safe_load(router["default-plugins.yaml"])
    plugins = {plugin["type"]: plugin for plugin in config["plugins"]}

    assert plugins["prefix-cache-scorer"]["parameters"] == {
        "prefixMatchInfoProducerName": "precise-prefix-cache-producer"
    }
    assert plugins["token-producer"]["parameters"]["modelName"] == values["model"]["name"]
    assert "endpoint-notification-source" in plugins
    # Without this wiring the producer never opens a subscriber to any pod.
    assert {
        "pluginRef": "endpoint-notification-source",
        "extractors": [{"pluginRef": "precise-prefix-cache-producer"}],
    } in config["dataLayer"]["sources"]
    producer = plugins["precise-prefix-cache-producer"]["parameters"]
    assert (
        producer["tokenProcessorConfig"]["blockSizeTokens"] == values["kvEvents"]["blockSizeTokens"]
    )
    # vLLM's default block size is 16; the picker must hash the same width.
    assert values["kvEvents"]["blockSizeTokens"] == 16


def _first_declared(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a container's environment by the first declaration of each name.

    The shared consumer block is emitted before any provider Secret mapping,
    so the first declaration is the one that block owns.
    """

    declared: dict[str, dict[str, Any]] = {}
    for entry in container["env"]:
        declared.setdefault(entry["name"], entry)
    return declared


def _stages(worker: dict[str, Any]) -> list[str]:
    """Return the stages a worker container is started for."""

    args: list[str] = worker["args"]
    return [args[index + 1] for index, arg in enumerate(args) if arg == "--stage"]


def _worker(rendered: list[dict[str, Any]], pool: str) -> dict[str, Any]:
    """Return one worker pool's worker container."""

    return _container(_pod(_one(rendered, "Deployment", f"{_FULLNAME}-{pool}")), "worker")


def _bootstrap(rendered: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the database bootstrap Job's only container."""

    return _container(_pod(_one(rendered, "Job", f"{_FULLNAME}-database-bootstrap")), "initialize")


def _executor_template(rendered: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the Pod template the Spark driver hands to its executors."""

    configmap = _one(rendered, "ConfigMap", f"{_FULLNAME}-spark-executor-template")
    template: dict[str, Any] = yaml.safe_load(configmap["data"]["executor-pod-template.yaml"])
    return template
