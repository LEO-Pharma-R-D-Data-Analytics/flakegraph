"""Whether the fleet can take a run before the run is submitted.

Worker deployments are allowed to sit at zero replicas because KEDA activates
them from durable queue demand; a zero-replica deployment is accepted only
when its ScaledObject reports Ready. Provider and model selections must match
the immutable worker profile so one run cannot mix semantic profiles - and so
a run is never queued that nothing will ever claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import yaml

from kg_processor.fleet.kubectl import (
    ClusterTarget,
    component,
    config_map,
    current_context,
    kubectl_json,
    labelled,
)
from kg_processor.fleet.profile import (
    ENVIRONMENT_PLACEHOLDER,
    REQUIRED_WORKER_COMPONENTS,
    load_deployed_profile,
    mounted_config_map,
    semantic_mismatches,
)

CREDENTIAL_SECTIONS = ("ocr", "llm", "embedding", "snowflake")


def fleet_preflight(
    run: Mapping[str, Any],
    namespace: str,
    target: ClusterTarget,
) -> dict[str, Any]:
    """Validate capacity, worker controllers, and processing compatibility for one run."""

    checks: list[str] = []
    errors: list[str] = []
    context = current_context(target)
    nodes = kubectl_json(["get", "nodes", "-o", "json"], target=target).get("items", [])
    ready_nodes = sum(_node_ready(item) for item in nodes)
    _require(
        errors,
        checks,
        bool(nodes) and ready_nodes == len(nodes),
        f"Kubernetes context {context} has {ready_nodes}/{len(nodes)} ready nodes",
        f"Kubernetes context {context} has only {ready_nodes}/{len(nodes)} ready nodes",
    )

    deployments = labelled(target, namespace, "deployments")
    by_component = {component(item): item for item in deployments if component(item)}
    service_accounts = {
        str(item.get("metadata", {}).get("name", ""))
        for item in kubectl_json(
            ["get", "serviceaccounts", "-n", namespace, "-o", "json"], target=target
        ).get("items", [])
    }
    try:
        scaled_objects = labelled(target, namespace, "scaledobjects.keda.sh")
    except RuntimeError:
        scaled_objects = []
    scaled_by_target = {
        str(item.get("spec", {}).get("scaleTargetRef", {}).get("name", "")): item
        for item in scaled_objects
    }
    config_maps = _validate_worker_deployments(
        by_component, service_accounts, nodes, scaled_by_target, checks, errors
    )

    _require(
        errors,
        checks,
        len(config_maps) == 1,
        f"Worker pools share processing profile: {next(iter(config_maps), 'unknown')}",
        "Worker pools must mount one shared FlakeGraph processing ConfigMap",
    )
    if len(config_maps) == 1:
        config_name = next(iter(config_maps))
        deployed = load_deployed_profile(config_map(target, namespace, config_name))
        profile_errors = semantic_mismatches(run, deployed)
        profile_errors.extend(_ontology_mismatches(run, deployed, namespace, target, deployments))
        errors.extend(profile_errors)
        if not profile_errors:
            checks.append("Selected OCR, LLM, embedding, and ontology match fleet workers")

    llm = _section(run, "llm")
    if llm.get("provider") == "vllm_local" and _loopback_endpoint(llm.get("endpoint")):
        model_server = next(
            (
                item
                for item in labelled(target, namespace, "statefulsets")
                if component(item) == "model-serving"
            ),
            None,
        )
        desired = int(model_server.get("spec", {}).get("replicas", 0) or 0) if model_server else 0
        ready = (
            int(model_server.get("status", {}).get("readyReplicas", 0) or 0) if model_server else 0
        )
        # One ready engine serves; the router only sends work to engines that
        # are ready, so a fleet mid-rollout is slower, not unavailable.
        _require(
            errors,
            checks,
            model_server is not None and ready > 0,
            f"Chart-managed vLLM model servers are ready: {ready}/{desired}",
            "Selected vLLM endpoint is node-local, but the fleet has no ready "
            f"model server ({ready}/{desired})",
        )

    _validate_fleet_credentials(run, namespace, by_component, checks, errors, target)
    return {"ok": not errors, "checks": checks, "errors": errors}


def _validate_worker_deployments(
    by_component: Mapping[str, Mapping[str, Any]],
    service_accounts: set[str],
    nodes: Sequence[Mapping[str, Any]],
    scaled_by_target: Mapping[str, Mapping[str, Any]],
    checks: list[str],
    errors: list[str],
) -> set[str]:
    """Validate queue workers and return their mounted processing profiles."""

    config_maps: set[str] = set()
    for name_of in sorted(REQUIRED_WORKER_COMPONENTS):
        deployment = by_component.get(name_of)
        if deployment is None:
            errors.append(f"Kubernetes fleet is missing the {name_of} deployment")
            continue
        name = str(deployment.get("metadata", {}).get("name", name_of))
        image = _controller_image(deployment)
        _require(
            errors,
            checks,
            bool(image),
            f"{name_of} worker image is configured: {image}",
            f"{name_of} deployment has no worker image",
        )
        desired = int(deployment.get("spec", {}).get("replicas", 0) or 0)
        pod_spec = deployment.get("spec", {}).get("template", {}).get("spec", {})
        service_account = str(pod_spec.get("serviceAccountName") or "default")
        _require(
            errors,
            checks,
            service_account in service_accounts,
            f"{name_of} service account is available: {service_account}",
            f"{name_of} references missing service account: {service_account}",
        )
        selector = pod_spec.get("nodeSelector") or {}
        _require(
            errors,
            checks,
            not selector
            or any(
                _node_ready(node)
                and _labels_match(node.get("metadata", {}).get("labels", {}), selector)
                for node in nodes
            ),
            f"{name_of} has ready node capacity for selector {dict(selector)}",
            f"{name_of} has no ready node matching selector {dict(selector)}",
        )
        available = int(deployment.get("status", {}).get("availableReplicas", 0) or 0)
        if desired > 0:
            _require(
                errors,
                checks,
                available == desired,
                f"{name_of} worker replicas are available: {available}/{desired}",
                f"{name_of} worker replicas are unavailable: {available}/{desired}",
            )
        scaled = scaled_by_target.get(name)
        if desired == 0:
            _require(
                errors,
                checks,
                bool(scaled and _condition_true(scaled, "Ready")),
                f"{name_of} is ready for queue-driven autoscaling",
                f"{name_of} is scaled to zero without a Ready KEDA ScaledObject",
            )
        config_name = mounted_config_map(deployment)
        if config_name:
            config_maps.add(config_name)
    return config_maps


def _ontology_mismatches(
    run: Mapping[str, Any],
    deployed: Mapping[str, Any],
    namespace: str,
    target: ClusterTarget,
    deployments: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Report an ontology the fleet workers could not agree with.

    A submitted run carries its ontology inline so it describes itself wherever
    it executes, while workers mount theirs as a file; the two are compared by
    reading the mounted ConfigMap.
    """

    ontology = run.get("ontology")
    selected = (
        dict(ontology["profile"])
        if isinstance(ontology, Mapping) and isinstance(ontology.get("profile"), Mapping)
        else None
    )
    mounted_name = next(
        (name for item in deployments if (name := mounted_config_map(item, "ontology"))),
        None,
    )
    deployed_reference = deployed.get("ontology")
    has_deployed = bool(mounted_name) or (
        isinstance(deployed_reference, Mapping)
        and any(deployed_reference.get(key) for key in ("profile", "profile_path"))
    )
    if selected and not has_deployed:
        return [
            "Selected ontology is not deployed to the fleet: workers mount no ontology, "
            "so no worker can claim this run. Install the chart with "
            "--set-file ontology.content=<profile.yaml>."
        ]
    if has_deployed and not selected:
        return [
            "Fleet workers mount an ontology but this run selects none, "
            "so no worker can claim this run."
        ]
    if not selected or not mounted_name:
        return []
    data = config_map(target, namespace, mounted_name).get("data", {})
    text = next((value for value in data.values() if isinstance(value, str)), "")
    loaded = yaml.safe_load(text) or {} if text.strip() else {}
    if not isinstance(loaded, Mapping):
        return [f"Fleet ontology ConfigMap {mounted_name} does not contain a mapping"]
    if dict(loaded) != selected:
        return [
            f"Selected ontology differs from the one fleet workers mount ({mounted_name}), "
            "so no worker can claim this run."
        ]
    return []


def _validate_fleet_credentials(
    run: Mapping[str, Any],
    namespace: str,
    by_component: Mapping[str, Mapping[str, Any]],
    checks: list[str],
    errors: list[str],
    target: ClusterTarget,
) -> None:
    """Verify Secret presence and the finalizer's mapping for every credential a run names.

    A run names a credential as an environment placeholder rather than a value,
    so what is checked is that the namespace holds a key by that name and, for
    output credentials, that the finalizer is given it.
    """

    required = sorted(placeholder_names(run))
    if not required:
        return
    secrets = kubectl_json(["get", "secrets", "-n", namespace, "-o", "json"], target=target).get(
        "items", []
    )
    available = {str(key) for secret in secrets for key in secret.get("data", {})}
    for credential in required:
        _require(
            errors,
            checks,
            credential in available,
            f"Fleet credential is available: {credential}",
            f"Fleet credential is missing from namespace {namespace}: {credential}",
        )
    snowflake = _section(run, "snowflake")
    finalizer = _controller_environment_names(by_component.get("worker-finalize", {}))
    for credential in sorted(placeholder_names({"snowflake": snowflake})):
        _require(
            errors,
            checks,
            credential in finalizer,
            f"Finalizer receives output credential: {credential}",
            f"Finalizer does not map output credential environment variable: {credential}",
        )


def placeholder_names(run: Mapping[str, Any]) -> set[str]:
    """Return the environment variables a run's credential fields refer to."""

    names: set[str] = set()
    for section in CREDENTIAL_SECTIONS:
        values = run.get(section)
        if not isinstance(values, Mapping):
            continue
        for key, value in values.items():
            if not isinstance(value, str) or not ENVIRONMENT_PLACEHOLDER.fullmatch(value):
                continue
            if "key" in str(key) or "password" in str(key) or "token" in str(key):
                names.add(value[2:-1])
    return names


def _section(run: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = run.get(name)
    return value if isinstance(value, Mapping) else {}


def _require(
    errors: list[str], checks: list[str], condition: bool, ok_message: str, error_message: str
) -> None:
    """Append one fleet assertion to the same JSON contract as core preflight."""

    (checks if condition else errors).append(ok_message if condition else error_message)


def _node_ready(node: Mapping[str, Any]) -> bool:
    return _condition_true(node, "Ready")


def _condition_true(resource: Mapping[str, Any], condition_type: str) -> bool:
    return any(
        item.get("type") == condition_type and item.get("status") == "True"
        for item in resource.get("status", {}).get("conditions", [])
    )


def _controller_image(resource: Mapping[str, Any]) -> str | None:
    containers = resource.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    if not containers or not containers[0].get("image"):
        return None
    return str(containers[0]["image"])


def _controller_environment_names(resource: Mapping[str, Any]) -> set[str]:
    containers = resource.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
    if not containers:
        return set()
    return {str(item.get("name")) for item in containers[0].get("env", []) if item.get("name")}


def _labels_match(labels: Mapping[str, object], selector: Mapping[str, object]) -> bool:
    return all(str(labels.get(key)) == str(value) for key, value in selector.items())


def _loopback_endpoint(endpoint: object) -> bool:
    if not isinstance(endpoint, str) or not endpoint:
        return False
    return urlparse(endpoint).hostname in {"localhost", "127.0.0.1", "::1"}
