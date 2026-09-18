"""The fleet checks a submit host runs before queueing a run nothing would claim."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import yaml

from kg_processor.fleet.kubectl import ClusterTarget
from kg_processor.fleet.preflight import fleet_preflight, placeholder_names
from kg_processor.fleet.profile import fleet_profile, semantic_mismatches

_RUN: dict[str, Any] = {
    "ocr": {"provider": "fallback"},
    "llm": {
        "provider": "vllm_local",
        "model": "unsloth/Qwen3.8-27B-NVFP4",
        "endpoint": "http://localhost:8000/v1",
        "api_key": "${KG_LLM_API_KEY}",
    },
    "embedding": {
        "provider": "sentence_transformers",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
    },
}


def _use(monkeypatch: pytest.MonkeyPatch, kubectl: object) -> None:
    for module in (
        "kg_processor.fleet.kubectl",
        "kg_processor.fleet.preflight",
        "kg_processor.fleet.profile",
    ):
        monkeypatch.setattr(f"{module}.kubectl_json", kubectl, raising=False)


def _run(**changes: Any) -> dict[str, Any]:
    run = {section: dict(values) for section, values in _RUN.items()}
    for section, values in changes.items():
        run[section] = values
    return run


def _fleet_kubectl_fixture(
    *,
    profile_ocr: str,
    model_servers: int,
    output_credential: str | None = None,
    service_account_available: bool = True,
    ontology: Mapping[str, object] | None = None,
) -> object:
    """Return a deterministic kubectl adapter for fleet-preflight unit tests."""

    deployments = []
    scaled_objects = []
    for component in ("worker-prepare", "worker-extract", "worker-finalize"):
        name = f"flakegraph-{component}"
        deployments.append(
            {
                "metadata": {
                    "name": name,
                    "labels": {"app.kubernetes.io/component": component},
                },
                "spec": {
                    "replicas": 0,
                    "template": {
                        "spec": {
                            "serviceAccountName": "flakegraph",
                            "containers": [
                                {
                                    "image": "flakegraph:1.0.0",
                                    "env": (
                                        [{"name": output_credential}]
                                        if component == "worker-finalize" and output_credential
                                        else []
                                    ),
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "config",
                                    "configMap": {"name": "flakegraph-runtime"},
                                },
                                *(
                                    [
                                        {
                                            "name": "ontology",
                                            "configMap": {"name": "flakegraph-ontology"},
                                        }
                                    ]
                                    if ontology
                                    else []
                                ),
                            ],
                        }
                    },
                },
            }
        )
        scaled_objects.append(
            {
                "spec": {"scaleTargetRef": {"name": name}},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            }
        )

    def kubectl(arguments: list[str], *, target: object, raw: bool = False) -> object:
        if arguments == ["config", "current-context"] and raw:
            return "flakegraph-fleet\n"
        if arguments[:2] == ["get", "nodes"]:
            return {
                "items": [
                    {
                        "metadata": {"labels": {"kubernetes.io/arch": "arm64"}},
                        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                    }
                ]
            }
        resource = arguments[1]
        if resource == "deployments":
            response: object = {"items": deployments}
        elif resource == "scaledobjects.keda.sh":
            response = {"items": scaled_objects}
        elif resource == "configmap" and arguments[2] == "flakegraph-ontology":
            response = {"data": {"ontology.yaml": yaml.safe_dump(ontology or {})}}
        elif resource == "configmap":
            response = {
                "data": {
                    "config.yaml": yaml.safe_dump(
                        {
                            "ocr": {"provider": profile_ocr},
                            "llm": {
                                "provider": "vllm_local",
                                "model": "unsloth/Qwen3.8-27B-NVFP4",
                            },
                            "embedding": {
                                "provider": "sentence_transformers",
                                "model": "sentence-transformers/all-MiniLM-L6-v2",
                            },
                        }
                    )
                }
            }
        elif resource == "statefulsets":
            response = {
                "items": (
                    [
                        {
                            "metadata": {
                                "labels": {"app.kubernetes.io/component": "model-serving"}
                            },
                            "spec": {"replicas": model_servers},
                            "status": {"readyReplicas": model_servers},
                        }
                    ]
                    if model_servers
                    else []
                )
            }
        elif resource == "secrets":
            response = {
                "items": ([{"data": {output_credential: "encoded"}}] if output_credential else [])
            }
        elif resource == "serviceaccounts":
            response = {
                "items": (
                    [{"metadata": {"name": "flakegraph"}}] if service_account_available else []
                )
            }
        else:
            raise AssertionError(f"Unexpected kubectl arguments: {arguments}")
        return response

    return kubectl


def test_fleet_preflight_accepts_keda_workers_and_ready_local_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat scale-to-zero workers as healthy when KEDA and model serving are ready."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback", model_servers=1, output_credential="KG_LLM_API_KEY"
        ),
    )

    result = fleet_preflight(_run(), "flakegraph", ClusterTarget())

    assert result["ok"] is True, result["errors"]
    assert "Chart-managed vLLM model servers are ready: 1/1" in result["checks"]
    assert "Fleet credential is available: KG_LLM_API_KEY" in result["checks"]


def test_fleet_preflight_rejects_a_run_whose_ontology_no_worker_mounts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unmatched ontology leaves a run queued forever rather than failing it."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback", model_servers=1, output_credential="KG_LLM_API_KEY"
        ),
    )

    result = fleet_preflight(
        _run(ontology={"profile": {"name": "general", "mode": "hybrid"}}),
        "flakegraph",
        ClusterTarget(),
    )

    assert result["ok"] is False
    assert any("ontology is not deployed" in item for item in result["errors"])


def test_fleet_preflight_rejects_an_ontology_that_differs_from_the_fleet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matching by name is not enough; changed ontology bytes change the graph."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback",
            model_servers=1,
            output_credential="KG_LLM_API_KEY",
            ontology={"name": "general", "mode": "strict"},
        ),
    )

    result = fleet_preflight(
        _run(ontology={"profile": {"name": "general", "mode": "hybrid"}}),
        "flakegraph",
        ClusterTarget(),
    )

    assert result["ok"] is False
    assert any("differs from the one fleet workers mount" in item for item in result["errors"])


def test_fleet_preflight_accepts_a_matching_ontology(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run the operator actually wants must still pass."""

    profile: dict[str, object] = {"name": "general", "mode": "hybrid"}
    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback",
            model_servers=1,
            output_credential="KG_LLM_API_KEY",
            ontology=profile,
        ),
    )

    result = fleet_preflight(_run(ontology={"profile": profile}), "flakegraph", ClusterTarget())

    assert result["ok"] is True, result["errors"]


def test_fleet_preflight_rejects_missing_worker_service_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a fleet that cannot create autoscaled workers before accepting work."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback", model_servers=1, service_account_available=False
        ),
    )

    result = fleet_preflight(_run(), "flakegraph", ClusterTarget())

    assert result["ok"] is False
    assert any("references missing service account" in error for error in result["errors"])


def test_fleet_preflight_explains_profile_and_model_serving_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block runs that the homogeneous worker profile cannot execute consistently."""

    _use(monkeypatch, _fleet_kubectl_fixture(profile_ocr="builtin_text", model_servers=0))

    result = fleet_preflight(_run(), "flakegraph", ClusterTarget())

    assert result["ok"] is False
    assert (
        "ocr.provider does not match fleet workers: selected 'fallback', deployed 'builtin_text'"
        in result["errors"]
    )
    assert (
        "Selected vLLM endpoint is node-local, but the fleet has no ready model server (0/0)"
        in result["errors"]
    )


def test_fleet_preflight_requires_output_secret_on_finalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a Snowflake destination that the finalizer cannot authenticate to."""

    _use(
        monkeypatch,
        _fleet_kubectl_fixture(
            profile_ocr="fallback", model_servers=1, output_credential="KG_LLM_API_KEY"
        ),
    )

    result = fleet_preflight(
        _run(snowflake={"account": "account", "password": "${SNOWFLAKE_PASSWORD}"}),
        "flakegraph",
        ClusterTarget(),
    )

    assert result["ok"] is False
    assert (
        "Fleet credential is missing from namespace flakegraph: SNOWFLAKE_PASSWORD"
        in result["errors"]
    )
    assert (
        "Finalizer does not map output credential environment variable: SNOWFLAKE_PASSWORD"
        in result["errors"]
    )


def test_semantic_mismatches_compare_only_what_both_sides_state() -> None:
    """A key the fleet omits takes the runtime default on both sides; placeholders are transport."""

    deployed = {
        "ocr": {
            "provider": "fallback",
            "mineru_method": "ocr",
            "mineru_api_url": "${KG_MINERU_API_URL}",
        }
    }

    assert semantic_mismatches(
        {"ocr": {"provider": "fallback", "mineru_method": "auto"}}, deployed
    ) == ["ocr.mineru_method does not match fleet workers: selected 'auto', deployed 'ocr'"]
    assert (
        semantic_mismatches(
            {"ocr": {"provider": "fallback", "mineru_api_url": "http://x"}}, deployed
        )
        == []
    )
    assert semantic_mismatches({"ocr": {"provider": "fallback"}}, deployed) == []


def test_placeholder_names_read_only_credential_fields() -> None:
    run = {
        "llm": {"api_key": "${KG_LLM_API_KEY}", "endpoint": "${KG_LLM_ENDPOINT}"},
        "snowflake": {"password": "${SNOWFLAKE_PASSWORD}", "account": "${SNOWFLAKE_ACCOUNT}"},
    }

    assert placeholder_names(run) == {"KG_LLM_API_KEY", "SNOWFLAKE_PASSWORD"}


def test_fleet_profile_reports_what_the_workers_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    """A submit host composes runs against the mounted profile and ontology."""

    ontology: Mapping[str, object] = {"name": "general", "mode": "hybrid"}
    _use(
        monkeypatch,
        _fleet_kubectl_fixture(profile_ocr="fallback", model_servers=1, ontology=ontology),
    )

    profile = fleet_profile(ClusterTarget(), "flakegraph")

    assert profile["config_map"] == "flakegraph-runtime"
    assert profile["config"]["ocr"] == {"provider": "fallback"}
    assert profile["ontology"] == dict(ontology)
    assert profile["parsing_endpoint"] is None
