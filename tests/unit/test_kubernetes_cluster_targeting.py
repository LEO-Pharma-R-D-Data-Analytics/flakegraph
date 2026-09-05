"""The selected cluster must reach the commands that act on it.

Two mechanisms carry the selection, and each fails silently on its own: fleet
control commands are subprocesses that need an environment, while kubectl takes
its cluster from an argument. A selection that reaches neither leaves the picker
appearing to work while every read and write goes to the ambient cluster.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from flakegraph_app.backends import kubernetes as backend_module
from flakegraph_app.backends.kubernetes import KubernetesBackend
from flakegraph_app.cluster_catalog import ClusterProfile, ClusterTarget, save_cluster


def _backend(tmp_path: Path) -> KubernetesBackend:
    backend = KubernetesBackend(tmp_path)
    save_cluster(
        backend.state_root,
        ClusterProfile(
            name="lab",
            namespace="lab-ns",
            context="lab-ctx",
            kubeconfig=str(tmp_path / "lab.kubeconfig"),
        ),
    )
    return backend


def test_a_registered_cluster_keeps_the_executable_reachable(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A subprocess environment without PATH cannot find the command to run.

    The cluster contributes a namespace and a kubeconfig; replacing the
    environment with only those leaves the fleet CLI unresolvable, and the
    resulting failure is reported as an empty run history rather than as a fault.
    """

    backend = _backend(tmp_path)
    captured: dict[str, Any] = {}

    def _record(command: Any, **kwargs: Any) -> Any:
        captured["env"] = kwargs.get("env")

        class _Result:
            returncode = 0
            stdout = '{"run": {"id": "run-1"}}'
            stderr = ""

        return _Result()

    monkeypatch.setattr("flakegraph_app.backends.kubernetes.subprocess.run", _record)
    monkeypatch.setattr(
        backend_module,
        "_fleet_database_environment",
        lambda namespace, target, base=None: _NullContext(dict(base or {})),
    )

    backend._run_json(["uv", "run", "flakegraph", "distributed", "status"])

    environment = captured["env"]
    assert environment is not None
    assert environment.get("PATH") == os.environ.get("PATH")
    assert environment["FLAKEGRAPH_APP_KUBERNETES_NAMESPACE"] == "lab-ns"
    assert environment["KUBECONFIG"].endswith("lab.kubeconfig")


def test_kubectl_is_pointed_at_the_selected_cluster(tmp_path: Path, monkeypatch: Any) -> None:
    """kubectl ignores the app's cluster choice unless it is passed the flag."""

    captured: dict[str, Any] = {}

    def _record(command: Any, **kwargs: Any) -> Any:
        captured["command"] = list(command)
        captured["env"] = kwargs.get("env")

        class _Result:
            returncode = 0
            stdout = "{}"
            stderr = ""

        return _Result()

    monkeypatch.setattr("flakegraph_app.backends.kubernetes.subprocess.run", _record)

    backend_module._kubectl_json(
        arguments=["get", "nodes", "-o", "json"],
        target=ClusterTarget(kubeconfig="/tmp/lab.yaml", context="lab-ctx"),
    )

    assert captured["command"][:3] == ["kubectl", "--context", "lab-ctx"]
    assert captured["env"]["KUBECONFIG"] == "/tmp/lab.yaml"
    assert captured["env"].get("PATH") == os.environ.get("PATH")


def test_fleet_access_follows_the_selected_namespace(tmp_path: Path, monkeypatch: Any) -> None:
    """Containment must track the chosen cluster, not one fixed environment value.

    Driven through the backend rather than the guard, so it fails if a caller
    ever validates against something other than the selected cluster: validating
    against an ambient default locks the fleet page out of every cluster
    registered on a namespace of its own.
    """

    backend = _backend(tmp_path)
    monkeypatch.setenv("FLAKEGRAPH_APP_KUBERNETES_NAMESPACE", "some-other-default")
    reads: list[list[str]] = []

    def _kubectl(arguments: Any, *, target: Any, raw: bool = False) -> Any:
        reads.append(list(arguments))
        return "ctx" if raw else {"items": []}

    monkeypatch.setattr(backend_module, "_kubectl_json", _kubectl)
    monkeypatch.setattr(backend_module, "_gpu_utilization_by_node", lambda *_: ({}, None))

    backend.cluster("lab-ns")

    assert reads, "the selected cluster's namespace must be accepted"
    assert any("lab-ns" in arguments for arguments in reads)

    with pytest.raises(RuntimeError, match="lab-ns"):
        backend.cluster("some-other-default")


class _NullContext:
    def __init__(self, value: dict[str, str]) -> None:
        self.value = value

    def __enter__(self) -> dict[str, str]:
        return self.value

    def __exit__(self, *_: object) -> None:
        return None
