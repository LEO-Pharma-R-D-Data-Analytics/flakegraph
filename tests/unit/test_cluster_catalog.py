"""Registering, selecting and targeting Kubernetes clusters."""

from __future__ import annotations

from pathlib import Path

import pytest
from flakegraph_app.cluster_catalog import (
    ClusterProfile,
    read_catalog,
    remove_cluster,
    save_cluster,
    select_cluster,
    validate_cluster_name,
    validate_namespace,
)


def test_a_registered_cluster_survives_a_round_trip(tmp_path: Path) -> None:
    save_cluster(
        tmp_path,
        ClusterProfile(name="leo-spark", namespace="flakegraph", context="k3s-leo"),
    )

    catalog = read_catalog(tmp_path)

    assert [cluster.name for cluster in catalog.clusters] == ["leo-spark"]
    assert catalog.selected == "leo-spark"


def test_the_only_cluster_is_active_without_being_chosen(tmp_path: Path) -> None:
    """A catalog with one cluster is unambiguous; asking would have one answer."""

    save_cluster(tmp_path, ClusterProfile(name="only"))
    catalog = read_catalog(tmp_path)
    catalog.selected = ""

    active = catalog.active()
    assert active is not None
    assert active.name == "only"


def test_the_environment_targets_the_selected_cluster(tmp_path: Path) -> None:
    """A selection that commands ignore would only appear to work."""

    cluster = ClusterProfile(
        name="lab", namespace="lab-ns", context="lab-ctx", kubeconfig="/tmp/kubeconfig"
    )

    environment = cluster.environment()

    assert environment["FLAKEGRAPH_APP_KUBERNETES_NAMESPACE"] == "lab-ns"
    assert environment["KUBECONFIG"] == "/tmp/kubeconfig"


def test_an_unset_context_is_omitted_rather_than_blanked(tmp_path: Path) -> None:
    """An empty context means "whatever the kubeconfig selects", not "no context"."""

    environment = ClusterProfile(name="plain").environment()

    assert "KUBECONFIG" not in environment


def test_removing_the_selected_cluster_moves_the_selection(tmp_path: Path) -> None:
    """Leaving a dangling selection would target a cluster that no longer exists."""

    save_cluster(tmp_path, ClusterProfile(name="first"))
    save_cluster(tmp_path, ClusterProfile(name="second"))
    select_cluster(tmp_path, "second")

    catalog = remove_cluster(tmp_path, "second")

    assert catalog.selected == "first"


def test_removing_the_last_cluster_clears_the_selection(tmp_path: Path) -> None:
    save_cluster(tmp_path, ClusterProfile(name="only"))

    catalog = remove_cluster(tmp_path, "only")

    assert catalog.clusters == []
    assert catalog.selected == ""


def test_a_damaged_catalog_does_not_prevent_startup(tmp_path: Path) -> None:
    """The operator can still register a cluster, which rewrites the file."""

    (tmp_path / "clusters.json").write_text("{ not json", encoding="utf-8")

    assert read_catalog(tmp_path).clusters == []


def test_names_and_namespaces_are_validated() -> None:
    assert validate_cluster_name("  LEO-Spark ") == "leo-spark"
    assert validate_namespace("Flake-Graph") == "flake-graph"
    for invalid in ("a", "-bad", "bad-", "has space", "x" * 64):
        with pytest.raises(ValueError):
            validate_namespace(invalid)


def test_the_cluster_contributes_only_what_distinguishes_it() -> None:
    """A fleet command still needs PATH and HOME to start at all.

    The cluster contributes only what distinguishes it, so a caller composing the
    subprocess environment can layer it over its own without losing the ambient
    entries the executable is found through.
    """

    profile = ClusterProfile(name="lab", namespace="lab-ns", kubeconfig="/tmp/lab.yaml")

    environment = profile.environment()

    assert environment == {
        "FLAKEGRAPH_APP_KUBERNETES_NAMESPACE": "lab-ns",
        "KUBECONFIG": "/tmp/lab.yaml",
    }
    assert "PATH" not in environment


def test_the_context_reaches_kubectl_as_a_flag() -> None:
    """kubectl reads its context from an argument, never from the environment.

    A context carried only as an environment variable selects nothing, so the
    cluster picker would silently read whatever the ambient kubeconfig points at.
    """

    target = ClusterProfile(name="lab", context="lab-ctx", kubeconfig="/tmp/lab.yaml").target()

    assert target.arguments() == ["--context", "lab-ctx"]
    assert target.environment({"PATH": "/usr/bin"}) == {
        "PATH": "/usr/bin",
        "KUBECONFIG": "/tmp/lab.yaml",
    }


def test_a_cluster_without_a_context_uses_the_ambient_selection() -> None:
    """A single-cluster host has one kubeconfig and one current context."""

    target = ClusterProfile(name="only").target()

    assert target.arguments() == []
    assert target.environment({"PATH": "/usr/bin"}) == {"PATH": "/usr/bin"}
