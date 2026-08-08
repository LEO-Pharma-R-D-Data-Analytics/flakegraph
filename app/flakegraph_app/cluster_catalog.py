"""Kubernetes clusters the app can run against.

A fleet is not one cluster. Machines arrive, a cluster is rebuilt, and work is
split between a lab cluster and a production one, so the target has to be a
choice rather than a single namespace read from the environment.

Clusters are stored as a catalog beside the run history: each entry names a
kubeconfig context and namespace, and the runtime applies them per command. The
kubeconfig itself is never copied here — it holds credentials, and duplicating it
into app state would put cluster admin keys somewhere nobody expects them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CLUSTERS_FILE = "clusters.json"
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,38}[a-z0-9]$")
_NAMESPACE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$")


@dataclass(frozen=True)
class ClusterTarget:
    """Where kubectl should point for one cluster.

    Empty values mean the ambient kubeconfig and its current context, which is
    the correct behaviour on a host that has only one cluster.
    """

    kubeconfig: str = ""
    context: str = ""

    def arguments(self) -> list[str]:
        """Return the kubectl flags that select this cluster."""

        return ["--context", self.context] if self.context else []

    def environment(self, base: Mapping[str, str]) -> dict[str, str]:
        """Return ``base`` with the kubeconfig this target selects, if any."""

        environment = dict(base)
        if self.kubeconfig:
            environment["KUBECONFIG"] = self.kubeconfig
        return environment


@dataclass(frozen=True)
class ClusterProfile:
    """One registered cluster and how to reach it."""

    name: str
    namespace: str = "flakegraph"
    context: str = ""
    kubeconfig: str = ""
    description: str = ""

    def environment(self) -> dict[str, str]:
        """Return the environment overrides a fleet command needs for this cluster.

        These are overlaid on the caller's own environment rather than replacing
        it: a command still needs its PATH and HOME to run at all. An empty
        kubeconfig means the ambient one, which is what a single-cluster host has.
        """

        environment = {"FLAKEGRAPH_APP_KUBERNETES_NAMESPACE": self.namespace}
        if self.kubeconfig:
            environment["KUBECONFIG"] = self.kubeconfig
        return environment

    def target(self) -> ClusterTarget:
        """Return how kubectl must be pointed at this cluster.

        kubectl takes its context from a flag rather than the environment, so the
        selection has to be carried as a value to every call. Without it the
        commands read whatever the ambient kubeconfig happens to select and the
        cluster picker only appears to work.
        """

        return ClusterTarget(kubeconfig=self.kubeconfig, context=self.context)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "context": self.context,
            "kubeconfig": self.kubeconfig,
            "description": self.description,
        }


@dataclass
class ClusterCatalog:
    """The registered clusters and which one is selected."""

    clusters: list[ClusterProfile] = field(default_factory=list)
    selected: str = ""

    def get(self, name: str) -> ClusterProfile | None:
        return next((cluster for cluster in self.clusters if cluster.name == name), None)

    def active(self) -> ClusterProfile | None:
        """Return the selected cluster, falling back to the only sensible default.

        A catalog with one cluster and no explicit selection is unambiguous, so
        the app selects it rather than asking a question with one answer.
        """

        return self.get(self.selected) or (self.clusters[0] if self.clusters else None)


def validate_cluster_name(value: str) -> str:
    """Normalize a cluster name usable as a stable catalog key."""

    normalized = value.strip().lower()
    if not _NAME_PATTERN.match(normalized):
        raise ValueError(
            "Cluster name must be 2-40 characters of lowercase letters, digits or "
            "hyphens, and must start and end with a letter or digit."
        )
    return normalized


def validate_namespace(value: str) -> str:
    """Normalize a Kubernetes namespace, which the cluster itself will enforce."""

    normalized = value.strip().lower()
    if not _NAMESPACE_PATTERN.match(normalized):
        raise ValueError(
            "Namespace must be 2-63 characters of lowercase letters, digits or "
            "hyphens, and must start and end with a letter or digit."
        )
    return normalized


def read_catalog(state_root: Path) -> ClusterCatalog:
    """Load registered clusters, treating a damaged file as empty rather than fatal.

    A malformed catalog must not prevent the app from starting: the operator can
    still register a cluster, which rewrites the file.
    """

    path = state_root / _CLUSTERS_FILE
    if not path.exists():
        return ClusterCatalog()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ClusterCatalog()
    if not isinstance(payload, dict):
        return ClusterCatalog()
    clusters = [
        ClusterProfile(
            name=str(item.get("name", "")),
            namespace=str(item.get("namespace") or "flakegraph"),
            context=str(item.get("context") or ""),
            kubeconfig=str(item.get("kubeconfig") or ""),
            description=str(item.get("description") or ""),
        )
        for item in payload.get("clusters", [])
        if isinstance(item, dict) and item.get("name")
    ]
    return ClusterCatalog(clusters=clusters, selected=str(payload.get("selected") or ""))


def write_catalog(state_root: Path, catalog: ClusterCatalog) -> None:
    """Persist the catalog atomically, so a crash cannot truncate it."""

    state_root.mkdir(parents=True, exist_ok=True)
    destination = state_root / _CLUSTERS_FILE
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "clusters": [cluster.to_json() for cluster in catalog.clusters],
                "selected": catalog.selected,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(destination)


def save_cluster(state_root: Path, cluster: ClusterProfile) -> ClusterCatalog:
    """Register or update one cluster, keeping names unique."""

    catalog = read_catalog(state_root)
    remaining = [item for item in catalog.clusters if item.name != cluster.name]
    catalog.clusters = sorted([*remaining, cluster], key=lambda item: item.name)
    if not catalog.selected:
        catalog.selected = cluster.name
    write_catalog(state_root, catalog)
    return catalog


def remove_cluster(state_root: Path, name: str) -> ClusterCatalog:
    """Forget one cluster, moving the selection if it pointed there."""

    catalog = read_catalog(state_root)
    catalog.clusters = [item for item in catalog.clusters if item.name != name]
    if catalog.selected == name:
        catalog.selected = catalog.clusters[0].name if catalog.clusters else ""
    write_catalog(state_root, catalog)
    return catalog


def select_cluster(state_root: Path, name: str) -> ClusterCatalog:
    """Choose which registered cluster subsequent commands target."""

    catalog = read_catalog(state_root)
    if catalog.get(name) is None:
        raise ValueError(f"Unknown cluster: {name}")
    catalog.selected = name
    write_catalog(state_root, catalog)
    return catalog
