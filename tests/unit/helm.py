"""Render the chart the way a cluster would, so tests assert on objects."""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

CHART = Path("deploy/helm/flakegraph")
RELEASE = "fg"
NAMESPACE = "fleet"
FULLNAME = f"{RELEASE}-flakegraph"
# The operator CRDs the chart's monitoring objects hang off, declared present
# so a render includes them without a cluster to ask.
API_VERSIONS = ",".join(
    f"monitoring.coreos.com/v1/{kind}"
    for kind in ("ServiceMonitor", "PodMonitor", "PrometheusRule")
)


def helm() -> str:
    """Return the helm binary, or skip the test that needs it."""

    found = shutil.which("helm")
    if found is None:
        pytest.skip("helm is not installed; the rendered-chart contracts need it")
    return found


@cache
def render(
    settings: tuple[str, ...] = (),
    *,
    values: tuple[Path, ...] = (),
    release: str = RELEASE,
) -> list[dict[str, Any]]:
    """Render the chart offline with ``--set`` overrides and values files."""

    command = [
        helm(),
        "template",
        release,
        str(CHART),
        "--namespace",
        NAMESPACE,
        "--api-versions",
        API_VERSIONS,
    ]
    for path in values:
        command += ["--values", str(path)]
    for setting in settings:
        command += ["--set", setting]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def fails(
    settings: tuple[str, ...],
    *,
    values: tuple[Path, ...] = (),
    api_versions: str = API_VERSIONS,
) -> str:
    """Render a configuration the chart must refuse, and return what it said.

    ``api_versions=""`` renders against a cluster without the monitoring CRDs.
    """

    command = [helm(), "template", RELEASE, str(CHART), "--namespace", NAMESPACE]
    if api_versions:
        command += ["--api-versions", api_versions]
    for path in values:
        command += ["--values", str(path)]
    for setting in settings:
        command += ["--set", setting]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0, "the chart rendered a configuration it must refuse"
    return result.stderr


@cache
def notes(settings: tuple[str, ...] = ()) -> str:
    """Return the NOTES a release would print, rendered without a cluster.

    ``helm template`` never prints NOTES.txt; a client-side dry-run install
    does, and with no kubeconfig to find it cannot reach for a cluster.
    """

    command = [
        helm(),
        "install",
        RELEASE,
        str(CHART),
        "--dry-run=client",
        "--namespace",
        NAMESPACE,
    ]
    for setting in settings:
        command += ["--set", setting]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "KUBECONFIG": os.devnull},
    )
    assert result.returncode == 0, result.stderr
    _, _, printed = result.stdout.partition("\nNOTES:\n")
    return printed


def one(rendered: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    """Return exactly one rendered object by kind and name."""

    matches = [doc for doc in rendered if doc["kind"] == kind and doc["metadata"]["name"] == name]
    assert len(matches) == 1, f"expected one {kind}/{name}, found {len(matches)}"
    return matches[0]
