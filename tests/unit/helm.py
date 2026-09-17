"""Render the chart the way a cluster would, so tests assert on objects."""

from __future__ import annotations

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


def one(rendered: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    """Return exactly one rendered object by kind and name."""

    matches = [doc for doc in rendered if doc["kind"] == kind and doc["metadata"]["name"] == name]
    assert len(matches) == 1, f"expected one {kind}/{name}, found {len(matches)}"
    return matches[0]
