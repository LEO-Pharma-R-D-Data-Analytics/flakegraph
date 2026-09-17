"""Pin what the container profiles lock and what CI runs them with."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

_COMPOSE_PATH = Path(".github/docker-compose.smoke.yaml")
_GITHUB_ACTION_SHA_RE = re.compile(r"uses: actions/[A-Za-z0-9_-]+@[a-f0-9]{40}")


def test_locked_container_profiles_use_cpu_torch_on_linux() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    lockfile = Path("uv.lock").read_text(encoding="utf-8")
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")

    sources = pyproject["tool"]["uv"]["sources"]
    indexes = {index["name"]: index for index in pyproject["tool"]["uv"]["index"]}
    assert sources["torch"] == [{"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}]
    assert "torchvision" not in sources
    assert indexes["pytorch-cpu"]["url"] == "https://download.pytorch.org/whl/cpu"
    assert indexes["pytorch-cpu"]["explicit"] is True

    assert 'version = "2.13.0+cpu"' in lockfile
    assert 'name = "torchvision"' not in lockfile
    assert 'source = { registry = "https://download.pytorch.org/whl/cpu" }' in lockfile
    assert "nvidia-" not in lockfile
    assert "cuda-toolkit" not in lockfile
    assert 'name = "triton"' not in lockfile
    assert "KG_TORCH_INDEX_URL" not in dockerfile
    assert "KG_TORCH_INDEX_URL" not in compose


def test_compose_smoke_service_is_small_and_self_contained() -> None:
    compose = yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))
    services = compose["services"]

    assert set(services) == {"flakegraph"}
    smoke_args = services["flakegraph"]["build"]["args"]
    assert smoke_args["KG_INSTALL_MINERU"] == "false"
    assert smoke_args["KG_INSTALL_LOCAL_EMBEDDINGS"] == "false"
    service = services["flakegraph"]
    assert service["command"] == ["worker", "--config", "/app/smoke.yaml"]
    assert set(service["volumes"]) == {
        "../.github/configs/smoke.yaml:/app/smoke.yaml:ro",
        "../data:/app/data:ro",
        "flakegraph-out:/app/out",
    }
    assert "environment" not in service
    assert "flakegraph-out" in compose["volumes"]


def test_ci_workflow_pins_first_party_actions_to_commit_shas() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    action_uses = [
        line.split("#", 1)[0].strip()
        for line in workflow.splitlines()
        if line.strip().startswith("uses: actions/")
    ]

    assert action_uses
    assert all(_GITHUB_ACTION_SHA_RE.fullmatch(line) for line in action_uses)
    assert "@v4" not in workflow
    assert "@v5" not in workflow
