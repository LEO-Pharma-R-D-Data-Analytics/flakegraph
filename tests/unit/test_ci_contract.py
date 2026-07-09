from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import kg_processor

_CI_WORKFLOW = Path(".github/workflows/ci.yml")


def test_ci_workflow_runs_documented_quality_gates_and_docker_smoke() -> None:
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))

    jobs = workflow["jobs"]
    python_steps = _steps_by_name(jobs["python-quality"]["steps"])
    docker_steps = _steps_by_name(jobs["docker-smoke"]["steps"])

    assert jobs["python-quality"]["runs-on"] == "ubuntu-latest"
    assert jobs["docker-smoke"]["runs-on"] == "ubuntu-latest"
    assert python_steps["Set up Python 3.13"]["with"]["python-version"] == "3.13"
    assert python_steps["Install uv"]["run"] == 'python -m pip install "uv==0.11.19"'
    assert python_steps["Install locked development environment"]["run"] == (
        "uv sync --locked --extra dev"
    )
    assert python_steps["Ruff"]["run"] == "uv run ruff check ."
    assert python_steps["Mypy"]["run"] == "uv run mypy src/kg_processor tests"
    assert python_steps["Pytest"]["run"] == "uv run pytest"
    assert docker_steps["Build and run lightweight smoke profile"]["run"] == (
        "docker compose -f .github/docker-compose.smoke.yaml up --build "
        "--abort-on-container-exit "
        "--exit-code-from flakegraph flakegraph"
    )
    assert docker_steps["Tear down smoke profile"]["run"] == (
        "docker compose -f .github/docker-compose.smoke.yaml down "
        "--volumes --remove-orphans"
    )
    assert docker_steps["Tear down smoke profile"]["if"] == "always()"
    assert kg_processor.__version__


def _steps_by_name(steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {step["name"]: step for step in steps}
