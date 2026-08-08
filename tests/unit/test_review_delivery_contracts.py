from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).parents[2]
_CI_WORKFLOW = _ROOT / ".github/workflows/ci.yml"
_CHART = _ROOT / "deploy/helm/flakegraph"


def test_ci_installs_and_checks_the_app_with_postgres_and_coverage() -> None:
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["python-quality"]
    steps = _steps_by_name(job["steps"])

    postgres = job["services"]["postgres"]
    assert postgres["image"].startswith("postgres:17.9-bookworm@sha256:")
    assert postgres["env"]["POSTGRES_DB"] == "flakegraph"
    assert "--health-cmd pg_isready" in postgres["options"]

    assert "--extra app" in steps["Install locked app environment"]["run"]
    dependency_audit = steps["Audit runtime dependencies"]["run"]
    assert "uv export --locked" in dependency_audit
    assert "--all-extras" in dependency_audit
    assert "uvx --python 3.14 pip-audit==2.10.1" in dependency_audit
    assert steps["Mypy app"]["run"] == "uv run mypy --explicit-package-bases app"
    assert "--cov=app" in steps["App coverage"]["run"]
    assert "--cov-fail-under=50" in steps["App coverage"]["run"]

    postgres_step = steps["PostgreSQL integration contract"]
    assert postgres_step["run"] == (
        "uv run pytest tests/integration/test_postgres_distributed_store.py"
    )
    assert postgres_step["env"]["KG_TEST_POSTGRES_DSN"].startswith("postgresql://")


def test_spark_image_gate_builds_and_smokes_the_real_dockerfile_selectively() -> None:
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["spark-image"]
    steps = _steps_by_name(job["steps"])

    condition = job["if"]
    assert "github.event_name == 'push'" in condition
    assert "github.event_name == 'workflow_dispatch'" in condition
    assert "ci:spark-image" in condition
    assert steps["Build Spark image"]["run"] == (
        "docker build --file Dockerfile.spark --tag flakegraph-spark:ci ."
    )
    smoke = steps["Smoke Spark image"]["run"]
    assert "/opt/venv/bin/python" in smoke
    assert "import graphframes, kg_processor, pyspark, sentence_transformers" in smoke
    assert "/opt/spark/bin/spark-submit" in smoke


def test_spark_account_defaults_are_release_scoped_and_schema_valid() -> None:
    values = yaml.safe_load((_CHART / "values.yaml").read_text(encoding="utf-8"))
    schema = json.loads((_CHART / "values.schema.json").read_text(encoding="utf-8"))
    documents = _render_chart("review-a")

    assert values["spark"]["serviceAccount"] == {"create": True, "name": ""}
    assert values["spark"]["serviceAccountName"] == ""
    service_account_schema = schema["properties"]["spark"]["properties"]["serviceAccount"]
    assert service_account_schema["required"] == ["create", "name"]

    spark_name = "review-a-flakegraph-spark"
    assert spark_name in _names_for_kind(documents, "ServiceAccount")
    assert _names_for_kind(documents, "Role") == {spark_name}
    assert _names_for_kind(documents, "RoleBinding") == {spark_name}

    binding = _document(documents, "RoleBinding", spark_name)
    assert binding["subjects"][0]["name"] == spark_name
    finalize = _document(documents, "Deployment", "review-a-flakegraph-finalize")
    pod_spec = finalize["spec"]["template"]["spec"]
    assert pod_spec["serviceAccountName"] == spark_name
    assert _container_env(pod_spec, "KG_DISTRIBUTED_SPARK_SERVICE_ACCOUNT") == spark_name

    long_release_documents = _render_chart("r" * 53)
    long_spark_names = _names_for_kind(long_release_documents, "Role")
    assert len(long_spark_names) == 1
    assert next(iter(long_spark_names)).endswith("-spark")
    assert len(next(iter(long_spark_names))) <= 63


def test_explicit_external_spark_account_name_is_not_rewritten_or_created() -> None:
    documents = _render_chart(
        "review-b",
        "--set",
        "spark.serviceAccount.create=false",
        "--set",
        "spark.serviceAccountName=platform-spark",
    )

    assert "platform-spark" not in _names_for_kind(documents, "ServiceAccount")
    assert _names_for_kind(documents, "Role") == {"review-b-flakegraph-spark"}
    binding = _document(documents, "RoleBinding", "review-b-flakegraph-spark")
    assert binding["subjects"][0]["name"] == "platform-spark"

    finalize = _document(documents, "Deployment", "review-b-flakegraph-finalize")
    pod_spec = finalize["spec"]["template"]["spec"]
    assert pod_spec["serviceAccountName"] == "platform-spark"
    assert _container_env(pod_spec, "KG_DISTRIBUTED_SPARK_SERVICE_ACCOUNT") == ("platform-spark")


def _render_chart(release: str, *arguments: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "helm",
            "template",
            release,
            str(_CHART),
            "--set",
            "spark.enabled=true",
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def _steps_by_name(steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {step["name"]: step for step in steps}


def _names_for_kind(documents: list[dict[str, Any]], kind: str) -> set[str]:
    return {document["metadata"]["name"] for document in documents if document.get("kind") == kind}


def _document(documents: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    return next(
        document
        for document in documents
        if document.get("kind") == kind and document["metadata"]["name"] == name
    )


def _container_env(pod_spec: dict[str, Any], name: str) -> str:
    worker = next(
        container for container in pod_spec["containers"] if container["name"] == "worker"
    )
    variable = next(item for item in worker["env"] if item["name"] == name)
    return str(variable["value"])
