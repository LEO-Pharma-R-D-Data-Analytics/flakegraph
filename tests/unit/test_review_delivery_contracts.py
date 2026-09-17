"""Protect the Spark service account the finalizer runs and submits as."""

from __future__ import annotations

from typing import Any

from helm import one as _one
from helm import render as _render
from helm import schema, values

_SPARK = ("spark.enabled=true",)


def test_spark_account_defaults_are_release_scoped_and_schema_valid() -> None:
    documents = _render(_SPARK, release="review-a")

    assert values()["spark"]["serviceAccount"] == {"create": True, "name": ""}
    assert values()["spark"]["serviceAccountName"] == ""
    service_account_schema = schema()["properties"]["spark"]["properties"]["serviceAccount"]
    assert set(service_account_schema["properties"]) == {"create", "name"}

    spark_name = "review-a-flakegraph-spark"
    assert spark_name in _names_for_kind(documents, "ServiceAccount")
    assert spark_name in _names_for_kind(documents, "Role")
    assert spark_name in _names_for_kind(documents, "RoleBinding")
    # The chart owns other roles now, so what matters is that every one of them
    # is scoped to this release rather than that Spark's is the only one.
    for kind in ("Role", "RoleBinding"):
        assert all(name.startswith("review-a-") for name in _names_for_kind(documents, kind))

    binding = _one(documents, "RoleBinding", spark_name)
    assert binding["subjects"][0]["name"] == spark_name
    finalize = _one(documents, "Deployment", "review-a-flakegraph-finalize")
    pod_spec = finalize["spec"]["template"]["spec"]
    assert pod_spec["serviceAccountName"] == spark_name
    assert _container_env(pod_spec, "KG_DISTRIBUTED_SPARK_SERVICE_ACCOUNT") == spark_name

    long_release_documents = _render(_SPARK, release="r" * 53)
    long_role_names = _names_for_kind(long_release_documents, "Role")
    long_spark_names = {name for name in long_role_names if name.endswith("-spark")}
    assert len(long_spark_names) == 1
    assert all(len(name) <= 63 for name in long_role_names)


def test_explicit_external_spark_account_name_is_not_rewritten_or_created() -> None:
    documents = _render(
        (*_SPARK, "spark.serviceAccount.create=false", "spark.serviceAccountName=platform-spark"),
        release="review-b",
    )

    assert "platform-spark" not in _names_for_kind(documents, "ServiceAccount")
    assert "review-b-flakegraph-spark" in _names_for_kind(documents, "Role")
    binding = _one(documents, "RoleBinding", "review-b-flakegraph-spark")
    assert binding["subjects"][0]["name"] == "platform-spark"

    finalize = _one(documents, "Deployment", "review-b-flakegraph-finalize")
    pod_spec = finalize["spec"]["template"]["spec"]
    assert pod_spec["serviceAccountName"] == "platform-spark"
    assert _container_env(pod_spec, "KG_DISTRIBUTED_SPARK_SERVICE_ACCOUNT") == ("platform-spark")


def _names_for_kind(documents: list[dict[str, Any]], kind: str) -> set[str]:
    return {document["metadata"]["name"] for document in documents if document.get("kind") == kind}


def _container_env(pod_spec: dict[str, Any], name: str) -> str:
    worker = next(
        container for container in pod_spec["containers"] if container["name"] == "worker"
    )
    variable = next(item for item in worker["env"] if item["name"] == name)
    return str(variable["value"])
