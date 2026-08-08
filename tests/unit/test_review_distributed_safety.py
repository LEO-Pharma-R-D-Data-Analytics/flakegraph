"""Regression contracts for distributed planning and publication safety."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from kg_processor.adapters.distributed.postgres import (
    _SCHEMA_STATEMENTS,
    _SCHEMA_VERSION,
    PostgresDistributedStore,
)
from kg_processor.application.distributed_planner import _graph_output_payload
from kg_processor.application.distributed_worker import DistributedWorker
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    ArtifactKind,
    ArtifactRef,
    PublicationLease,
    StoredArtifact,
    TaskStage,
)
from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.ports.task_store import FencedPublicationStore


def test_schema_v4_started_at_backfill_is_preaggregated_scoped_and_clamped() -> None:
    """Avoid corpus-wide correlated stage scans and impossible historical timestamps."""

    migration = next(
        statement for statement in _SCHEMA_STATEMENTS if "WITH prerequisite_events AS" in statement
    )

    assert _SCHEMA_VERSION >= 6
    assert "flakegraph_task_dependency" in migration
    assert "dependency.task_id" in migration
    assert "GROUP BY task_id" in migration
    assert "LEAST(" in migration
    assert "GREATEST(" in migration
    assert "WHEN 'compact_document'" not in migration


def test_postgres_publication_callback_uses_optimistic_fence_without_holding_locks() -> None:
    """Cancellation stays responsive while completion checks the original lease fence."""

    source = inspect.getsource(PostgresDistributedStore.publish_claimed)

    assert "FOR UPDATE OF publication, run" not in source
    assert "lease_expires_at >= CURRENT_TIMESTAMP" in source
    assert "publication.attempts = %s" in source
    assert "publication.generation = %s" in source
    assert source.index("publish(lease)") < source.index("UPDATE flakegraph_publication")


def test_postgres_task_reclaims_are_bounded_with_an_activity_aware_terminal_path() -> None:
    claim_source = inspect.getsource(PostgresDistributedStore.claim_task)
    expiry_source = inspect.getsource(PostgresDistributedStore._fail_abandoned_exhausted_tasks)

    assert "attempts = task.attempts + 1" in claim_source
    assert "task.attempts < task.max_attempts" in claim_source
    assert "self._fail_abandoned_exhausted_tasks(connection)" in claim_source
    assert "attempts >= max_attempts" in expiry_source
    assert "GREATEST(lease_expires_at, updated_at)" in expiry_source


def test_barrierless_completion_serializes_on_the_run_row() -> None:
    source = inspect.getsource(PostgresDistributedStore._complete_run_without_final_barrier)

    assert "SELECT id FROM flakegraph_run WHERE id = %s FOR UPDATE" in source
    assert source.index("FOR UPDATE") < source.index("UPDATE flakegraph_run")
    assert "completed_at" not in source
    assert "NOT EXISTS" in source


def test_finalizer_rejects_arbitrary_environment_credential_references(tmp_path: Path) -> None:
    """A queued task cannot ask a worker to reveal an unrelated environment secret."""

    settings = _settings(tmp_path)
    settings.writer.provider = "snowflake_bulk"
    settings.snowflake.database = "DB"
    settings.snowflake.schema_name = "GRAPH"
    settings.snowflake.bulk_stage = "@DB.GRAPH.LOAD_STAGE"
    settings.snowflake.password_environment_variable = "AWS_SECRET_ACCESS_KEY"

    with pytest.raises(ValueError, match="dedicated password credential slot"):
        _graph_output_payload(settings)


def test_worker_drains_fenced_publication_before_claiming_another_task(tmp_path: Path) -> None:
    """Final task completion queues publication instead of exposing Snowflake inline."""

    manifest = GraphDatasetManifest(
        run_id="run-1",
        graph_id="graph-1",
        engine="spark",
        tables={},
    )
    artifact = StoredArtifact(
        ref=ArtifactRef(
            id="artifact-1",
            run_id="run-1",
            kind=ArtifactKind.GRAPH_RESULT,
            media_type="application/vnd.flakegraph.graph-manifest+json",
            checksum="abc",
            size_bytes=len(manifest.model_dump_json()),
        ),
        payload=manifest.model_dump_json().encode(),
    )
    store = _PublicationStore(artifact)
    publisher = _Publisher()
    worker = DistributedWorker(
        _settings(tmp_path),
        "finalizer-1",
        {TaskStage.FINALIZE_GRAPH},
        pipeline=cast(Any, object()),
        task_store=cast(Any, store),
        artifact_store=cast(Any, store),
        manifest_publisher=publisher,
    )

    result = worker.process_one()

    assert result.succeeded is True
    assert store.claim_task_calls == 0
    assert publisher.payloads[0]["publication"] == {
        "id": "publication-1",
        "generation": 17,
        "attempt": 1,
    }


class _PublicationStore:
    """Small structural fake for the optional fenced publication capability."""

    def __init__(self, artifact: StoredArtifact) -> None:
        self.artifact = artifact
        self.claim_task_calls = 0
        self.claimed = False

    def claim_publication(
        self,
        worker_id: str,
        lease_duration: timedelta,
    ) -> PublicationLease | None:
        del lease_duration
        if self.claimed:
            return None
        self.claimed = True
        return PublicationLease(
            id="publication-1",
            run_id="run-1",
            artifact_id=self.artifact.ref.id,
            task_payload={"output": {"provider": "snowflake_bulk"}},
            worker_id=worker_id,
            attempt=1,
            generation=17,
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )

    def publish_claimed(self, publication_id: str, worker_id: str, publish: Any) -> None:
        publish(self.claim_publication_lease(publication_id, worker_id))

    def heartbeat_publication(
        self,
        _publication_id: str,
        _worker_id: str,
        _lease_duration: timedelta,
    ) -> None:
        return None

    def claim_publication_lease(
        self,
        publication_id: str,
        worker_id: str,
    ) -> PublicationLease:
        return PublicationLease(
            id=publication_id,
            run_id="run-1",
            artifact_id=self.artifact.ref.id,
            task_payload={"output": {"provider": "snowflake_bulk"}},
            worker_id=worker_id,
            attempt=1,
            generation=17,
            lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )

    def fail_publication(self, *args: object, **kwargs: object) -> None:
        raise AssertionError((args, kwargs))

    def claim_task(self, *args: object, **kwargs: object) -> None:
        self.claim_task_calls += 1

    def get(self, artifact_id: str) -> StoredArtifact:
        assert artifact_id == self.artifact.ref.id
        return self.artifact


class _Publisher:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def publish(self, manifest: GraphDatasetManifest, task_payload: Any) -> None:
        assert manifest.run_id == "run-1"
        self.payloads.append(dict(task_payload))


def _settings(tmp_path: Path) -> Settings:
    return Settings.load(
        env={},
        overrides={
            "job": {"graph_id": "graph-1"},
            "files": {"input_path": str(tmp_path)},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "api_key": "test"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path)},
            "cache": {"provider": "none"},
            "distributed": {"database_url": "postgresql://example/test"},
        },
    )


def test_artifact_retention_clears_every_restricting_reference_before_deleting_rows() -> None:
    """A surviving reference must fail the row delete, never orphan a live payload.

    ``flakegraph_artifact`` is referenced with ON DELETE RESTRICT by more than one
    table. Retention clears each of them first, so a reference it forgot cannot
    destroy objects and then fail the delete on every retry.
    """

    source = inspect.getsource(PostgresDistributedStore.delete_run_artifacts)
    restricting_tables = {
        statement.split("CREATE TABLE IF NOT EXISTS ", 1)[1].split(" ", 1)[0]
        for statement in _SCHEMA_STATEMENTS
        if "CREATE TABLE IF NOT EXISTS " in statement
        and "REFERENCES flakegraph_artifact(id) ON DELETE RESTRICT" in statement
    }

    assert restricting_tables
    for table in restricting_tables:
        assert f"DELETE FROM {table} WHERE run_id = %s" in source
    assert source.index("DELETE FROM flakegraph_artifact") < source.index("_delete_blob_batch")


def test_fenced_publication_contract_describes_the_fence_its_stores_implement() -> None:
    """A port contract that contradicts every implementation misdirects the next one."""

    contract = inspect.getdoc(FencedPublicationStore) or ""
    implementation = inspect.getsource(PostgresDistributedStore.publish_claimed)

    assert "must not hold a store lock across the callback" in contract
    assert "generation" in contract
    assert "attempt" in contract
    assert "FOR UPDATE" not in implementation
