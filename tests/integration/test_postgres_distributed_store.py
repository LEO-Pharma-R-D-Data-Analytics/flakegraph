"""Live PostgreSQL contract tests for leases, barriers, retries, and artifacts."""

from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.adapters.distributed.postgres import PostgresDistributedStore
from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.application.distributed_planner import DistributedRunPlanner
from kg_processor.application.distributed_worker import DistributedWorker
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    ArtifactKind,
    RunDefinition,
    RunStatus,
    TaskDefinition,
    TaskProgress,
    TaskStage,
    TaskStatus,
)
from kg_processor.factories import build_pipeline

_POSTGRES_DSN = os.getenv("KG_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(
    not _POSTGRES_DSN,
    reason="Set KG_TEST_POSTGRES_DSN to run live PostgreSQL coordination checks.",
)


class _RecordingBlobStore:
    """Retain test payloads and record bulk retention calls without external I/O."""

    def __init__(self) -> None:
        """Create an empty object namespace and deletion-call log."""

        self.objects: dict[str, bytes] = {}
        self.deleted_batches: list[list[str]] = []

    def initialize(self) -> None:
        """Satisfy the idempotent blob-store initialization contract."""

    def put(self, key: str, payload: bytes, media_type: str) -> str:
        """Store exact bytes under a deterministic in-memory URI."""

        _ = media_type
        uri = f"memory://artifacts/{key}"
        self.objects[uri] = payload
        return uri

    def get(self, uri: str) -> bytes:
        """Return bytes for PostgreSQL artifact integrity checks."""

        return self.objects[uri]

    def delete(self, uri: str) -> None:
        """Delete one object idempotently for the fallback blob contract."""

        self.objects.pop(uri, None)

    def delete_many(self, uris: list[str]) -> None:
        """Record one bounded batch and delete all selected objects."""

        self.deleted_batches.append(list(uris))
        for uri in uris:
            self.delete(uri)


@pytest.fixture
def isolated_postgres_dsn() -> Iterator[str]:
    """Create one private schema per test so queued work cannot leak between runs."""

    assert _POSTGRES_DSN is not None
    schema_name = f"flakegraph_test_{uuid4().hex}"
    with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
    try:
        yield make_conninfo(_POSTGRES_DSN, options=f"-c search_path={schema_name}")
    finally:
        with psycopg.connect(_POSTGRES_DSN, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name))
            )


def test_postgres_schema_initialization_is_safe_during_worker_startup_burst(
    isolated_postgres_dsn: str,
) -> None:
    """Every replica may initialize concurrently without PostgreSQL DDL races."""

    stores = [PostgresDistributedStore(isolated_postgres_dsn) for _ in range(12)]
    with ThreadPoolExecutor(max_workers=len(stores)) as executor:
        list(executor.map(lambda store: store.initialize(), stores))


def test_postgres_schema_indexes_dependency_completion_lookup(
    isolated_postgres_dsn: str,
) -> None:
    """Resolve a completed prerequisite's dependants without scanning the edge table."""

    _store(isolated_postgres_dsn)
    with psycopg.connect(isolated_postgres_dsn, row_factory=dict_row) as connection:
        row = connection.execute(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND indexname = 'flakegraph_task_dependency_prerequisite_idx'
            """
        ).fetchone()

    assert row is not None
    assert "(depends_on_task_id, task_id)" in row["indexdef"]


def test_postgres_schema_indexes_the_ordered_queued_claim_path(
    isolated_postgres_dsn: str,
) -> None:
    """Avoid sorting a corpus-sized ready queue for every worker claim."""

    _store(isolated_postgres_dsn)
    with psycopg.connect(isolated_postgres_dsn, row_factory=dict_row) as connection:
        row = connection.execute(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND indexname = 'flakegraph_task_queued_claim_idx'
            """
        ).fetchone()

    assert row is not None
    assert "(stage, remaining_dependencies, priority DESC, created_at, id)" in row["indexdef"]
    assert "WHERE (status = 'queued'::text)" in row["indexdef"]


def test_postgres_preserves_distinct_source_identities_with_identical_bytes(
    isolated_postgres_dsn: str,
) -> None:
    """Content equality must not collapse separately identified source documents."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))

    first = store.put(
        run_id,
        ArtifactKind.SOURCE_DOCUMENT,
        b"same public document bytes",
        "text/plain",
        identity_key="source-a",
    )
    second = store.put(
        run_id,
        ArtifactKind.SOURCE_DOCUMENT,
        b"same public document bytes",
        "text/plain",
        identity_key="source-b",
    )

    assert first.id != second.id
    assert store.get(first.id).payload == store.get(second.id).payload


def test_postgres_deletes_external_artifacts_in_bounded_retryable_batches(
    isolated_postgres_dsn: str,
) -> None:
    """Retention must not materialize or delete a corpus-sized object set at once."""

    blob_store = _RecordingBlobStore()
    store = PostgresDistributedStore(isolated_postgres_dsn, blob_store=blob_store)
    store.initialize()
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    for index in range(1_001):
        store.put(
            run_id,
            ArtifactKind.PREPARED_DOCUMENT,
            f"payload-{index}".encode(),
            "application/json",
        )
    store.cancel_run(run_id)

    assert store.delete_run_artifacts(run_id) == 1_001
    assert [len(batch) for batch in blob_store.deleted_batches] == [1_000, 1]
    assert blob_store.objects == {}


def test_retention_removes_a_superseded_published_run_and_its_objects(
    isolated_postgres_dsn: str,
) -> None:
    """A run that published must not be permanently undeletable once superseded.

    ``flakegraph_publication`` references artifacts with ON DELETE RESTRICT, so
    retention clears the outbox record before the payload rows it names.
    """

    blob_store = _RecordingBlobStore()
    store = PostgresDistributedStore(isolated_postgres_dsn, blob_store=blob_store)
    store.initialize()
    published_run = f"run_{uuid4().hex}"
    store.create_run(_run(published_run))
    store.add_tasks(
        published_run,
        [
            _task(published_run, "final", TaskStage.FINALIZE_GRAPH, "graph").model_copy(
                update={"payload": {"output": {"provider": "snowflake_bulk"}}}
            )
        ],
    )
    store.activate_run(published_run)
    claim = store.claim_task("finalizer", {TaskStage.FINALIZE_GRAPH}, timedelta(minutes=1))
    assert claim is not None
    manifest = store.put(
        published_run,
        ArtifactKind.GRAPH_RESULT,
        b'{"format":"flakegraph.graph-dataset-manifest/v1"}',
        "application/vnd.flakegraph.graph-manifest+json",
    )
    store.complete_task(claim.task.id, claim.worker_id, [manifest.id])
    publication = store.claim_publication("publisher", timedelta(minutes=1))
    assert publication is not None
    store.publish_claimed(publication.id, publication.worker_id, lambda _lease: None)
    assert store.get_run(published_run).run.status == RunStatus.SUCCEEDED

    superseding_run = f"run_{uuid4().hex}"
    store.create_run(_run(superseding_run))
    store.add_tasks(
        superseding_run,
        [_task(superseding_run, "final", TaskStage.FINALIZE_GRAPH, "graph")],
    )
    store.activate_run(superseding_run)
    newer = store.claim_task("finalizer-2", {TaskStage.FINALIZE_GRAPH}, timedelta(minutes=1))
    assert newer is not None
    newer_manifest = store.put(
        superseding_run,
        ArtifactKind.GRAPH_RESULT,
        b'{"format":"flakegraph.graph-dataset-manifest/v1","version":2}',
        "application/vnd.flakegraph.graph-manifest+json",
    )
    store.complete_task(newer.task.id, newer.worker_id, [newer_manifest.id])

    assert store.delete_run_artifacts(published_run) == 1
    assert all(published_run not in uri for uri in blob_store.objects)
    assert store.get(newer_manifest.id).ref.id == newer_manifest.id


def test_postgres_rejects_unknown_artifact_owner_before_external_upload(
    isolated_postgres_dsn: str,
) -> None:
    """The run-identity cache must not weaken the external-object ownership check."""

    blob_store = _RecordingBlobStore()
    store = PostgresDistributedStore(isolated_postgres_dsn, blob_store=blob_store)
    store.initialize()

    with pytest.raises(KeyError, match="unknown distributed run"):
        store.put(
            "missing-run",
            ArtifactKind.SOURCE_DOCUMENT,
            b"payload",
            "application/octet-stream",
        )

    assert blob_store.objects == {}


def test_postgres_store_enforces_bounded_session_waits(
    isolated_postgres_dsn: str,
) -> None:
    """Keep abandoned or lock-blocked worker transactions from pinning the queue."""

    store = _store(isolated_postgres_dsn)
    with store._connection() as connection:  # noqa: SLF001 - adapter session contract.
        settings = connection.execute(
            """
            SELECT current_setting('lock_timeout') AS lock_timeout,
                   current_setting('idle_in_transaction_session_timeout') AS idle_timeout,
                   current_setting('client_connection_check_interval') AS client_check
            """
        ).fetchone()

    assert settings == {
        "lock_timeout": "30s",
        "idle_timeout": "1min",
        "client_check": "1s",
    }


def test_postgres_streams_initial_plan_without_materializing_a_general_dag(
    isolated_postgres_dsn: str,
) -> None:
    """Bulk-load preparation tasks and one run-wide barrier from an iterator."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))

    def tasks() -> Iterator[TaskDefinition]:
        """Yield the restricted corpus plan accepted by the streaming API."""

        for index in range(3):
            yield _task(
                run_id,
                f"prepare-{index}",
                TaskStage.PREPARE_DOCUMENT,
                f"file-{index}",
            )
        yield _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph")

    store.add_initial_tasks(run_id, tasks())
    store.activate_run(run_id)

    summary = store.get_run_summary(run_id)
    assert summary.total_tasks == 4
    assert {(row.stage, row.status, row.count) for row in summary.task_counts} == {
        (TaskStage.PREPARE_DOCUMENT, TaskStatus.QUEUED, 3),
        (TaskStage.FINALIZE_GRAPH, TaskStatus.QUEUED, 1),
    }


def test_postgres_summary_reports_latest_task_activity_without_hot_run_writes(
    isolated_postgres_dsn: str,
) -> None:
    """Expose worker liveness from task heartbeats while leaving the run row cold."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)
    run_row_updated_at = store.get_run(run_id).updated_at

    claim = store.claim_task(
        "prepare-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert claim is not None

    summary = store.get_run_summary(run_id)
    recent = next(item for item in store.list_runs() if item.id == run_id)
    assert summary.updated_at > run_row_updated_at
    assert recent.updated_at == summary.updated_at


def test_postgres_completes_a_custom_plan_without_finalizer(
    isolated_postgres_dsn: str,
) -> None:
    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    store.add_tasks(
        run_id,
        [_task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")],
    )
    store.activate_run(run_id)
    claim = store.claim_task(
        "prepare-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert claim is not None

    store.complete_task(claim.task.id, "prepare-worker", [])

    snapshot = store.get_run(run_id)
    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert snapshot.tasks[0].status == TaskStatus.SUCCEEDED


def test_postgres_claims_are_exclusive_and_dependencies_form_a_barrier(
    isolated_postgres_dsn: str,
) -> None:
    """Competing workers must get unique tasks and finalization must wait for both."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    prepare_tasks = [
        _task(run_id, f"prepare-{index}", TaskStage.PREPARE_DOCUMENT, f"file-{index}")
        for index in range(2)
    ]
    final = _task(
        run_id,
        "final",
        TaskStage.FINALIZE_GRAPH,
        "graph",
        dependency_ids=[task.id for task in prepare_tasks],
    )
    store.add_tasks(run_id, [*prepare_tasks, final])
    store.activate_run(run_id)

    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = list(
            executor.map(
                lambda index: store.claim_task(
                    f"worker-{index}",
                    {TaskStage.PREPARE_DOCUMENT},
                    timedelta(minutes=1),
                ),
                range(8),
            )
        )
    claimed = [claim for claim in claims if claim is not None]
    assert len(claimed) == 2
    assert len({claim.task.id for claim in claimed}) == 2
    assert (
        store.claim_task(
            "final-worker",
            {TaskStage.FINALIZE_GRAPH},
            timedelta(minutes=1),
        )
        is None
    )

    artifact_ids: list[str] = []
    for claim in claimed:
        payload = f"output:{claim.task.id}".encode()
        ref = store.put(
            run_id,
            ArtifactKind.PREPARED_DOCUMENT,
            payload,
            "application/json",
            metadata={"task_id": claim.task.id},
        )
        assert store.get(ref.id).payload == payload
        artifact_ids.append(ref.id)
        store.complete_task(claim.task.id, claim.worker_id, [ref.id])
    batched = store.get_many(list(reversed(artifact_ids)))
    assert [artifact.ref.id for artifact in batched] == list(reversed(artifact_ids))

    final_claim = store.claim_task(
        "final-worker",
        {TaskStage.FINALIZE_GRAPH},
        timedelta(minutes=1),
    )
    assert final_claim is not None
    assert set(final_claim.dependency_outputs) == {task.id for task in prepare_tasks}
    assert {
        artifact_id
        for outputs in final_claim.dependency_outputs.values()
        for artifact_id in outputs
    } == set(artifact_ids)
    store.complete_task(final_claim.task.id, final_claim.worker_id, [])

    snapshot = store.get_run(run_id)
    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert all(task.status == TaskStatus.SUCCEEDED for task in snapshot.tasks)
    summary = store.get_run_summary(run_id)
    assert summary.run.status == RunStatus.SUCCEEDED
    assert summary.total_tasks == 3
    assert sum(item.count for item in summary.task_counts) == 3
    assert {item.status for item in summary.task_counts} == {TaskStatus.SUCCEEDED}
    assert store.delete_run_artifacts(run_id) == 2


def test_running_task_progress_is_lease_owned_and_bounded(
    isolated_postgres_dsn: str,
) -> None:
    """Persist one finalizer phase in detailed, summary, and recent-run status."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    final = _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph")
    store.add_tasks(run_id, [final])
    store.activate_run(run_id)
    claim = store.claim_task(
        "finalizer-1",
        {TaskStage.FINALIZE_GRAPH},
        timedelta(minutes=1),
    )
    assert claim is not None
    progress = TaskProgress(
        phase="write_tables",
        phase_index=10,
        phase_total=12,
        completed=4,
        total=12,
        message="Writing graph tables",
    )

    with pytest.raises(RuntimeError, match="no longer owns"):
        store.report_task_progress(final.id, "another-worker", progress)
    store.report_task_progress(final.id, claim.worker_id, progress)

    detailed = store.get_run(run_id)
    summary = store.get_run_summary(run_id)
    recent = next(item for item in store.list_runs() if item.id == run_id)
    assert detailed.tasks[0].progress == progress
    assert summary.task_counts[0].progress == progress
    assert recent.task_counts[0].progress == progress


def test_worker_demand_view_tracks_ready_work_and_active_leases(
    isolated_postgres_dsn: str,
) -> None:
    """Keep autoscaling capacity until leases finish and hide blocked barriers."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    prepare_tasks = [
        _task(run_id, f"prepare-{index}", TaskStage.PREPARE_DOCUMENT, f"file-{index}")
        for index in range(2)
    ]
    final = _task(
        run_id,
        "final",
        TaskStage.FINALIZE_GRAPH,
        "graph",
        dependency_ids=[task.id for task in prepare_tasks],
    )
    store.add_tasks(run_id, [*prepare_tasks, final])
    store.activate_run(run_id)

    assert _worker_demand(isolated_postgres_dsn) == {"prepare_document": 2}

    first = store.claim_task("prepare-1", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert first is not None
    assert _worker_demand(isolated_postgres_dsn) == {"prepare_document": 2}
    store.complete_task(first.task.id, first.worker_id, [])
    assert _worker_demand(isolated_postgres_dsn) == {"prepare_document": 1}

    second = store.claim_task("prepare-2", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert second is not None
    store.complete_task(second.task.id, second.worker_id, [])
    assert _worker_demand(isolated_postgres_dsn) == {"finalize_graph": 1}

    final_claim = store.claim_task(
        "finalize-1",
        {TaskStage.FINALIZE_GRAPH},
        timedelta(minutes=1),
    )
    assert final_claim is not None
    assert _worker_demand(isolated_postgres_dsn) == {"finalize_graph": 1}
    store.complete_task(final_claim.task.id, final_claim.worker_id, [])
    assert _worker_demand(isolated_postgres_dsn) == {}


def test_worker_demand_keeps_finalize_capacity_for_pending_publication(
    isolated_postgres_dsn: str,
) -> None:
    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    final = _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "graph").model_copy(
        update={"payload": {"output": {"provider": "snowflake_bulk"}}}
    )
    store.add_tasks(run_id, [final])
    store.activate_run(run_id)
    claim = store.claim_task("finalizer", {TaskStage.FINALIZE_GRAPH}, timedelta(minutes=1))
    assert claim is not None
    manifest = store.put(
        run_id,
        ArtifactKind.GRAPH_RESULT,
        b'{"format":"flakegraph.graph-dataset-manifest/v1"}',
        "application/vnd.flakegraph.graph-manifest+json",
    )

    store.complete_task(claim.task.id, claim.worker_id, [manifest.id])

    assert store.get_run(run_id).run.status == RunStatus.RUNNING
    assert _worker_demand(isolated_postgres_dsn) == {"finalize_graph": 1}
    publication = store.claim_publication("publisher", timedelta(minutes=1))
    assert publication is not None
    assert _worker_demand(isolated_postgres_dsn) == {"finalize_graph": 1}


def test_publication_only_retry_reactivates_run_and_demand(
    isolated_postgres_dsn: str,
) -> None:
    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    final = _task(
        run_id,
        "final",
        TaskStage.FINALIZE_GRAPH,
        "graph",
        max_attempts=1,
    ).model_copy(update={"payload": {"output": {"provider": "snowflake_bulk"}}})
    store.add_tasks(run_id, [final])
    store.activate_run(run_id)
    task = store.claim_task("finalizer", {TaskStage.FINALIZE_GRAPH}, timedelta(minutes=1))
    assert task is not None
    manifest = store.put(
        run_id,
        ArtifactKind.GRAPH_RESULT,
        b'{"format":"flakegraph.graph-dataset-manifest/v1"}',
        "application/vnd.flakegraph.graph-manifest+json",
    )
    store.complete_task(task.task.id, task.worker_id, [manifest.id])
    publication = store.claim_publication("publisher", timedelta(minutes=1))
    assert publication is not None
    store.fail_publication(
        publication.id,
        publication.worker_id,
        {"message": "expired destination credential"},
        timedelta(0),
    )
    assert store.get_run(run_id).run.status == RunStatus.FAILED

    store.retry_run(run_id)

    assert store.get_run(run_id).run.status == RunStatus.RUNNING
    assert _worker_demand(isolated_postgres_dsn) == {"finalize_graph": 1}
    assert store.claim_publication("publisher-2", timedelta(minutes=1)) is not None


def test_postgres_claims_filter_runs_by_processing_config(
    isolated_postgres_dsn: str,
) -> None:
    """Workers must ignore tasks produced by an incompatible runtime configuration."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)

    assert (
        store.claim_task(
            "incompatible-worker",
            {TaskStage.PREPARE_DOCUMENT},
            timedelta(minutes=1),
            expected_config_digest="different-config",
        )
        is None
    )
    claim = store.claim_task(
        "compatible-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
        expected_config_digest="integration-config",
    )
    assert claim is not None
    assert claim.task.id == task.id


def test_postgres_retries_then_fails_run_after_attempt_budget(
    isolated_postgres_dsn: str,
) -> None:
    """A task remains claimable until its configured attempt budget is exhausted."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(
        run_id,
        "prepare",
        TaskStage.PREPARE_DOCUMENT,
        "file",
        max_attempts=2,
    )
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)

    first = store.claim_task("worker-1", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert first is not None and first.attempt == 1
    store.fail_task(first.task.id, first.worker_id, {"message": "first"}, timedelta(0))

    second = store.claim_task("worker-2", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert second is not None and second.attempt == 2
    store.fail_task(second.task.id, second.worker_id, {"message": "second"}, timedelta(0))

    snapshot = store.get_run(run_id)
    assert snapshot.run.status == RunStatus.FAILED
    assert snapshot.tasks[0].status == TaskStatus.FAILED
    assert snapshot.tasks[0].attempts == 2
    assert snapshot.tasks[0].last_error == {"message": "second"}
    summary = store.get_run_summary(run_id)
    assert summary.error == {"task_id": task.id, "message": "second"}

    store.retry_run(run_id)
    retried = store.get_run(run_id)
    retried_summary = store.get_run_summary(run_id)
    assert retried.run.status == RunStatus.QUEUED
    assert retried.tasks[0].status == TaskStatus.QUEUED
    assert retried.tasks[0].attempts == 0
    assert retried.tasks[0].last_error is None
    assert retried_summary.error is None
    assert retried_summary.task_counts[0].started_at is None
    assert retried_summary.task_counts[0].completed_at is None
    assert retried_summary.task_counts[0].progress is None


def test_postgres_bounds_abandoned_reclaims_and_fails_after_recovery_window(
    isolated_postgres_dsn: str,
) -> None:
    """Consume retry budget on reclaim and terminally fail a repeatedly lost task."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(
        run_id,
        "prepare",
        TaskStage.PREPARE_DOCUMENT,
        "file",
        max_attempts=2,
    )
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)
    lease = store.claim_task(
        "abandoned-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert lease is not None

    with psycopg.connect(isolated_postgres_dsn) as connection:
        connection.execute(
            "UPDATE flakegraph_task SET lease_expires_at = "
            "CURRENT_TIMESTAMP - INTERVAL '1 minute', "
            "updated_at = CURRENT_TIMESTAMP - INTERVAL '1 minute' "
            "WHERE id = %s",
            (task.id,),
        )

    reclaimed = store.claim_task(
        "recovery-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert reclaimed is not None
    assert reclaimed.attempt == 2
    snapshot = store.get_run(run_id)
    assert snapshot.run.status == RunStatus.RUNNING
    assert snapshot.tasks[0].status == TaskStatus.RUNNING
    assert snapshot.tasks[0].lease_owner == "recovery-worker"
    with pytest.raises(RuntimeError, match="no longer owns"):
        store.complete_task(task.id, "abandoned-worker", [])

    with psycopg.connect(isolated_postgres_dsn) as connection:
        connection.execute(
            "UPDATE flakegraph_task SET lease_expires_at = "
            "CURRENT_TIMESTAMP - INTERVAL '10 minutes', "
            "updated_at = CURRENT_TIMESTAMP - INTERVAL '10 minutes' "
            "WHERE id = %s",
            (task.id,),
        )

    assert (
        store.claim_task(
            "third-worker",
            {TaskStage.PREPARE_DOCUMENT},
            timedelta(minutes=1),
        )
        is None
    )
    failed = store.get_run(run_id)
    assert failed.run.status == RunStatus.FAILED
    assert failed.tasks[0].status == TaskStatus.FAILED
    assert failed.tasks[0].attempts == 2
    assert failed.tasks[0].last_error == {
        "type": "LeaseExpired",
        "message": "worker lease expired after retry budget was exhausted",
    }


def test_postgres_final_attempt_can_complete_during_expiry_recovery_window(
    isolated_postgres_dsn: str,
) -> None:
    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    task = _task(
        run_id,
        "prepare",
        TaskStage.PREPARE_DOCUMENT,
        "file",
        max_attempts=1,
    )
    store.add_tasks(run_id, [task])
    store.activate_run(run_id)
    lease = store.claim_task("slow-worker", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert lease is not None
    with psycopg.connect(isolated_postgres_dsn) as connection:
        connection.execute(
            "UPDATE flakegraph_task SET lease_expires_at = "
            "CURRENT_TIMESTAMP - INTERVAL '1 minute', "
            "updated_at = CURRENT_TIMESTAMP - INTERVAL '1 minute' WHERE id = %s",
            (task.id,),
        )

    assert (
        store.claim_task(
            "recovery-worker",
            {TaskStage.PREPARE_DOCUMENT},
            timedelta(minutes=1),
        )
        is None
    )
    store.complete_task(task.id, "slow-worker", [])

    assert store.get_run(run_id).tasks[0].status == TaskStatus.SUCCEEDED


def test_postgres_reclaims_an_expired_lease_without_losing_priority(
    isolated_postgres_dsn: str,
) -> None:
    """Compare queued and expired candidates while retaining global task priority."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    high = _task(run_id, "high", TaskStage.PREPARE_DOCUMENT, "high").model_copy(
        update={"priority": 10}
    )
    low = _task(run_id, "low", TaskStage.PREPARE_DOCUMENT, "low").model_copy(update={"priority": 0})
    store.add_tasks(run_id, [high, low])
    store.activate_run(run_id)

    first = store.claim_task("first-worker", {TaskStage.PREPARE_DOCUMENT}, timedelta(minutes=1))
    assert first is not None and first.task.id == high.id and first.attempt == 1
    with psycopg.connect(isolated_postgres_dsn) as connection:
        connection.execute(
            "UPDATE flakegraph_task SET lease_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 minute' "
            "WHERE id = %s",
            (high.id,),
        )

    reclaimed = store.claim_task(
        "recovery-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert reclaimed is not None
    assert reclaimed.task.id == high.id
    assert reclaimed.attempt == 2


def test_postgres_dynamic_windows_extend_final_barrier_atomically(
    isolated_postgres_dsn: str,
) -> None:
    """Finalization must wait for the document task that joins every window."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    prepare = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    final = _task(
        run_id,
        "final",
        TaskStage.FINALIZE_GRAPH,
        "graph",
    )
    store.add_tasks(run_id, [prepare, final])
    store.activate_run(run_id)

    parent = store.claim_task(
        "prepare-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert parent is not None
    context = _task(
        run_id,
        "context",
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
        "file",
        dependency_ids=[prepare.id],
    )
    store.complete_task(parent.task.id, parent.worker_id, [], [context], final.id)
    context_claim = store.claim_task(
        "context-worker",
        {TaskStage.EXTRACT_DOCUMENT_CONTEXT},
        timedelta(minutes=1),
    )
    assert context_claim is not None
    windows = [
        _task(
            run_id,
            f"window-{index}",
            TaskStage.EXTRACT_ENTITY_WINDOW,
            f"window-{index}",
            dependency_ids=[context.id],
        )
        for index in range(4)
    ]
    inventory = _task(
        run_id,
        "inventory",
        TaskStage.COMPACT_ENTITY_INVENTORY,
        "file",
        dependency_ids=[context.id, *(window.id for window in windows)],
    )
    store.complete_task(context.id, context_claim.worker_id, [], [*windows, inventory], final.id)

    assert (
        store.claim_task(
            "final-worker",
            {TaskStage.FINALIZE_GRAPH},
            timedelta(minutes=1),
        )
        is None
    )
    claimed_windows = [
        store.claim_task(
            f"extract-worker-{index}",
            {TaskStage.EXTRACT_ENTITY_WINDOW},
            timedelta(minutes=1),
        )
        for index in range(4)
    ]
    assert all(claim is not None for claim in claimed_windows)
    assert len({claim.task.id for claim in claimed_windows if claim is not None}) == 4
    for claim in claimed_windows:
        assert claim is not None
        store.complete_task(claim.task.id, claim.worker_id, [])

    assert (
        store.claim_task(
            "final-worker",
            {TaskStage.FINALIZE_GRAPH},
            timedelta(minutes=1),
        )
        is None
    )
    inventory_claim = store.claim_task(
        "inventory-worker",
        {TaskStage.COMPACT_ENTITY_INVENTORY},
        timedelta(minutes=1),
    )
    assert inventory_claim is not None
    assert set(inventory_claim.dependency_outputs) == {
        context.id,
        *(window.id for window in windows),
    }
    relation_windows = [
        _task(
            run_id,
            f"relation-{index}",
            TaskStage.EXTRACT_RELATION_WINDOW,
            f"window-{index}",
            dependency_ids=[inventory.id],
        )
        for index in range(4)
    ]
    compact = _task(
        run_id,
        "compact",
        TaskStage.COMPACT_DOCUMENT,
        "file",
        dependency_ids=[inventory.id, *(window.id for window in relation_windows)],
    )
    store.complete_task(
        inventory_claim.task.id,
        inventory_claim.worker_id,
        [],
        [*relation_windows, compact],
        final.id,
    )

    claimed_relations = [
        store.claim_task(
            f"relation-worker-{index}",
            {TaskStage.EXTRACT_RELATION_WINDOW},
            timedelta(minutes=1),
        )
        for index in range(4)
    ]
    assert all(claim is not None for claim in claimed_relations)
    for claim in claimed_relations:
        assert claim is not None
        store.complete_task(claim.task.id, claim.worker_id, [])

    compact_claim = store.claim_task(
        "compact-worker",
        {TaskStage.COMPACT_DOCUMENT},
        timedelta(minutes=1),
    )
    assert compact_claim is not None
    store.complete_task(compact_claim.task.id, compact_claim.worker_id, [])

    final_claim = store.claim_task(
        "final-worker",
        {TaskStage.FINALIZE_GRAPH},
        timedelta(minutes=1),
    )
    assert final_claim is not None
    # Run-wide finalization discovers stage artifacts through the artifact store;
    # scheduling therefore stays constant-size regardless of document count.
    assert final_claim.dependency_outputs == {}


def test_spark_finalizer_claim_keeps_barrier_without_loading_outputs(
    isolated_postgres_dsn: str,
) -> None:
    """Avoid materializing corpus-sized dependency maps in a Spark driver lease."""

    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    prepare = _task(run_id, "prepare", TaskStage.PREPARE_DOCUMENT, "file")
    final = _task(
        run_id,
        "final",
        TaskStage.FINALIZE_GRAPH,
        "graph",
        dependency_ids=[prepare.id],
    )
    store.add_tasks(run_id, [prepare, final])
    store.activate_run(run_id)

    assert (
        store.claim_task(
            "spark-driver",
            {TaskStage.FINALIZE_GRAPH},
            timedelta(minutes=1),
            skip_dependency_outputs_for_stages={TaskStage.FINALIZE_GRAPH},
        )
        is None
    )
    parent = store.claim_task(
        "prepare-worker",
        {TaskStage.PREPARE_DOCUMENT},
        timedelta(minutes=1),
    )
    assert parent is not None
    artifact = store.put(
        run_id,
        ArtifactKind.PREPARED_DOCUMENT,
        b"{}",
        "application/json",
    )
    store.complete_task(parent.task.id, parent.worker_id, [artifact.id])

    final_claim = store.claim_task(
        "spark-driver",
        {TaskStage.FINALIZE_GRAPH},
        timedelta(minutes=1),
        skip_dependency_outputs_for_stages={TaskStage.FINALIZE_GRAPH},
    )
    assert final_claim is not None
    assert final_claim.dependency_outputs == {}
    assert final_claim.task.dependency_ids == []


def test_graph_manifest_and_active_version_publish_in_one_transaction(
    isolated_postgres_dsn: str,
    tmp_path: Path,
) -> None:
    """A successful final task must activate only its complete immutable artifact."""

    blob_store = LocalBlobStore((tmp_path / "objects").as_uri())
    store = PostgresDistributedStore(isolated_postgres_dsn, blob_store=blob_store)
    store.initialize()
    run_id = f"run_{uuid4().hex}"
    store.create_run(_run(run_id))
    final = _task(run_id, "final", TaskStage.FINALIZE_GRAPH, "integration-graph")
    store.add_tasks(run_id, [final])
    store.activate_run(run_id)
    claim = store.claim_task("final-worker", {TaskStage.FINALIZE_GRAPH}, timedelta(minutes=1))
    assert claim is not None
    payload = b'{"format":"flakegraph.graph-dataset-manifest/v1"}'
    artifact = store.put(
        run_id,
        ArtifactKind.GRAPH_RESULT,
        payload,
        "application/vnd.flakegraph.graph-manifest+json",
    )

    store.complete_task(claim.task.id, claim.worker_id, [artifact.id])

    with psycopg.connect(isolated_postgres_dsn, row_factory=dict_row) as connection:
        head = connection.execute(
            """
            SELECT head.graph_id, version.run_id, version.artifact_id
            FROM flakegraph_graph_head AS head
            JOIN flakegraph_graph_version AS version ON version.id = head.version_id
            """
        ).fetchone()
    assert head == {
        "graph_id": "integration-graph",
        "run_id": run_id,
        "artifact_id": artifact.id,
    }
    assert store.get(artifact.id).payload == payload
    with pytest.raises(ValueError, match="active graph version"):
        store.delete_run_artifacts(run_id)


def test_multiple_workers_drain_dynamic_pipeline_end_to_end(
    isolated_postgres_dsn: str,
    tmp_path: Path,
) -> None:
    """Workers should pull all discovered windows and publish one complete graph."""

    input_path = tmp_path / "input"
    input_path.mkdir()
    for index in range(4):
        (input_path / f"source-{index}.txt").write_text(
            " ".join(
                [
                    f"Teacher {index} founded School {index} in City {index}.",
                    f"School {index} teaches Method {index} to Student {index}.",
                ]
                * 8
            ),
            encoding="utf-8",
        )
    settings = Settings.load(
        env={},
        overrides={
            "runtime": {"runtime": "kubernetes"},
            "job": {"job_id": "distributed-test", "graph_id": "distributed-test-graph"},
            "files": {"source": "local", "input_path": input_path},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "model": "hash", "dimension": 8},
            "graph": {
                "chunk_token_size": 30,
                "chunk_token_overlap": 5,
                "max_chunks_per_llm_call": 1,
                "extraction_parallelism": 1,
                "gleaning_max_passes": 0,
                "fail_on_quality_error": False,
            },
            "writer": {
                "provider": "local_artifacts",
                "output_path": tmp_path / "output",
            },
            "cache": {"provider": "none"},
            "distributed": {
                "database_url": isolated_postgres_dsn,
                "poll_interval_seconds": 0.01,
            },
        },
    )
    store = _store(isolated_postgres_dsn)
    run_id = f"run_{uuid4().hex}"
    DistributedRunPlanner(
        settings,
        LocalFileSource(input_path),
        store,
        store,
    ).submit(run_id)
    workers = [
        DistributedWorker(
            settings,
            f"worker-{index}",
            set(TaskStage),
            build_pipeline(settings),
            store,
            store,
        )
        for index in range(6)
    ]

    for _iteration in range(100):
        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            list(executor.map(lambda worker: worker.process_one(), workers))
        snapshot = store.get_run(run_id)
        if snapshot.run.status == RunStatus.SUCCEEDED:
            break
        assert snapshot.run.status != RunStatus.FAILED
    else:
        pytest.fail("distributed workers did not drain the queue within 100 iterations")

    snapshot = store.get_run(run_id)
    preparation_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.PREPARE_DOCUMENT
    ]
    context_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.EXTRACT_DOCUMENT_CONTEXT
    ]
    entity_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.EXTRACT_ENTITY_WINDOW
    ]
    inventory_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.COMPACT_ENTITY_INVENTORY
    ]
    relation_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.EXTRACT_RELATION_WINDOW
    ]
    compact_tasks = [
        task for task in snapshot.tasks if task.task.stage == TaskStage.COMPACT_DOCUMENT
    ]
    final_tasks = [task for task in snapshot.tasks if task.task.stage == TaskStage.FINALIZE_GRAPH]
    assert len(preparation_tasks) == 4
    assert len(context_tasks) == len(preparation_tasks)
    assert len(entity_tasks) > len(preparation_tasks)
    assert len(relation_tasks) == len(entity_tasks)
    assert len(inventory_tasks) == len(preparation_tasks)
    assert len(compact_tasks) == len(preparation_tasks)
    assert len(final_tasks) == 1
    assert all(task.status == TaskStatus.SUCCEEDED for task in snapshot.tasks)
    assert len(final_tasks[0].output_artifact_ids) == 1
    graph = store.get(final_tasks[0].output_artifact_ids[0])
    assert graph.ref.kind == ArtifactKind.GRAPH_RESULT
    assert (tmp_path / "output" / "run_report.json").is_file()


def _worker_demand(dsn: str) -> dict[str, int]:
    """Read the database contract consumed by KEDA's PostgreSQL scaler."""

    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        rows = connection.execute(
            "SELECT stage, desired_workers FROM flakegraph_worker_demand ORDER BY stage"
        ).fetchall()
    return {str(row["stage"]): int(row["desired_workers"]) for row in rows}


def _store(dsn: str) -> PostgresDistributedStore:
    """Return an initialized adapter using the opt-in integration DSN."""

    store = PostgresDistributedStore(dsn)
    store.initialize()
    return store


def _run(run_id: str) -> RunDefinition:
    """Build one unique live-test run definition."""

    return RunDefinition(
        id=run_id,
        graph_id="integration-graph",
        config={"test": True},
        config_digest="integration-config",
    )


def _task(
    run_id: str,
    suffix: str,
    stage: TaskStage,
    scope_id: str,
    *,
    dependency_ids: list[str] | None = None,
    max_attempts: int = 3,
) -> TaskDefinition:
    """Build a task whose identity is isolated to the current live-test run."""

    return TaskDefinition(
        id=f"{run_id}-{suffix}",
        run_id=run_id,
        stage=stage,
        scope_id=scope_id,
        dependency_ids=dependency_ids or [],
        max_attempts=max_attempts,
    )
