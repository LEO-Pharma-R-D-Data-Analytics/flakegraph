"""PostgreSQL task and artifact store for production distributed execution.

The adapter uses short transactions and row-level leases. Long OCR or LLM work
never holds a database lock: workers claim, commit, process externally, then finish
through an ownership-checked transition. Expired leases are safely reclaimable and
immutable artifact checksums protect stage handoffs from silent corruption.
"""

from __future__ import annotations

import zlib
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta
from threading import Lock
from typing import Any, cast
from weakref import finalize

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from kg_processor.domain.distributed import (
    ArtifactKind,
    ArtifactRef,
    PublicationLease,
    PublicationStatus,
    RunDefinition,
    RunOverview,
    RunSnapshot,
    RunStatus,
    RunSummary,
    StoredArtifact,
    TaskCount,
    TaskDefinition,
    TaskLease,
    TaskProgress,
    TaskSnapshot,
    TaskStage,
    TaskStatus,
)
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.ports.blob_store import BatchBlobDeleter, BlobStore
from kg_processor.ports.task_store import TaskStoreUnavailableError

_TERMINAL_RUN_STATUSES = {
    RunStatus.SUCCEEDED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
}
_MAX_ZLIB_COMPRESSION_LEVEL = 9
_ARTIFACT_READ_PARALLELISM = 8
_ARTIFACT_DELETE_BATCH_SIZE = 1_000
_INITIAL_TASK_COPY_BATCH_SIZE = 2_000
_MAX_RUN_LIST_LIMIT = 500
_SCHEMA_VERSION = 7
_EXHAUSTED_TASK_RECOVERY_GRACE_SECONDS = 300
_POSTGRES_SESSION_OPTIONS = " ".join(
    (
        "-c lock_timeout=30000",
        "-c idle_in_transaction_session_timeout=60000",
        "-c client_connection_check_interval=1000",
    )
)
# PostgreSQL can race while two sessions concurrently create the same table's
# implicit composite type even with IF NOT EXISTS. A stable transaction-scoped
# advisory lock serializes only schema initialization, never normal task work.
_SCHEMA_INITIALIZATION_LOCK_ID = 2_359_716_441


class PostgresDistributedStore:
    """Implement durable task and artifact ports over one PostgreSQL database.

    One database keeps task completion and artifact references transactionally
    consistent while the interfaces remain separate at application boundaries.
    Payloads are compressed before insertion; large source systems can later use a
    different ``ArtifactStore`` without changing coordinator or worker semantics.
    """

    def __init__(
        self,
        dsn: str,
        *,
        compression_level: int = 6,
        max_artifact_bytes: int = 512 * 1024 * 1024,
        blob_store: BlobStore | None = None,
    ) -> None:
        """Store connection and bounded artifact settings without opening a socket."""

        if not dsn.strip():
            raise ValueError("PostgreSQL DSN must not be blank")
        if not 0 <= compression_level <= _MAX_ZLIB_COMPRESSION_LEVEL:
            raise ValueError("artifact compression level must be between 0 and 9")
        if max_artifact_bytes <= 0:
            raise ValueError("max artifact bytes must be positive")
        self.dsn = dsn
        self.compression_level = compression_level
        self.max_artifact_bytes = max_artifact_bytes
        self.blob_store = blob_store
        # A planner creates its run before concurrently staging sources, and a
        # worker can write only for a run it successfully leased. Remember those
        # database-proven identities so every immutable artifact does not spend a
        # separate transaction rechecking the same foreign-key precondition.
        self._known_run_ids: set[str] = set()
        self._known_run_ids_lock = Lock()
        try:
            configured_options = str(psycopg.conninfo.conninfo_to_dict(dsn).get("options") or "")
        except psycopg.ProgrammingError:
            raise ValueError("PostgreSQL DSN is malformed") from None
        session_options = " ".join(
            value for value in (configured_options, _POSTGRES_SESSION_OPTIONS) if value
        )
        # A worker performs several short transactions per task (claim, artifact
        # metadata, completion, and heartbeats). Reusing a small connection pool
        # removes a network/TLS handshake from each transition while preserving
        # the adapter's existing transaction boundary around every method call.
        self._pool = cast(
            "ConnectionPool[psycopg.Connection[dict[str, Any]]]",
            ConnectionPool(
                conninfo=dsn,
                min_size=0,
                # A worker executes one task at a time; two connections cover its
                # main transition and concurrent lease heartbeat. Keeping a larger
                # idle pool in every autoscaled pod needlessly consumes the shared
                # PostgreSQL connection budget on multi-node fleets.
                max_size=2,
                max_idle=60,
                timeout=10,
                open=True,
                kwargs={
                    "row_factory": dict_row,
                    "connect_timeout": 10,
                    "application_name": "flakegraph",
                    # Queue transactions should never wait indefinitely behind a
                    # lost worker. PostgreSQL enforces these limits even when a
                    # process disappears mid-COPY: blocked transitions retry after
                    # 30 seconds, abandoned transactions close after one minute,
                    # and active statements check their client socket every second.
                    # No statement timeout is imposed because streaming an initial
                    # 250,000-document plan can legitimately take several minutes.
                    "options": session_options,
                },
            ),
        )
        # Most stores live for the worker process lifetime. The finalizer still
        # closes retained connections when a short-lived CLI/test store is
        # collected, without requiring connection lifecycle on the port itself.
        self._pool_finalizer = finalize(self, self._pool.close)

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        """Yield a transaction or translate transient driver connectivity errors."""

        try:
            with self._pool.connection() as connection:
                yield connection
        except psycopg.OperationalError, PoolTimeout:
            # Driver messages can echo a full conninfo string. Suppress exception
            # chaining so passwords cannot reappear in CLI or pod tracebacks.
            raise TaskStoreUnavailableError(
                "PostgreSQL coordination store is unavailable"
            ) from None

    def close(self) -> None:
        """Release pooled database connections held by this adapter instance.

        Worker processes normally rely on process shutdown, while tests and
        embedded callers may close a store deterministically. Closing is
        idempotent and does not alter durable run, task, or artifact state.
        """

        self._pool_finalizer()

    def initialize(self) -> None:
        """Apply migrations once and make ordinary replica startup a cheap check."""

        if self.blob_store is not None:
            initialize = getattr(self.blob_store, "initialize", None)
            if callable(initialize):
                initialize()

        with self._connection() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (_SCHEMA_INITIALIZATION_LOCK_ID,),
            )
            version_table = connection.execute(
                "SELECT to_regclass('flakegraph_schema_version') AS relation"
            ).fetchone()
            if version_table is not None and version_table["relation"] is not None:
                version_row = connection.execute(
                    "SELECT version FROM flakegraph_schema_version WHERE singleton = TRUE"
                ).fetchone()
                if version_row is not None:
                    installed_version = int(version_row["version"])
                    if installed_version > _SCHEMA_VERSION:
                        raise RuntimeError(
                            "coordination schema is newer than this FlakeGraph build"
                        )
                    if installed_version == _SCHEMA_VERSION:
                        return
            for statement in _SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS flakegraph_schema_version (
                    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
                    version INTEGER NOT NULL CHECK (version > 0),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                INSERT INTO flakegraph_schema_version (singleton, version)
                VALUES (TRUE, %s)
                ON CONFLICT (singleton) DO UPDATE
                SET version = EXCLUDED.version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (_SCHEMA_VERSION,),
            )

    def create_run(self, run: RunDefinition) -> None:
        """Insert a planning run idempotently while rejecting identity collisions."""

        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO flakegraph_run (
                    id, graph_id, config_json, config_digest, status
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    run.id,
                    run.graph_id,
                    Jsonb(run.config),
                    run.config_digest,
                    run.status.value,
                ),
            )
            existing = connection.execute(
                """
                SELECT graph_id, config_digest, status
                FROM flakegraph_run
                WHERE id = %s
                """,
                (run.id,),
            ).fetchone()
            if existing is None:
                raise RuntimeError(f"failed to create distributed run {run.id}")
            if (
                str(existing["graph_id"]) != run.graph_id
                or str(existing["config_digest"]) != run.config_digest
            ):
                raise ValueError(f"run id {run.id} already exists with different inputs")
        self._remember_run_id(run.id)

    def add_tasks(self, run_id: str, tasks: list[TaskDefinition]) -> None:
        """Insert a complete dependency graph after validating run ownership."""

        if not tasks:
            raise ValueError("a distributed run requires at least one task")
        task_ids = {task.id for task in tasks}
        if len(task_ids) != len(tasks):
            raise ValueError("distributed task ids must be unique")
        if any(task.run_id != run_id for task in tasks):
            raise ValueError("all tasks must belong to the selected run")
        # Production submission has a known bounded shape: independent preparation
        # tasks plus one run-wide final barrier. Recognizing that shape avoids
        # allocating two additional O(V + E) adjacency mappings for a
        # 250,000-document plan.
        # Arbitrary plans retain the full cycle check.
        if not _is_initial_document_plan(tasks, task_ids=task_ids):
            if any(set(task.dependency_ids) - task_ids for task in tasks):
                raise ValueError("task dependencies must reference tasks in the same plan")
            _validate_acyclic(tasks)
        with self._connection() as connection:
            status_row = connection.execute(
                "SELECT status FROM flakegraph_run WHERE id = %s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if status_row is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            if str(status_row["status"]) != RunStatus.PLANNING.value:
                raise ValueError("tasks may only be added while a run is planning")
            count_row = connection.execute(
                "SELECT COUNT(*) AS count FROM flakegraph_task WHERE run_id = %s",
                (run_id,),
            ).fetchone()
            if count_row is None:
                raise RuntimeError(f"failed to inspect task plan for run {run_id}")
            existing_count = cast(int, count_row["count"])
            if existing_count:
                raise ValueError(f"distributed run {run_id} already has a task plan")
            # COPY keeps one round trip and one server-side parse for the complete
            # initial plan. At hundreds of thousands of documents, issuing one
            # INSERT per task and dependency would otherwise dominate submission.
            with connection.cursor().copy(
                """
                COPY flakegraph_task (
                    id, run_id, stage, scope_id, payload_json, status,
                    priority, max_attempts, remaining_dependencies
                ) FROM STDIN
                """
            ) as task_copy:
                for task in tasks:
                    task_copy.write_row(
                        (
                            task.id,
                            task.run_id,
                            task.stage.value,
                            task.scope_id,
                            Jsonb(task.payload),
                            TaskStatus.QUEUED.value,
                            task.priority,
                            task.max_attempts,
                            len(task.dependency_ids),
                        )
                    )
            dependencies = (
                (task.id, dependency_id) for task in tasks for dependency_id in task.dependency_ids
            )
            first_dependency = next(dependencies, None)
            if first_dependency is not None:
                with connection.cursor().copy(
                    """
                    COPY flakegraph_task_dependency (task_id, depends_on_task_id)
                    FROM STDIN
                    """
                ) as dependency_copy:
                    dependency_copy.write_row(first_dependency)
                    for dependency in dependencies:
                        dependency_copy.write_row(dependency)

    def add_initial_tasks(
        self,
        run_id: str,
        tasks: Iterable[TaskDefinition],
    ) -> None:
        """Stream the restricted initial plan through short PostgreSQL transactions.

        Unlike ``add_tasks``, this entry point intentionally accepts only
        independent document preparation tasks and one final barrier. That known
        acyclic shape can be validated row by row, letting PostgreSQL's ``COPY``
        consume a 250,000-document plan without an equally large Python list. The
        run remains in ``planning`` until every batch is committed, so workers
        cannot observe a partial plan. Crucially, source uploads happen while no
        database transaction or pooled connection is retained.
        """

        with self._connection() as connection:
            status_row = connection.execute(
                "SELECT status FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
            if status_row is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            if str(status_row["status"]) != RunStatus.PLANNING.value:
                raise ValueError("tasks may only be added while a run is planning")
            # Existing rows are a resumable prefix left by an interrupted
            # coordinator. Every replayed row is checked for exact identity in
            # ``_copy_initial_task_batch`` before missing rows are inserted.

        preparation_count = 0
        final_count = 0
        final_seen = False
        batch: list[TaskDefinition] = []
        for task in tasks:
            preparation_increment, final_increment = _validate_initial_task(
                task,
                run_id,
                final_seen,
            )
            preparation_count += preparation_increment
            final_count += final_increment
            final_seen = bool(final_count)
            batch.append(task)
            if len(batch) >= _INITIAL_TASK_COPY_BATCH_SIZE:
                self._copy_initial_task_batch(run_id, batch)
                batch = []
        if batch:
            self._copy_initial_task_batch(run_id, batch)
        if preparation_count == 0 or final_count != 1:
            raise ValueError("initial plan requires preparation tasks and exactly one final task")
        with self._connection() as connection:
            counts = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE stage = %s) AS preparations,
                       COUNT(*) FILTER (WHERE stage = %s) AS finalizers
                FROM flakegraph_task
                WHERE run_id = %s
                """,
                (
                    TaskStage.PREPARE_DOCUMENT.value,
                    TaskStage.FINALIZE_GRAPH.value,
                    run_id,
                ),
            ).fetchone()
            if (
                counts is None
                or int(counts["total"]) != preparation_count + final_count
                or int(counts["preparations"]) != preparation_count
                or int(counts["finalizers"]) != 1
            ):
                raise ValueError("streamed initial plan conflicts with existing planning rows")
        # COPY can add hundreds of thousands of rows before autovacuum has a
        # chance to refresh planner statistics. Analyze while the run is still
        # invisible so the first worker immediately selects the ordered claim
        # index instead of sorting the complete ready queue.
        with self._connection() as connection:
            connection.execute("ANALYZE flakegraph_task")

    def _copy_initial_task_batch(
        self,
        run_id: str,
        tasks: list[TaskDefinition],
    ) -> None:
        """Commit one bounded initial-plan batch while the run stays invisible."""

        with self._connection() as connection:
            status = connection.execute(
                "SELECT status FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
            if status is None or str(status["status"]) != RunStatus.PLANNING.value:
                raise ValueError("run left planning state while initial tasks were streamed")
            connection.execute(
                """
                CREATE TEMPORARY TABLE flakegraph_initial_task_batch (
                    id TEXT, run_id TEXT, stage TEXT, scope_id TEXT,
                    payload_json JSONB, status TEXT, priority INTEGER,
                    max_attempts INTEGER, remaining_dependencies INTEGER
                ) ON COMMIT DROP
                """
            )
            with connection.cursor().copy(
                """
                COPY flakegraph_initial_task_batch (
                    id, run_id, stage, scope_id, payload_json, status,
                    priority, max_attempts, remaining_dependencies
                ) FROM STDIN
                """
            ) as task_copy:
                for task in tasks:
                    task_copy.write_row(
                        (
                            task.id,
                            task.run_id,
                            task.stage.value,
                            task.scope_id,
                            Jsonb(task.payload),
                            TaskStatus.QUEUED.value,
                            task.priority,
                            task.max_attempts,
                            0,
                        )
                    )
            conflict = connection.execute(
                """
                SELECT incoming.id
                FROM flakegraph_initial_task_batch AS incoming
                JOIN flakegraph_task AS existing
                  ON existing.id = incoming.id
                  OR (
                      existing.run_id = incoming.run_id
                      AND existing.stage = incoming.stage
                      AND existing.scope_id = incoming.scope_id
                  )
                WHERE existing.id <> incoming.id
                   OR existing.run_id <> incoming.run_id
                   OR existing.stage <> incoming.stage
                   OR existing.scope_id <> incoming.scope_id
                   OR existing.payload_json <> incoming.payload_json
                   OR existing.priority <> incoming.priority
                   OR existing.max_attempts <> incoming.max_attempts
                LIMIT 1
                """
            ).fetchone()
            if conflict is not None:
                raise ValueError(
                    f"streamed initial task conflicts with existing row: {conflict['id']}"
                )
            connection.execute(
                """
                INSERT INTO flakegraph_task (
                    id, run_id, stage, scope_id, payload_json, status,
                    priority, max_attempts, remaining_dependencies
                )
                SELECT id, run_id, stage, scope_id, payload_json, status,
                       priority, max_attempts, remaining_dependencies
                FROM flakegraph_initial_task_batch
                ON CONFLICT (id) DO NOTHING
                """
            )

    def activate_run(self, run_id: str) -> None:
        """Transition a fully planned run to queued exactly once."""

        with self._connection() as connection:
            updated = connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                  AND status = %s
                  AND EXISTS (
                      SELECT 1 FROM flakegraph_task WHERE run_id = flakegraph_run.id
                  )
                RETURNING id
                """,
                (RunStatus.QUEUED.value, run_id, RunStatus.PLANNING.value),
            ).fetchone()
            if updated is None:
                raise ValueError("run must be planning with a non-empty task graph")

    def claim_task(
        self,
        worker_id: str,
        stages: set[TaskStage],
        lease_duration: timedelta,
        expected_config_digest: str | None = None,
        skip_dependency_outputs_for_stages: set[TaskStage] | None = None,
    ) -> TaskLease | None:
        """Claim one dependency-ready task using row locks that skip competitors.

        A skipped stage retains the exact SQL dependency barrier but does not load
        prerequisite IDs or outputs into Python. Spark finalization reads run-scoped
        immutable object prefixes, so transferring millions of redundant artifact
        references to its driver would add latency and an avoidable memory spike.
        """

        if not worker_id.strip():
            raise ValueError("worker id must not be blank")
        if not stages:
            raise ValueError("worker must accept at least one task stage")
        _validate_duration(lease_duration, "lease duration")
        with self._connection() as connection:
            self._fail_abandoned_exhausted_tasks(connection)
            row = connection.execute(
                """
                WITH candidate AS (
                    SELECT task.id
                    FROM flakegraph_task AS task
                    JOIN flakegraph_run AS run ON run.id = task.run_id
                    WHERE run.status IN (%s, %s)
                      AND task.stage = ANY(%s)
                      AND (CAST(%s AS TEXT) IS NULL OR run.config_digest = %s)
                      AND task.available_at <= CURRENT_TIMESTAMP
                      AND (
                          task.status = %s
                          OR (
                              task.status = %s
                              AND task.lease_expires_at < CURRENT_TIMESTAMP
                              AND task.attempts < task.max_attempts
                          )
                      )
                      AND (
                          (
                              task.stage <> %s
                              AND task.remaining_dependencies = 0
                          )
                          OR (
                              task.stage = %s
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM flakegraph_task AS unfinished
                                  WHERE unfinished.run_id = task.run_id
                                    AND unfinished.id <> task.id
                                    AND unfinished.status <> %s
                              )
                          )
                      )
                    ORDER BY task.priority DESC, task.created_at, task.id
                    FOR UPDATE OF task SKIP LOCKED
                    LIMIT 1
                )
                UPDATE flakegraph_task AS task
                SET status = %s,
                    attempts = task.attempts + 1,
                    lease_owner = %s,
                    lease_expires_at = CURRENT_TIMESTAMP + %s,
                    started_at = COALESCE(task.started_at, CURRENT_TIMESTAMP),
                    completed_at = NULL,
                    progress_json = NULL,
                    progress_updated_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                FROM candidate
                WHERE task.id = candidate.id
                RETURNING task.*
                """,
                (
                    RunStatus.QUEUED.value,
                    RunStatus.RUNNING.value,
                    [stage.value for stage in sorted(stages, key=str)],
                    expected_config_digest,
                    expected_config_digest,
                    TaskStatus.QUEUED.value,
                    TaskStatus.RUNNING.value,
                    TaskStage.FINALIZE_GRAPH.value,
                    TaskStage.FINALIZE_GRAPH.value,
                    TaskStatus.SUCCEEDED.value,
                    TaskStatus.RUNNING.value,
                    worker_id,
                    lease_duration,
                ),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = %s
                """,
                (RunStatus.RUNNING.value, row["run_id"], RunStatus.QUEUED.value),
            )
            claimed_stage = TaskStage(str(row["stage"]))
            skip_outputs = claimed_stage in (skip_dependency_outputs_for_stages or set())
            dependency_outputs = (
                {} if skip_outputs else self._dependency_outputs(connection, str(row["id"]))
            )
            self._remember_run_id(str(row["run_id"]))
            return TaskLease(
                task=_task_definition(row, dependency_outputs.keys()),
                worker_id=worker_id,
                attempt=int(row["attempts"]),
                lease_expires_at=row["lease_expires_at"],
                dependency_outputs=dependency_outputs,
            )

    def heartbeat(self, task_id: str, worker_id: str, lease_duration: timedelta) -> None:
        """Extend an owned running lease without changing its attempt count."""

        _validate_duration(lease_duration, "lease duration")
        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE flakegraph_task
                SET lease_expires_at = CURRENT_TIMESTAMP + %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = %s AND lease_owner = %s
                RETURNING id
                """,
                (lease_duration, task_id, TaskStatus.RUNNING.value, worker_id),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"worker {worker_id} no longer owns task {task_id}")

    def report_task_progress(
        self,
        task_id: str,
        worker_id: str,
        progress: TaskProgress,
    ) -> None:
        """Persist bounded progress only while the worker owns the running lease."""

        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE flakegraph_task
                SET progress_json = %s,
                    progress_updated_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = %s AND lease_owner = %s
                RETURNING run_id
                """,
                (
                    Jsonb(progress.model_dump(mode="json")),
                    task_id,
                    TaskStatus.RUNNING.value,
                    worker_id,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"worker {worker_id} no longer owns task {task_id}")
            connection.execute(
                "UPDATE flakegraph_run SET updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                (row["run_id"],),
            )

    def claim_publication(
        self,
        worker_id: str,
        lease_duration: timedelta,
    ) -> PublicationLease | None:
        """Lease one durable external-publication command for a finalizer worker."""

        if not worker_id.strip():
            raise ValueError("worker id must not be blank")
        _validate_duration(lease_duration, "publication lease duration")
        with self._connection() as connection:
            expired = connection.execute(
                """
                UPDATE flakegraph_publication
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    last_error_json = %s, completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = %s
                  AND lease_expires_at < CURRENT_TIMESTAMP
                  AND attempts >= max_attempts
                RETURNING run_id, id
                """,
                (
                    PublicationStatus.FAILED.value,
                    Jsonb(
                        {
                            "type": "LeaseExpired",
                            "message": "publication lease expired",
                        }
                    ),
                    PublicationStatus.RUNNING.value,
                ),
            ).fetchall()
            for item in expired:
                connection.execute(
                    """
                    UPDATE flakegraph_run
                    SET status = %s, error_json = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND status = %s
                    """,
                    (
                        RunStatus.FAILED.value,
                        Jsonb(
                            {
                                "publication_id": str(item["id"]),
                                "type": "LeaseExpired",
                                "message": "publication lease expired",
                            }
                        ),
                        item["run_id"],
                        RunStatus.RUNNING.value,
                    ),
                )
            row = connection.execute(
                """
                WITH candidate AS (
                    SELECT publication.id
                    FROM flakegraph_publication AS publication
                    JOIN flakegraph_run AS run ON run.id = publication.run_id
                    WHERE run.status = %s
                      AND publication.available_at <= CURRENT_TIMESTAMP
                      AND (
                          publication.status = %s
                          OR (
                              publication.status = %s
                              AND publication.lease_expires_at < CURRENT_TIMESTAMP
                              AND publication.attempts < publication.max_attempts
                          )
                      )
                    ORDER BY publication.generation, publication.id
                    FOR UPDATE OF publication SKIP LOCKED
                    LIMIT 1
                )
                UPDATE flakegraph_publication AS publication
                SET status = %s,
                    attempts = publication.attempts + 1,
                    lease_owner = %s,
                    lease_expires_at = CURRENT_TIMESTAMP + %s,
                    updated_at = CURRENT_TIMESTAMP
                FROM candidate
                WHERE publication.id = candidate.id
                RETURNING publication.*
                """,
                (
                    RunStatus.RUNNING.value,
                    PublicationStatus.QUEUED.value,
                    PublicationStatus.RUNNING.value,
                    PublicationStatus.RUNNING.value,
                    worker_id,
                    lease_duration,
                ),
            ).fetchone()
            return self._publication_lease(row, worker_id) if row is not None else None

    def publish_claimed(
        self,
        publication_id: str,
        worker_id: str,
        publish: Callable[[PublicationLease], None],
    ) -> None:
        """Publish externally without holding coordination locks across the callback."""

        # Read and validate the fencing identity in a short transaction. The
        # generation and attempt are checked again when committing completion, so
        # cancellation and retries remain responsive while publication is slow.
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT publication.*
                FROM flakegraph_publication AS publication
                JOIN flakegraph_run AS run ON run.id = publication.run_id
                WHERE publication.id = %s
                  AND publication.status = %s
                  AND publication.lease_owner = %s
                  AND publication.lease_expires_at >= CURRENT_TIMESTAMP
                  AND run.status = %s
                """,
                (
                    publication_id,
                    PublicationStatus.RUNNING.value,
                    worker_id,
                    RunStatus.RUNNING.value,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"worker {worker_id} no longer owns publication {publication_id}"
                )
            lease = self._publication_lease(row, worker_id)
        publish(lease)
        with self._connection() as connection:
            completed = connection.execute(
                """
                UPDATE flakegraph_publication AS publication
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                FROM flakegraph_run AS run
                WHERE publication.id = %s
                  AND publication.run_id = run.id
                  AND publication.status = %s
                  AND publication.lease_owner = %s
                  AND publication.attempts = %s
                  AND publication.generation = %s
                  AND run.status = %s
                RETURNING publication.run_id
                """,
                (
                    PublicationStatus.SUCCEEDED.value,
                    publication_id,
                    PublicationStatus.RUNNING.value,
                    worker_id,
                    lease.attempt,
                    lease.generation,
                    RunStatus.RUNNING.value,
                ),
            ).fetchone()
            if completed is None:
                raise RuntimeError(
                    f"worker {worker_id} no longer owns publication {publication_id}"
                )
            self._complete_run_after_final_barrier(connection, lease.run_id)

    def heartbeat_publication(
        self,
        publication_id: str,
        worker_id: str,
        lease_duration: timedelta,
    ) -> None:
        """Extend an owned publication lease while its destination write is active."""

        _validate_duration(lease_duration, "publication lease duration")
        with self._connection() as connection:
            refreshed = connection.execute(
                """
                UPDATE flakegraph_publication
                SET lease_expires_at = CURRENT_TIMESTAMP + %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = %s AND lease_owner = %s
                  AND lease_expires_at >= CURRENT_TIMESTAMP
                RETURNING id
                """,
                (
                    lease_duration,
                    publication_id,
                    PublicationStatus.RUNNING.value,
                    worker_id,
                ),
            ).fetchone()
            if refreshed is None:
                raise RuntimeError(
                    f"worker {worker_id} no longer owns publication {publication_id}"
                )

    def fail_publication(
        self,
        publication_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
    ) -> None:
        """Retry an owned publication or terminally fail its parent run."""

        if retry_delay < timedelta(0):
            raise ValueError("retry delay must not be negative")
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM flakegraph_publication
                WHERE id = %s AND status = %s AND lease_owner = %s
                FOR UPDATE
                """,
                (publication_id, PublicationStatus.RUNNING.value, worker_id),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"worker {worker_id} no longer owns publication {publication_id}"
                )
            exhausted = int(row["attempts"]) >= int(row["max_attempts"])
            next_status = PublicationStatus.FAILED if exhausted else PublicationStatus.QUEUED
            connection.execute(
                """
                UPDATE flakegraph_publication
                SET status = %s, available_at = CURRENT_TIMESTAMP + %s,
                    lease_owner = NULL, lease_expires_at = NULL,
                    last_error_json = %s,
                    completed_at = CASE WHEN %s = 'failed'
                                        THEN CURRENT_TIMESTAMP ELSE NULL END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    next_status.value,
                    retry_delay,
                    Jsonb(error),
                    next_status.value,
                    publication_id,
                ),
            )
            if exhausted:
                connection.execute(
                    """
                    UPDATE flakegraph_run
                    SET status = %s, error_json = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND status = %s
                    """,
                    (
                        RunStatus.FAILED.value,
                        Jsonb({"publication_id": publication_id, **error}),
                        row["run_id"],
                        RunStatus.RUNNING.value,
                    ),
                )

    @staticmethod
    def _publication_lease(row: dict[str, Any], worker_id: str) -> PublicationLease:
        """Convert a durable outbox row into the callback-facing lease model."""

        return PublicationLease(
            id=str(row["id"]),
            run_id=str(row["run_id"]),
            artifact_id=str(row["artifact_id"]),
            task_payload=cast(dict[str, Any], row["task_payload_json"]),
            worker_id=worker_id,
            attempt=int(row["attempts"]),
            generation=int(row["generation"]),
            lease_expires_at=row["lease_expires_at"],
        )

    def complete_task(
        self,
        task_id: str,
        worker_id: str,
        output_artifact_ids: list[str],
        follow_up_tasks: list[TaskDefinition] | None = None,
        barrier_task_id: str | None = None,
    ) -> None:
        """Commit outputs and dynamically discovered work in one transaction.

        A running parent remains unfinished while follow-up rows and their local
        dependency edges are inserted. Marking the parent successful only
        afterward closes the race in which the run-wide finalizer could otherwise
        observe no unfinished work between parent completion and dynamic fan-out.
        """

        follow_ups = follow_up_tasks or []
        if len(set(output_artifact_ids)) != len(output_artifact_ids):
            raise ValueError("task output artifact ids must be unique")
        with self._connection() as connection:
            task = self._owned_task(connection, task_id, worker_id)
            run_id = str(task["run_id"])
            output_rows: list[dict[str, Any]] = []
            if output_artifact_ids:
                output_rows = connection.execute(
                    """
                    SELECT id, kind, media_type FROM flakegraph_artifact
                    WHERE run_id = %s AND id = ANY(%s)
                    """,
                    (run_id, output_artifact_ids),
                ).fetchall()
                if {str(row["id"]) for row in output_rows} != set(output_artifact_ids):
                    raise ValueError("task outputs must reference artifacts from the same run")
            if follow_ups:
                self._insert_follow_up_tasks(
                    connection,
                    run_id,
                    task_id,
                    follow_ups,
                    barrier_task_id,
                )
            elif barrier_task_id is not None:
                raise ValueError("barrier_task_id requires at least one follow-up task")
            for position, artifact_id in enumerate(output_artifact_ids):
                connection.execute(
                    """
                    INSERT INTO flakegraph_task_output (task_id, position, artifact_id)
                    VALUES (%s, %s, %s)
                    """,
                    (task_id, position, artifact_id),
                )
            connection.execute(
                """
                UPDATE flakegraph_task
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (TaskStatus.SUCCEEDED.value, task_id),
            )
            # Make dependent work ready in the same transaction that publishes
            # this task's outputs. Worker and autoscaler polls can then use one
            # indexed integer predicate instead of repeatedly joining the entire
            # dependency frontier for a large corpus.
            connection.execute(
                """
                UPDATE flakegraph_task AS dependent
                SET remaining_dependencies = dependent.remaining_dependencies - 1,
                    updated_at = CURRENT_TIMESTAMP
                FROM flakegraph_task_dependency AS dependency
                WHERE dependency.depends_on_task_id = %s
                  AND dependent.id = dependency.task_id
                  AND dependent.status = %s
                  AND dependent.stage <> %s
                  AND dependent.remaining_dependencies > 0
                """,
                (
                    task_id,
                    TaskStatus.QUEUED.value,
                    TaskStage.FINALIZE_GRAPH.value,
                ),
            )
            if str(task["stage"]) == TaskStage.FINALIZE_GRAPH.value:
                self._publish_graph_version(connection, run_id, output_rows)
                self._enqueue_graph_publication(connection, task, output_rows)
                self._complete_run_after_final_barrier(connection, run_id)
            else:
                self._complete_run_without_final_barrier(connection, run_id)

    def _complete_run_without_final_barrier(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        run_id: str,
    ) -> None:
        """Complete custom task graphs that deliberately omit a finalization barrier."""

        finalizer = connection.execute(
            """
            SELECT 1 FROM flakegraph_task
            WHERE run_id = %s AND stage = %s
            LIMIT 1
            """,
            (run_id, TaskStage.FINALIZE_GRAPH.value),
        ).fetchone()
        if finalizer is not None:
            return
        connection.execute(
            "SELECT id FROM flakegraph_run WHERE id = %s FOR UPDATE",
            (run_id,),
        )
        connection.execute(
            """
            UPDATE flakegraph_run AS run
            SET status = %s, updated_at = CURRENT_TIMESTAMP
            WHERE run.id = %s AND run.status = %s
              AND NOT EXISTS (
                  SELECT 1 FROM flakegraph_task AS finalizer
                  WHERE finalizer.run_id = run.id AND finalizer.stage = %s
              )
              AND NOT EXISTS (
                  SELECT 1 FROM flakegraph_task AS unfinished
                  WHERE unfinished.run_id = run.id AND unfinished.status <> %s
              )
            """,
            (
                RunStatus.SUCCEEDED.value,
                run_id,
                RunStatus.RUNNING.value,
                TaskStage.FINALIZE_GRAPH.value,
                TaskStatus.SUCCEEDED.value,
            ),
        )

    def _enqueue_graph_publication(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        task: dict[str, Any],
        output_rows: list[dict[str, Any]],
    ) -> None:
        """Atomically append an external-destination command with final task success."""

        payload = cast(dict[str, Any], task["payload_json"])
        output = payload.get("output")
        if not isinstance(output, dict) or output.get("provider") != "snowflake_bulk":
            return
        graph_outputs = [
            row for row in output_rows if str(row["kind"]) == ArtifactKind.GRAPH_RESULT.value
        ]
        if len(graph_outputs) != 1:
            raise ValueError("Snowflake publication requires exactly one graph result")
        publication_id = stable_id("publication", str(task["run_id"]), str(task["id"]))
        connection.execute(
            """
            INSERT INTO flakegraph_publication (
                id, run_id, task_id, artifact_id, task_payload_json,
                status, max_attempts
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                publication_id,
                task["run_id"],
                task["id"],
                graph_outputs[0]["id"],
                Jsonb(payload),
                PublicationStatus.QUEUED.value,
                task["max_attempts"],
            ),
        )

    def _complete_run_after_final_barrier(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        run_id: str,
    ) -> None:
        """Mark a run successful only after its leased final barrier succeeds.

        The initial planner creates exactly one final task. Its readiness query
        uses the run-scoped unfinished-task index instead of storing one dependency
        edge per document. Consequently ordinary preparation and extraction
        completions never lock one shared barrier row. Restricting the serialized
        run transition to finalization keeps millions of independent queue
        completions concurrent while retaining a defensive unfinished-task check
        for malformed plans created through the low-level store API.
        """

        connection.execute(
            "SELECT id FROM flakegraph_run WHERE id = %s FOR UPDATE",
            (run_id,),
        ).fetchone()
        unfinished = connection.execute(
            """
            SELECT 1 FROM flakegraph_task
            WHERE run_id = %s AND status <> %s
            LIMIT 1
            """,
            (run_id, TaskStatus.SUCCEEDED.value),
        ).fetchone()
        pending_publication = connection.execute(
            """
            SELECT 1 FROM flakegraph_publication
            WHERE run_id = %s AND status <> %s
            LIMIT 1
            """,
            (run_id, PublicationStatus.SUCCEEDED.value),
        ).fetchone()
        if unfinished is None and pending_publication is None:
            connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status = %s
                """,
                (RunStatus.SUCCEEDED.value, run_id, RunStatus.RUNNING.value),
            )

    def _publish_graph_version(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        run_id: str,
        output_rows: list[dict[str, Any]],
    ) -> None:
        """Atomically activate the final task's immutable graph artifact.

        Metadata-only final tasks may emit no graph result. A real finalizer emits
        exactly one; conflicting outputs are rejected before task completion can
        commit, so readers retain the previous complete graph version.
        """

        graph_outputs = [
            row for row in output_rows if str(row["kind"]) == ArtifactKind.GRAPH_RESULT.value
        ]
        if not graph_outputs:
            return
        if len(graph_outputs) != 1:
            raise ValueError("finalize_graph must publish exactly one graph result")
        artifact = graph_outputs[0]
        run = connection.execute(
            "SELECT graph_id FROM flakegraph_run WHERE id = %s",
            (run_id,),
        ).fetchone()
        if run is None:
            raise KeyError(f"unknown distributed run: {run_id}")
        artifact_id = str(artifact["id"])
        graph_id = str(run["graph_id"])
        connection.execute(
            """
            INSERT INTO flakegraph_graph_version (
                id, graph_id, run_id, artifact_id, media_type
            ) VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE
            SET artifact_id = EXCLUDED.artifact_id,
                media_type = EXCLUDED.media_type
            """,
            (artifact_id, graph_id, run_id, artifact_id, str(artifact["media_type"])),
        )
        connection.execute(
            """
            INSERT INTO flakegraph_graph_head (graph_id, version_id, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (graph_id) DO UPDATE
            SET version_id = EXCLUDED.version_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (graph_id, artifact_id),
        )

    def _insert_follow_up_tasks(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        run_id: str,
        parent_task_id: str,
        tasks: list[TaskDefinition],
        barrier_task_id: str | None,
    ) -> None:
        """Insert a validated dynamic subgraph before its parent can succeed.

        Context extraction publishes a small two-level DAG: independent window
        tasks followed by one document compaction task. The complete subgraph is
        inserted in the parent's completion transaction. Run-wide final readiness
        therefore sees either the running parent or all of its queued children and
        can never observe a transiently complete run.
        """

        if barrier_task_id is None:
            raise ValueError("dynamic follow-up tasks require a barrier_task_id")
        parent_stage_row = connection.execute(
            "SELECT stage FROM flakegraph_task WHERE id = %s",
            (parent_task_id,),
        ).fetchone()
        if parent_stage_row is None:
            raise ValueError("dynamic follow-up parent task does not exist")
        parent_stage = TaskStage(str(parent_stage_row["stage"]))
        _validate_dynamic_follow_ups(
            run_id,
            parent_task_id,
            parent_stage,
            tasks,
        )
        barrier = connection.execute(
            """
            SELECT id, run_id, stage, status
            FROM flakegraph_task
            WHERE id = %s
            """,
            (barrier_task_id,),
        ).fetchone()
        if (
            barrier is None
            or str(barrier["run_id"]) != run_id
            or str(barrier["stage"]) != TaskStage.FINALIZE_GRAPH.value
            or str(barrier["status"]) != TaskStatus.QUEUED.value
        ):
            raise ValueError("dynamic follow-ups require a queued final barrier in the same run")
        # A long document may create hundreds of windows. COPY keeps task and edge
        # creation to two protocol exchanges. The finalizer observes all unfinished
        # tasks in the run, so dynamic work needs no explicit edge to a shared final
        # row and document completions never serialize on a global lock.
        with connection.cursor().copy(
            """
            COPY flakegraph_task (
                id, run_id, stage, scope_id, payload_json, status,
                priority, max_attempts, remaining_dependencies
            ) FROM STDIN
            """
        ) as task_copy:
            for follow_up in tasks:
                task_copy.write_row(
                    (
                        follow_up.id,
                        follow_up.run_id,
                        follow_up.stage.value,
                        follow_up.scope_id,
                        Jsonb(follow_up.payload),
                        TaskStatus.QUEUED.value,
                        follow_up.priority,
                        follow_up.max_attempts,
                        len(follow_up.dependency_ids),
                    )
                )
        with connection.cursor().copy(
            """
            COPY flakegraph_task_dependency (task_id, depends_on_task_id)
            FROM STDIN
            """
        ) as dependency_copy:
            for follow_up in tasks:
                for dependency_id in follow_up.dependency_ids:
                    dependency_copy.write_row((follow_up.id, dependency_id))

    def fail_task(
        self,
        task_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
    ) -> None:
        """Retry an owned task within its budget or fail the run terminally."""

        if retry_delay < timedelta(0):
            raise ValueError("retry delay must not be negative")
        with self._connection() as connection:
            task = self._owned_task(connection, task_id, worker_id)
            exhausted = int(task["attempts"]) >= int(task["max_attempts"])
            next_status = TaskStatus.FAILED if exhausted else TaskStatus.QUEUED
            connection.execute(
                """
                UPDATE flakegraph_task
                SET status = %s,
                    available_at = CURRENT_TIMESTAMP + %s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    last_error_json = %s,
                    completed_at = CASE WHEN %s = 'failed' THEN CURRENT_TIMESTAMP ELSE NULL END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (next_status.value, retry_delay, Jsonb(error), next_status.value, task_id),
            )
            if exhausted:
                connection.execute(
                    """
                    UPDATE flakegraph_run
                    SET status = %s, error_json = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND status IN (%s, %s)
                    """,
                    (
                        RunStatus.FAILED.value,
                        Jsonb({"task_id": task_id, **error}),
                        task["run_id"],
                        RunStatus.QUEUED.value,
                        RunStatus.RUNNING.value,
                    ),
                )

    def cancel_run(self, run_id: str) -> None:
        """Cancel a non-terminal run and all unfinished tasks in one transaction."""

        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT status FROM flakegraph_run WHERE id = %s
                """,
                (run_id,),
            ).fetchone()
            if existing is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            if str(existing["status"]) in {
                RunStatus.SUCCEEDED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
            }:
                return
            # Workers lock task/publication rows before the parent run row.
            # Cancellation follows the same hierarchy to avoid a task<->run
            # ABBA deadlock with concurrent completion.
            connection.execute(
                """
                UPDATE flakegraph_task
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND status IN (%s, %s)
                """,
                (
                    TaskStatus.CANCELLED.value,
                    run_id,
                    TaskStatus.QUEUED.value,
                    TaskStatus.RUNNING.value,
                ),
            )
            connection.execute(
                """
                UPDATE flakegraph_publication
                SET status = %s, lease_owner = NULL, lease_expires_at = NULL,
                    completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND status IN (%s, %s)
                """,
                (
                    PublicationStatus.CANCELLED.value,
                    run_id,
                    PublicationStatus.QUEUED.value,
                    PublicationStatus.RUNNING.value,
                ),
            )
            connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status NOT IN (%s, %s, %s)
                """,
                (
                    RunStatus.CANCELLED.value,
                    run_id,
                    RunStatus.SUCCEEDED.value,
                    RunStatus.FAILED.value,
                    RunStatus.CANCELLED.value,
                ),
            )

    def retry_run(self, run_id: str) -> None:
        """Reset failed tasks and reactivate a failed run after operator intervention.

        Successful prerequisites and immutable outputs are preserved. Attempt counts
        reset only for terminally failed tasks, creating a fresh bounded retry budget
        after the underlying configuration or provider problem has been corrected.
        """

        with self._connection() as connection:
            run = connection.execute(
                "SELECT status FROM flakegraph_run WHERE id = %s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            if str(run["status"]) != RunStatus.FAILED.value:
                raise ValueError("only a failed distributed run can be retried")
            failed = connection.execute(
                """
                UPDATE flakegraph_task
                SET status = %s,
                    attempts = 0,
                    available_at = CURRENT_TIMESTAMP,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    last_error_json = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    progress_json = NULL,
                    progress_updated_at = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = %s AND status = %s
                """,
                (TaskStatus.QUEUED.value, run_id, TaskStatus.FAILED.value),
            )
            publication_only_retry = failed.rowcount == 0
            if publication_only_retry:
                publications = connection.execute(
                    """
                    UPDATE flakegraph_publication
                    SET status = %s, attempts = 0,
                        available_at = CURRENT_TIMESTAMP,
                        lease_owner = NULL, lease_expires_at = NULL,
                        last_error_json = NULL, completed_at = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE run_id = %s AND status = %s
                    """,
                    (
                        PublicationStatus.QUEUED.value,
                        run_id,
                        PublicationStatus.FAILED.value,
                    ),
                )
                if publications.rowcount == 0:
                    raise ValueError(
                        "failed run does not contain a terminally failed task or publication"
                    )
            connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s, error_json = NULL, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (
                    RunStatus.RUNNING.value if publication_only_retry else RunStatus.QUEUED.value,
                    run_id,
                ),
            )

    def get_run(self, run_id: str) -> RunSnapshot:
        """Read one run and its ordered tasks from a repeatable transaction snapshot."""

        with self._connection() as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            run = connection.execute(
                "SELECT * FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            rows = connection.execute(
                """
                SELECT task.*,
                       COALESCE(
                           (
                               SELECT jsonb_agg(output.artifact_id ORDER BY output.position)
                               FROM flakegraph_task_output AS output
                               WHERE output.task_id = task.id
                           ),
                           '[]'::jsonb
                       ) AS output_artifact_ids,
                       COALESCE(
                           (
                               SELECT jsonb_agg(
                                   dependency.depends_on_task_id
                                   ORDER BY dependency.depends_on_task_id
                               )
                               FROM flakegraph_task_dependency AS dependency
                               WHERE dependency.task_id = task.id
                           ),
                           '[]'::jsonb
                       ) AS dependency_ids
                FROM flakegraph_task AS task
                WHERE task.run_id = %s
                ORDER BY task.created_at, task.id
                """,
                (run_id,),
            ).fetchall()
            tasks = [
                TaskSnapshot(
                    task=_task_definition(row, row["dependency_ids"]),
                    status=TaskStatus(str(row["status"])),
                    attempts=int(row["attempts"]),
                    lease_owner=cast(str | None, row["lease_owner"]),
                    lease_expires_at=row["lease_expires_at"],
                    output_artifact_ids=list(row["output_artifact_ids"]),
                    last_error=cast(dict[str, Any] | None, row["last_error_json"]),
                    progress=(
                        TaskProgress.model_validate(row["progress_json"])
                        if row["progress_json"]
                        else None
                    ),
                )
                for row in rows
            ]
            definition = RunDefinition(
                id=str(run["id"]),
                graph_id=str(run["graph_id"]),
                config=cast(dict[str, Any], run["config_json"]),
                config_digest=str(run["config_digest"]),
                status=RunStatus(str(run["status"])),
            )
            effective_updated_at = max(
                [run["updated_at"], *(row["updated_at"] for row in rows)],
            )
            return RunSnapshot(
                run=definition,
                tasks=tasks,
                created_at=run["created_at"],
                updated_at=effective_updated_at,
            )

    def get_run_summary(self, run_id: str) -> RunSummary:
        """Read constant-size stage progress for operator-facing commands.

        The aggregate executes entirely in PostgreSQL and returns at most one row
        per stage/status pair, regardless of whether the run contains ten tasks or
        several million extraction windows.
        """

        with self._connection() as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            run = connection.execute(
                "SELECT * FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            rows = connection.execute(
                """
                SELECT stage, status, COUNT(*) AS task_count,
                       MIN(started_at) AS started_at,
                       MAX(completed_at) AS completed_at,
                       MAX(updated_at) AS updated_at,
                       (jsonb_agg(progress_json ORDER BY progress_updated_at DESC)
                           FILTER (WHERE progress_json IS NOT NULL))->0 AS progress_json
                FROM flakegraph_task
                WHERE run_id = %s
                GROUP BY stage, status
                ORDER BY stage, status
                """,
                (run_id,),
            ).fetchall()
            task_counts = [
                TaskCount(
                    stage=TaskStage(str(row["stage"])),
                    status=TaskStatus(str(row["status"])),
                    count=int(row["task_count"]),
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    progress=(
                        TaskProgress.model_validate(row["progress_json"])
                        if row["progress_json"]
                        else None
                    ),
                )
                for row in rows
            ]
            # Heartbeats intentionally update task rows rather than one shared run
            # row, which avoids a write hotspot across large worker fleets. The
            # status aggregate already visits each stage/status bucket, so its
            # maximum supplies truthful run liveness without another task scan.
            latest_update = max(
                [
                    run["updated_at"],
                    *(row["updated_at"] for row in rows if row["updated_at"] is not None),
                ]
            )
            return RunSummary(
                run=RunDefinition(
                    id=str(run["id"]),
                    graph_id=str(run["graph_id"]),
                    config=cast(dict[str, Any], run["config_json"]),
                    config_digest=str(run["config_digest"]),
                    status=RunStatus(str(run["status"])),
                ),
                task_counts=task_counts,
                total_tasks=sum(item.count for item in task_counts),
                created_at=run["created_at"],
                updated_at=latest_update,
                error=cast(dict[str, Any] | None, run["error_json"]),
            )

    def list_runs(self, limit: int = 100) -> list[RunOverview]:
        """List recent runs with bounded grouped task counts in two SQL queries."""

        if limit <= 0 or limit > _MAX_RUN_LIST_LIMIT:
            raise ValueError("distributed run list limit must be between 1 and 500")
        with self._connection() as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            runs = connection.execute(
                """
                SELECT id, graph_id, status, created_at, updated_at
                FROM flakegraph_run
                ORDER BY updated_at DESC, id
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
            if not runs:
                return []
            run_ids = [str(run["id"]) for run in runs]
            count_rows = connection.execute(
                """
                SELECT run_id, stage, status, COUNT(*) AS task_count,
                       MIN(started_at) AS started_at,
                       MAX(completed_at) AS completed_at,
                       MAX(updated_at) AS updated_at,
                       (jsonb_agg(progress_json ORDER BY progress_updated_at DESC)
                           FILTER (WHERE progress_json IS NOT NULL))->0 AS progress_json
                FROM flakegraph_task
                WHERE run_id = ANY(%s)
                GROUP BY run_id, stage, status
                ORDER BY run_id, stage, status
                """,
                (run_ids,),
            ).fetchall()
            counts_by_run: dict[str, list[TaskCount]] = {run_id: [] for run_id in run_ids}
            updated_by_run = {str(run["id"]): run["updated_at"] for run in runs}
            for row in count_rows:
                row_run_id = str(row["run_id"])
                counts_by_run[row_run_id].append(
                    TaskCount(
                        stage=TaskStage(str(row["stage"])),
                        status=TaskStatus(str(row["status"])),
                        count=int(row["task_count"]),
                        started_at=row["started_at"],
                        completed_at=row["completed_at"],
                        progress=(
                            TaskProgress.model_validate(row["progress_json"])
                            if row["progress_json"]
                            else None
                        ),
                    )
                )
                if row["updated_at"] is not None:
                    updated_by_run[row_run_id] = max(
                        updated_by_run[row_run_id],
                        row["updated_at"],
                    )
            return [
                RunOverview(
                    id=str(run["id"]),
                    graph_id=str(run["graph_id"]),
                    status=RunStatus(str(run["status"])),
                    task_counts=counts_by_run[str(run["id"])],
                    total_tasks=sum(item.count for item in counts_by_run[str(run["id"])]),
                    created_at=run["created_at"],
                    updated_at=updated_by_run[str(run["id"])],
                )
                for run in runs
            ]

    def put(
        self,
        run_id: str,
        kind: ArtifactKind,
        payload: bytes,
        media_type: str,
        metadata: dict[str, Any] | None = None,
        identity_key: str | None = None,
    ) -> ArtifactRef:
        """Store immutable bytes inline or in the configured scalable blob plane."""

        if len(payload) > self.max_artifact_bytes:
            raise ValueError(
                f"artifact payload exceeds configured {self.max_artifact_bytes}-byte limit"
            )
        checksum = sha256_hex(payload)
        artifact_id = (
            stable_id("artifact", run_id, kind.value, checksum, identity_key)
            if identity_key is not None
            else stable_id("artifact", run_id, kind.value, checksum)
        )
        # Validate once per store process before transferring bytes. Planner and
        # worker paths normally populate this cache from their successful create
        # or claim transaction, while direct artifact callers still receive the
        # same unknown-run error before an external object can be orphaned.
        self._ensure_known_run_id(run_id)
        storage_uri: str | None = None
        database_payload: bytes | None
        compression: str
        if self.blob_store is None:
            database_payload = zlib.compress(payload, self.compression_level)
            compression = "zlib"
        else:
            suffix = _artifact_suffix(media_type)
            storage_uri = self.blob_store.put(
                f"{run_id}/{kind.value}/{artifact_id}{suffix}",
                payload,
                media_type,
            )
            database_payload = None
            compression = "none"
        artifact_metadata = metadata or {}
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO flakegraph_artifact (
                    id, run_id, kind, media_type, checksum, size_bytes,
                    compression, payload, storage_uri, metadata_json
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    artifact_id,
                    run_id,
                    kind.value,
                    media_type,
                    checksum,
                    len(payload),
                    compression,
                    database_payload,
                    storage_uri,
                    Jsonb(artifact_metadata),
                ),
            )
            row = connection.execute(
                "SELECT * FROM flakegraph_artifact WHERE id = %s",
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"failed to persist artifact {artifact_id}")
            return _artifact_ref(row)

    def _ensure_known_run_id(self, run_id: str) -> None:
        """Validate an uncached artifact owner once without racing staging threads."""

        with self._known_run_ids_lock:
            if run_id in self._known_run_ids:
                return
        with self._connection() as connection:
            run = connection.execute(
                "SELECT id FROM flakegraph_run WHERE id = %s",
                (run_id,),
            ).fetchone()
        if run is None:
            raise KeyError(f"unknown distributed run: {run_id}")
        with self._known_run_ids_lock:
            self._known_run_ids.add(run_id)

    def _remember_run_id(self, run_id: str) -> None:
        """Cache a run identity already proven by a successful transaction."""

        with self._known_run_ids_lock:
            self._known_run_ids.add(run_id)

    def get(self, artifact_id: str) -> StoredArtifact:
        """Load, decompress, size-check, and checksum-verify immutable artifact bytes."""

        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM flakegraph_artifact WHERE id = %s",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown distributed artifact: {artifact_id}")
        return self._load_artifact_row(row)

    def get_many(self, artifact_ids: list[str]) -> list[StoredArtifact]:
        """Resolve metadata once and overlap independent object-store reads.

        Caller order is preserved because window recombination is deterministic.
        A single ``ANY`` lookup replaces one PostgreSQL transaction per artifact;
        bounded parallel reads then hide object-store latency without allowing a
        large document to create an unbounded number of in-flight payloads.
        """

        if not artifact_ids:
            return []
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("batch artifact ids must be unique")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM flakegraph_artifact WHERE id = ANY(%s)",
                (artifact_ids,),
            ).fetchall()
        rows_by_id = {str(row["id"]): row for row in rows}
        missing = [artifact_id for artifact_id in artifact_ids if artifact_id not in rows_by_id]
        if missing:
            raise KeyError(f"unknown distributed artifact: {missing[0]}")
        ordered_rows = [rows_by_id[artifact_id] for artifact_id in artifact_ids]
        workers = min(_ARTIFACT_READ_PARALLELISM, len(ordered_rows))
        if workers == 1:
            return [self._load_artifact_row(ordered_rows[0])]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(
                executor.map(
                    self._load_artifact_row,
                    ordered_rows,
                    buffersize=workers,
                )
            )

    def get_run_artifacts(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
    ) -> list[StoredArtifact]:
        """Discover selected stage outputs without encoding them as scheduling edges."""

        if not kinds:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id
                FROM flakegraph_artifact
                WHERE run_id = %s AND kind = ANY(%s)
                ORDER BY created_at, id
                """,
                (run_id, [kind.value for kind in sorted(kinds, key=str)]),
            ).fetchall()
        return self.get_many([str(row["id"]) for row in rows])

    def _load_artifact_row(self, row: dict[str, Any]) -> StoredArtifact:
        """Decode and verify one metadata row after its database lookup."""

        artifact_id = str(row["id"])
        compression = str(row["compression"])
        storage_uri = cast(str | None, row.get("storage_uri"))
        if compression == "zlib" and row["payload"] is not None:
            try:
                payload = zlib.decompress(bytes(row["payload"]))
            except zlib.error as exc:
                raise ValueError(f"artifact {artifact_id} has corrupt compressed data") from exc
        elif compression == "none" and storage_uri and self.blob_store is not None:
            payload = self.blob_store.get(storage_uri)
        else:
            raise ValueError(
                f"artifact {artifact_id} storage is unavailable or inconsistent "
                f"(compression={compression}, storage_uri={storage_uri!r})"
            )
        if len(payload) != int(row["size_bytes"]):
            raise ValueError(f"artifact {artifact_id} size does not match its metadata")
        if sha256_hex(payload) != str(row["checksum"]):
            raise ValueError(f"artifact {artifact_id} checksum verification failed")
        return StoredArtifact(ref=_artifact_ref(row), payload=payload)

    def delete_run_artifacts(self, run_id: str) -> int:
        """Delete an inactive terminal run's artifacts in retry-safe batches.

        Every restricting reference to the run's artifacts is removed first, so a
        surviving reference surfaces as a failed ``DELETE`` rather than as objects
        that were already destroyed. Each batch deletes its metadata rows and only
        then its payloads inside one transaction: an object-store or process
        failure rolls the rows back, leaving metadata discoverable for an
        idempotent retry. Batches are bounded so a large corpus never creates one
        multi-million-row transaction.
        """

        with self._connection() as connection:
            run = connection.execute(
                "SELECT status FROM flakegraph_run WHERE id = %s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"unknown distributed run: {run_id}")
            run_status = str(run["status"])
            if run_status not in _TERMINAL_RUN_STATUSES:
                raise ValueError("artifacts may only be deleted for a terminal run")
            if run_status == RunStatus.FAILED.value:
                raise ValueError(
                    "artifacts for a failed run cannot be deleted because the run remains retryable"
                )
            active = connection.execute(
                """
                SELECT 1
                FROM flakegraph_graph_head AS head
                JOIN flakegraph_graph_version AS version ON version.id = head.version_id
                WHERE version.run_id = %s
                """,
                (run_id,),
            ).fetchone()
            if active is not None:
                raise ValueError("artifacts for the active graph version cannot be deleted")
            # Both tables reference flakegraph_artifact with ON DELETE RESTRICT.
            # Superseded version and publication records describe artifacts that
            # this call discards, so they are cleared before the payload rows.
            connection.execute(
                "DELETE FROM flakegraph_graph_version WHERE run_id = %s",
                (run_id,),
            )
            connection.execute(
                "DELETE FROM flakegraph_publication WHERE run_id = %s",
                (run_id,),
            )
        deleted_count = 0
        while True:
            with self._connection() as connection:
                rows = connection.execute(
                    """
                    SELECT id, storage_uri
                    FROM flakegraph_artifact
                    WHERE run_id = %s
                    ORDER BY id
                    LIMIT %s
                    """,
                    (run_id, _ARTIFACT_DELETE_BATCH_SIZE),
                ).fetchall()
            if not rows:
                return deleted_count
            storage_uris = [
                str(row["storage_uri"]) for row in rows if row["storage_uri"] is not None
            ]
            artifact_ids = [str(row["id"]) for row in rows]
            with self._connection() as connection:
                deleted = connection.execute(
                    """
                    DELETE FROM flakegraph_artifact
                    WHERE run_id = %s AND id = ANY(%s)
                    """,
                    (run_id, artifact_ids),
                )
                batch_count = deleted.rowcount
                self._delete_blob_batch(storage_uris)
            deleted_count += batch_count

    def _delete_blob_batch(self, storage_uris: list[str]) -> None:
        """Use bulk deletion when available and preserve adapter compatibility."""

        if self.blob_store is None or not storage_uris:
            return
        if isinstance(self.blob_store, BatchBlobDeleter):
            self.blob_store.delete_many(storage_uris)
            return
        for storage_uri in storage_uris:
            self.blob_store.delete(storage_uri)

    def _owned_task(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        task_id: str,
        worker_id: str,
    ) -> dict[str, Any]:
        """Lock and return a running task only when the caller owns its lease."""

        row = connection.execute(
            """
            SELECT * FROM flakegraph_task
            WHERE id = %s AND status = %s AND lease_owner = %s
            FOR UPDATE
            """,
            (task_id, TaskStatus.RUNNING.value, worker_id),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"worker {worker_id} no longer owns task {task_id}")
        return row

    def _fail_abandoned_exhausted_tasks(
        self,
        connection: psycopg.Connection[dict[str, Any]],
    ) -> None:
        """Fail exhausted task leases only after a heartbeat-aware recovery window."""

        failed_runs = connection.execute(
            """
            WITH exhausted AS (
                UPDATE flakegraph_task
                SET status = %s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    last_error_json = %s,
                    completed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status = %s
                  AND attempts >= max_attempts
                  AND GREATEST(lease_expires_at, updated_at) < (
                      CURRENT_TIMESTAMP - (%s * INTERVAL '1 second')
                  )
                RETURNING run_id, id
            )
            SELECT run_id, MIN(id) AS representative_task_id
            FROM exhausted
            GROUP BY run_id
            """,
            (
                TaskStatus.FAILED.value,
                Jsonb(
                    {
                        "type": "LeaseExpired",
                        "message": "worker lease expired after retry budget was exhausted",
                    }
                ),
                TaskStatus.RUNNING.value,
                _EXHAUSTED_TASK_RECOVERY_GRACE_SECONDS,
            ),
        ).fetchall()
        for failed in failed_runs:
            connection.execute(
                """
                UPDATE flakegraph_run
                SET status = %s,
                    error_json = jsonb_build_object(
                        'type', 'LeaseExpired',
                        'task_id', %s::text,
                        'message', 'worker lease expired after retry budget was exhausted'
                    ),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND status IN (%s, %s)
                """,
                (
                    RunStatus.FAILED.value,
                    failed["representative_task_id"],
                    failed["run_id"],
                    RunStatus.QUEUED.value,
                    RunStatus.RUNNING.value,
                ),
            )

    def _dependency_outputs(
        self,
        connection: psycopg.Connection[dict[str, Any]],
        task_id: str,
    ) -> dict[str, list[str]]:
        """Return ordered artifact outputs for every direct prerequisite task."""

        rows = connection.execute(
            """
            SELECT dependency.depends_on_task_id, output.artifact_id
            FROM flakegraph_task_dependency AS dependency
            LEFT JOIN flakegraph_task_output AS output
              ON output.task_id = dependency.depends_on_task_id
            WHERE dependency.task_id = %s
            ORDER BY dependency.depends_on_task_id, output.position
            """,
            (task_id,),
        ).fetchall()
        outputs: dict[str, list[str]] = {}
        for row in rows:
            dependency_id = str(row["depends_on_task_id"])
            outputs.setdefault(dependency_id, [])
            if row["artifact_id"] is not None:
                outputs[dependency_id].append(str(row["artifact_id"]))
        return outputs


def _task_definition(
    row: dict[str, Any],
    dependency_ids: Any,
) -> TaskDefinition:
    """Convert a database task row into its provider-neutral domain contract."""

    return TaskDefinition(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        stage=TaskStage(str(row["stage"])),
        scope_id=str(row["scope_id"]),
        payload=cast(dict[str, Any], row["payload_json"]),
        dependency_ids=sorted(str(value) for value in dependency_ids),
        priority=int(row["priority"]),
        max_attempts=int(row["max_attempts"]),
    )


def _artifact_ref(row: dict[str, Any]) -> ArtifactRef:
    """Convert one artifact row into immutable public metadata."""

    return ArtifactRef(
        id=str(row["id"]),
        run_id=str(row["run_id"]),
        kind=ArtifactKind(str(row["kind"])),
        media_type=str(row["media_type"]),
        checksum=str(row["checksum"]),
        size_bytes=int(row["size_bytes"]),
        storage_uri=cast(str | None, row.get("storage_uri")),
        metadata=cast(dict[str, Any], row["metadata_json"]),
    )


def _artifact_suffix(media_type: str) -> str:
    """Choose an inspectable object suffix without making it part of identity."""

    if media_type == "application/json":
        return ".json"
    if media_type in {"application/vnd.apache.parquet", "application/x-parquet"}:
        return ".parquet"
    return ".bin"


def _validate_initial_task(
    task: TaskDefinition,
    run_id: str,
    final_seen: bool,
) -> tuple[int, int]:
    """Validate one streamed plan row and return preparation/final increments."""

    if task.run_id != run_id:
        raise ValueError("all tasks must belong to the selected run")
    if task.dependency_ids:
        raise ValueError("initial document tasks cannot have dependencies")
    if final_seen:
        raise ValueError("the final task must be the last initial-plan record")
    if task.stage is TaskStage.PREPARE_DOCUMENT:
        return 1, 0
    if task.stage is TaskStage.FINALIZE_GRAPH:
        return 0, 1
    raise ValueError("initial plan may contain only preparation and final tasks")


def _validate_dynamic_follow_ups(
    run_id: str,
    parent_task_id: str,
    parent_stage: TaskStage,
    tasks: list[TaskDefinition],
) -> set[str]:
    """Validate the bounded runtime DAG and return its terminal task IDs.

    Preparation adds a context child. Context extraction adds entity windows plus
    their inventory barrier. Inventory compaction adds relation windows plus the
    final document barrier. Restricting runtime fan-out to these explicit shapes
    keeps cycle safety obvious while allowing both provider-heavy phases to use the
    shared work-stealing queue.
    """

    task_ids = {task.id for task in tasks}
    if len(task_ids) != len(tasks):
        raise ValueError("dynamic follow-up task ids must be unique")
    if any(task.run_id != run_id for task in tasks):
        raise ValueError("dynamic follow-up tasks must belong to their parent run")
    allowed_children = {
        TaskStage.PREPARE_DOCUMENT: {TaskStage.EXTRACT_DOCUMENT_CONTEXT},
        TaskStage.EXTRACT_DOCUMENT_CONTEXT: {
            TaskStage.EXTRACT_ENTITY_WINDOW,
            TaskStage.COMPACT_ENTITY_INVENTORY,
        },
        TaskStage.COMPACT_ENTITY_INVENTORY: {
            TaskStage.EXTRACT_RELATION_WINDOW,
            TaskStage.COMPACT_DOCUMENT,
        },
    }
    if any(task.stage not in allowed_children.get(parent_stage, set()) for task in tasks):
        raise ValueError(f"{parent_stage.value} cannot create the requested follow-up stage")
    allowed_dependency_ids = {parent_task_id, *task_ids}
    if any(not task.dependency_ids for task in tasks):
        raise ValueError("each dynamic follow-up requires at least one dependency")
    if any(set(task.dependency_ids) - allowed_dependency_ids for task in tasks):
        raise ValueError("dynamic follow-up dependencies must stay inside their subgraph")
    if any(task.id in task.dependency_ids for task in tasks):
        raise ValueError("dynamic follow-up tasks cannot depend on themselves")

    if parent_stage == TaskStage.EXTRACT_DOCUMENT_CONTEXT:
        _validate_entity_fan_out(parent_task_id, tasks)
    elif parent_stage == TaskStage.COMPACT_ENTITY_INVENTORY:
        _validate_relation_fan_out(parent_task_id, tasks)
    elif parent_stage == TaskStage.PREPARE_DOCUMENT and any(
        task.dependency_ids != [parent_task_id] for task in tasks
    ):
        raise ValueError("each preparation follow-up must depend directly on its parent")

    depended_on = {
        dependency_id
        for follow_up in tasks
        for dependency_id in follow_up.dependency_ids
        if dependency_id in task_ids
    }
    return task_ids - depended_on


def _validate_entity_fan_out(parent_task_id: str, tasks: list[TaskDefinition]) -> None:
    """Require entity windows followed by exactly one complete inventory barrier."""

    window_ids = {task.id for task in tasks if task.stage == TaskStage.EXTRACT_ENTITY_WINDOW}
    compact_tasks = [task for task in tasks if task.stage == TaskStage.COMPACT_ENTITY_INVENTORY]
    if len(compact_tasks) != 1:
        raise ValueError("document context must create exactly one entity inventory task")
    if any(
        task.dependency_ids != [parent_task_id]
        for task in tasks
        if task.stage == TaskStage.EXTRACT_ENTITY_WINDOW
    ):
        raise ValueError("each entity window must depend directly on document context")
    if set(compact_tasks[0].dependency_ids) != {parent_task_id, *window_ids}:
        raise ValueError("entity inventory must depend on context and every entity window")


def _validate_relation_fan_out(parent_task_id: str, tasks: list[TaskDefinition]) -> None:
    """Require relation windows followed by exactly one complete document barrier."""

    window_ids = {task.id for task in tasks if task.stage == TaskStage.EXTRACT_RELATION_WINDOW}
    compact_tasks = [task for task in tasks if task.stage == TaskStage.COMPACT_DOCUMENT]
    if len(compact_tasks) != 1:
        raise ValueError("entity inventory must create exactly one document compaction task")
    if any(
        task.dependency_ids != [parent_task_id]
        for task in tasks
        if task.stage == TaskStage.EXTRACT_RELATION_WINDOW
    ):
        raise ValueError("each relation window must depend directly on entity inventory")
    if set(compact_tasks[0].dependency_ids) != {parent_task_id, *window_ids}:
        raise ValueError("document compaction must depend on inventory and every relation window")


def _validate_duration(value: timedelta, label: str) -> None:
    """Require positive finite durations before generating lease timestamps."""

    if value <= timedelta(0):
        raise ValueError(f"{label} must be positive")


def _validate_acyclic(tasks: list[TaskDefinition]) -> None:
    """Reject cyclic plans in linear time using Kahn's topological algorithm.

    A corpus starts with one preparation task per document and one final barrier,
    so production plans can contain hundreds of thousands of vertices and edges.
    Maintaining reverse adjacency avoids scanning the complete plan after every
    visited task; memory and runtime are both proportional to ``V + E``.
    """

    remaining_dependencies = {task.id: len(task.dependency_ids) for task in tasks}
    dependents: dict[str, list[str]] = {task.id: [] for task in tasks}
    for task in tasks:
        for dependency_id in task.dependency_ids:
            dependents[dependency_id].append(task.id)
    ready = deque(
        task_id
        for task_id, dependency_count in remaining_dependencies.items()
        if not dependency_count
    )
    visited = 0
    while ready:
        task_id = ready.popleft()
        visited += 1
        for dependent_id in dependents[task_id]:
            remaining_dependencies[dependent_id] -= 1
            if remaining_dependencies[dependent_id] == 0:
                ready.append(dependent_id)
    if visited != len(tasks):
        raise ValueError("distributed task graph must be acyclic")


def _is_initial_document_plan(
    tasks: list[TaskDefinition],
    *,
    task_ids: set[str] | None = None,
) -> bool:
    """Recognize the planner's trivially acyclic initial document plan.

    This is deliberately strict: every non-final task must be an independent
    document preparation and the sole final task must have no explicit dependencies.
    Its run-wide readiness predicate supplies the barrier without a 250,000-edge
    star. Any extension or malformed shape falls back to the general linear-time
    validator rather than receiving a special-case exemption.
    """

    known_task_ids = task_ids if task_ids is not None else {task.id for task in tasks}
    final_task: TaskDefinition | None = None
    preparation_count = 0
    for task in tasks:
        if task.stage == TaskStage.PREPARE_DOCUMENT and not task.dependency_ids:
            preparation_count += 1
        elif task.stage == TaskStage.FINALIZE_GRAPH and final_task is None:
            final_task = task
        else:
            return False
    if final_task is None or preparation_count + 1 != len(tasks):
        return False
    return not final_task.dependency_ids and final_task.id in known_task_ids


_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS flakegraph_run (
        id TEXT PRIMARY KEY,
        graph_id TEXT NOT NULL,
        config_json JSONB NOT NULL,
        config_digest TEXT NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('planning', 'queued', 'running', 'succeeded', 'failed', 'cancelled')
        ),
        error_json JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_task (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES flakegraph_run(id) ON DELETE CASCADE,
        stage TEXT NOT NULL CHECK (
            stage IN (
                'prepare_document',
                'extract_document_context',
                'extract_entity_window',
                'compact_entity_inventory',
                'extract_relation_window',
                'compact_document',
                'finalize_graph'
            )
        ),
        scope_id TEXT NOT NULL,
        payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        status TEXT NOT NULL CHECK (
            status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')
        ),
        priority INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        remaining_dependencies INTEGER NOT NULL DEFAULT 0
            CHECK (remaining_dependencies >= 0),
        available_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        lease_owner TEXT,
        lease_expires_at TIMESTAMPTZ,
        last_error_json JSONB,
        progress_json JSONB,
        progress_updated_at TIMESTAMPTZ,
        started_at TIMESTAMPTZ,
        completed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (run_id, stage, scope_id),
        CHECK (
            (status = 'running' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
            OR status <> 'running'
        )
    )
    """,
    """
    ALTER TABLE flakegraph_task
    ADD COLUMN IF NOT EXISTS progress_json JSONB
    """,
    """
    ALTER TABLE flakegraph_task
    ADD COLUMN IF NOT EXISTS progress_updated_at TIMESTAMPTZ
    """,
    """
    ALTER TABLE flakegraph_task
    ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ
    """,
    """
    ALTER TABLE flakegraph_task
    ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_task_dependency (
        task_id TEXT NOT NULL REFERENCES flakegraph_task(id) ON DELETE CASCADE,
        depends_on_task_id TEXT NOT NULL REFERENCES flakegraph_task(id) ON DELETE CASCADE,
        PRIMARY KEY (task_id, depends_on_task_id),
        CHECK (task_id <> depends_on_task_id)
    )
    """,
    """
    WITH prerequisite_events AS (
        SELECT dependency.task_id, prerequisite.updated_at
        FROM flakegraph_task_dependency AS dependency
        JOIN flakegraph_task AS prerequisite
          ON prerequisite.id = dependency.depends_on_task_id
        WHERE prerequisite.status = 'succeeded'
        UNION ALL
        SELECT finalizer.id, prerequisite.updated_at
        FROM flakegraph_task AS finalizer
        JOIN flakegraph_task AS prerequisite
          ON prerequisite.run_id = finalizer.run_id
         AND prerequisite.id <> finalizer.id
        WHERE finalizer.stage = 'finalize_graph'
          AND prerequisite.status = 'succeeded'
    ), prerequisite_times AS (
        SELECT task_id, MAX(updated_at) AS ready_at
        FROM prerequisite_events
        GROUP BY task_id
    )
    UPDATE flakegraph_task AS task
    SET started_at = LEAST(
        task.updated_at,
        GREATEST(task.created_at, COALESCE(prerequisite.ready_at, task.created_at))
    )
    FROM prerequisite_times AS prerequisite
    WHERE prerequisite.task_id = task.id
      AND task.started_at IS NULL
      AND task.status <> 'queued'
    """,
    """
    UPDATE flakegraph_task
    SET started_at = LEAST(updated_at, created_at)
    WHERE started_at IS NULL AND status <> 'queued'
    """,
    """
    UPDATE flakegraph_task
    SET completed_at = updated_at
    WHERE completed_at IS NULL AND status IN ('succeeded', 'failed', 'cancelled')
    """,
    """
    ALTER TABLE flakegraph_task
    DROP CONSTRAINT IF EXISTS flakegraph_task_stage_check
    """,
    """
    UPDATE flakegraph_task
    SET stage = 'extract_entity_window'
    WHERE stage IN ('extract_document', 'extract_window')
    """,
    """
    ALTER TABLE flakegraph_task
    ADD CONSTRAINT flakegraph_task_stage_check CHECK (
        stage IN (
            'prepare_document',
            'extract_document_context',
            'extract_entity_window',
            'compact_entity_inventory',
            'extract_relation_window',
            'compact_document',
            'finalize_graph'
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_task_dependency (
        task_id TEXT NOT NULL REFERENCES flakegraph_task(id) ON DELETE CASCADE,
        depends_on_task_id TEXT NOT NULL REFERENCES flakegraph_task(id) ON DELETE CASCADE,
        PRIMARY KEY (task_id, depends_on_task_id),
        CHECK (task_id <> depends_on_task_id)
    )
    """,
    """
    ALTER TABLE flakegraph_task
    ADD COLUMN IF NOT EXISTS remaining_dependencies INTEGER
    """,
    """
    UPDATE flakegraph_task AS task
    SET remaining_dependencies = (
        SELECT COUNT(*)
        FROM flakegraph_task_dependency AS dependency
        JOIN flakegraph_task AS prerequisite
          ON prerequisite.id = dependency.depends_on_task_id
        WHERE dependency.task_id = task.id
          AND prerequisite.status <> 'succeeded'
    )
    WHERE task.remaining_dependencies IS NULL
    """,
    """
    ALTER TABLE flakegraph_task
    ALTER COLUMN remaining_dependencies SET DEFAULT 0
    """,
    """
    ALTER TABLE flakegraph_task
    ALTER COLUMN remaining_dependencies SET NOT NULL
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_artifact (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES flakegraph_run(id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (
            kind IN (
                'source_document',
                'prepared_document',
                'document_context',
                'extraction_window',
                'extracted_entity_window',
                'document_entity_inventory',
                'extracted_relation_window',
                'extracted_document',
                'graph_result'
            )
        ),
        media_type TEXT NOT NULL,
        checksum TEXT NOT NULL,
        size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
        compression TEXT NOT NULL,
        payload BYTEA,
        storage_uri TEXT,
        metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CHECK ((payload IS NOT NULL) <> (storage_uri IS NOT NULL))
    )
    """,
    """
    ALTER TABLE flakegraph_artifact
    DROP CONSTRAINT IF EXISTS flakegraph_artifact_run_id_kind_checksum_key
    """,
    """
    ALTER TABLE flakegraph_artifact
    DROP CONSTRAINT IF EXISTS flakegraph_artifact_kind_check
    """,
    """
    UPDATE flakegraph_artifact
    SET kind = 'extracted_entity_window'
    WHERE kind = 'extracted_window'
    """,
    """
    ALTER TABLE flakegraph_artifact
    ADD CONSTRAINT flakegraph_artifact_kind_check CHECK (
        kind IN (
            'source_document',
            'prepared_document',
            'document_context',
            'extraction_window',
            'extracted_entity_window',
            'document_entity_inventory',
            'extracted_relation_window',
            'extracted_document',
            'graph_result'
        )
    )
    """,
    """
    ALTER TABLE flakegraph_artifact
    ADD COLUMN IF NOT EXISTS storage_uri TEXT
    """,
    """
    ALTER TABLE flakegraph_artifact
    ALTER COLUMN payload DROP NOT NULL
    """,
    """
    ALTER TABLE flakegraph_artifact
    DROP CONSTRAINT IF EXISTS flakegraph_artifact_payload_location_check
    """,
    """
    ALTER TABLE flakegraph_artifact
    ADD CONSTRAINT flakegraph_artifact_payload_location_check CHECK (
        (payload IS NOT NULL) <> (storage_uri IS NOT NULL)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_task_output (
        task_id TEXT NOT NULL REFERENCES flakegraph_task(id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        artifact_id TEXT NOT NULL REFERENCES flakegraph_artifact(id) ON DELETE CASCADE,
        PRIMARY KEY (task_id, position),
        UNIQUE (task_id, artifact_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_graph_version (
        id TEXT PRIMARY KEY,
        graph_id TEXT NOT NULL,
        run_id TEXT NOT NULL UNIQUE REFERENCES flakegraph_run(id) ON DELETE RESTRICT,
        artifact_id TEXT NOT NULL UNIQUE REFERENCES flakegraph_artifact(id) ON DELETE RESTRICT,
        media_type TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_graph_head (
        graph_id TEXT PRIMARY KEY,
        version_id TEXT NOT NULL REFERENCES flakegraph_graph_version(id) ON DELETE RESTRICT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS flakegraph_publication (
        generation BIGSERIAL PRIMARY KEY,
        id TEXT NOT NULL UNIQUE,
        run_id TEXT NOT NULL REFERENCES flakegraph_run(id) ON DELETE CASCADE,
        task_id TEXT NOT NULL UNIQUE REFERENCES flakegraph_task(id) ON DELETE CASCADE,
        artifact_id TEXT NOT NULL REFERENCES flakegraph_artifact(id) ON DELETE RESTRICT,
        task_payload_json JSONB NOT NULL,
        status TEXT NOT NULL CHECK (
            status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')
        ),
        max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
        attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
        available_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        lease_owner TEXT,
        lease_expires_at TIMESTAMPTZ,
        last_error_json JSONB,
        completed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CHECK (
            (status = 'running' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)
            OR status <> 'running'
        )
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_publication_ready_idx
    ON flakegraph_publication (status, available_at, generation)
    """,
    """
    -- The OCR shim's admission queue. It lives here because the parsing plane
    -- must order waiting requests across every shim replica, which process
    -- memory cannot do, and because a restart should not drop work a client is
    -- still blocked on. Rows are short-lived: one exists only while a caller
    -- waits or is being parsed.
    --
    -- Priority follows the serving convention, where a LOWER value is served
    -- first. That is the opposite of flakegraph_task above, which orders by
    -- priority DESC. The two are different queues: this one shares its bands
    -- with the inference sidecar, not with the run planner.
    CREATE TABLE IF NOT EXISTS flakegraph_ocr_request (
        id TEXT PRIMARY KEY,
        priority INTEGER NOT NULL,
        consumer_class TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('waiting', 'dispatched')),
        shim_owner TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_ocr_request_waiting_idx
    ON flakegraph_ocr_request (priority, created_at, id)
    WHERE status = 'waiting'
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_ocr_request_stale_idx
    ON flakegraph_ocr_request (heartbeat_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_ready_claim_idx
    ON flakegraph_task (
        stage, status, remaining_dependencies, available_at,
        priority DESC, created_at
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_queued_claim_idx
    ON flakegraph_task (
        stage, remaining_dependencies, priority DESC, created_at, id
    )
    INCLUDE (run_id, available_at)
    WHERE status = 'queued'
    """,
    """
    DROP INDEX IF EXISTS flakegraph_task_run_idx
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_run_summary_idx
    ON flakegraph_task (run_id, status, stage)
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_dependency_prerequisite_idx
    ON flakegraph_task_dependency (depends_on_task_id, task_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_unfinished_run_idx
    ON flakegraph_task (run_id)
    WHERE status <> 'succeeded'
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_task_expired_lease_idx
    ON flakegraph_task (lease_expires_at)
    WHERE status = 'running'
    """,
    """
    CREATE INDEX IF NOT EXISTS flakegraph_artifact_run_idx
    ON flakegraph_artifact (run_id, kind)
    """,
    """
    -- The view gained a priority_band column, and a replace cannot rename or
    -- reorder an existing view's output.
    DROP VIEW IF EXISTS flakegraph_worker_demand
    """,
    """
    CREATE VIEW flakegraph_worker_demand AS
    SELECT demand.stage,
           CASE WHEN demand.priority >= 1000 THEN 'interactive' ELSE 'bulk' END
               AS priority_band,
           COUNT(*)::BIGINT AS desired_workers
    FROM (
        SELECT task.stage, task.priority
        FROM flakegraph_task AS task
        JOIN flakegraph_run AS run ON run.id = task.run_id
        WHERE run.status IN ('queued', 'running')
          AND (
              task.status = 'running'
              OR (
                  task.status = 'queued'
                  AND task.available_at <= CURRENT_TIMESTAMP
              )
          )
          AND (
              (
                  task.stage <> 'finalize_graph'
                  AND task.remaining_dependencies = 0
              )
              OR (
                  task.stage = 'finalize_graph'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM flakegraph_task AS unfinished
                      WHERE unfinished.run_id = task.run_id
                        AND unfinished.id <> task.id
                        AND unfinished.status <> 'succeeded'
                  )
              )
          )
        UNION ALL
        -- A publication carries no priority of its own, so it inherits the band
        -- of the run that produced it. Every task in a run shares one offset,
        -- so any of them answers the question.
        SELECT 'finalize_graph' AS stage,
               COALESCE((
                   SELECT max(sibling.priority)
                   FROM flakegraph_task AS sibling
                   WHERE sibling.run_id = publication.run_id
               ), 0) AS priority
        FROM flakegraph_publication AS publication
        JOIN flakegraph_run AS run ON run.id = publication.run_id
        WHERE run.status = 'running'
          AND (
              publication.status = 'running'
              OR (
                  publication.status = 'queued'
                  AND publication.available_at <= CURRENT_TIMESTAMP
              )
          )
    ) AS demand
    GROUP BY demand.stage,
             CASE WHEN demand.priority >= 1000 THEN 'interactive' ELSE 'bulk' END
    """,
    """
    COMMENT ON VIEW flakegraph_worker_demand IS
    'Autoscaling demand: ready tasks, active leases, and pending graph publications.'
    """,
)
