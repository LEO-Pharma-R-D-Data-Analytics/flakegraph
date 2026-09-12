from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from kg_processor.adapters.jobs.snowflake import (
    SnowflakeJobFileProgressSink,
    SnowflakeJobManager,
    build_claim_job_files_statement,
    build_claim_job_statement,
    build_complete_file_queue_ownership_statement,
    build_complete_job_file_statement,
    build_complete_job_if_file_queue_drained_statement,
    build_complete_job_statement,
    build_create_job_file_submission_table_statement,
    build_create_pending_job_statement,
    build_fail_job_file_statement,
    build_fail_job_statement,
    build_heartbeat_job_files_statement,
    build_heartbeat_job_statement,
    build_insert_job_file_submission_statement,
    build_job_progress_statement,
    build_mark_job_running_statement,
    build_merge_job_file_submission_statement,
    build_reset_job_for_retry_statement,
    build_retry_failed_job_files_statement,
    build_select_claimed_job_files_statement,
    build_select_claimed_job_statement,
    build_update_job_file_progress_statement,
)
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
from kg_processor.application.progress import ProgressEvent
from kg_processor.domain.documents import InputFile
from kg_processor.domain.jobs import JobFileResult


class FakeCursor:
    def __init__(
        self,
        rows: list[Sequence[object] | None],
        all_rows: list[Sequence[object]] | None = None,
        rowcount: int = 1,
    ) -> None:
        self.rows = rows
        self.all_rows = all_rows or []
        self.index = 0
        self.executed: list[tuple[str, Sequence[object] | None]] = []
        self.executed_many: list[tuple[str, Sequence[Sequence[object]], dict[str, object]]] = []
        self.closed = False
        self.rowcount = rowcount

    def execute(self, sql: str, params: Sequence[object] | None = None) -> object:
        self.executed.append((sql, params))
        return None

    def executemany(
        self,
        sql: str,
        params: Sequence[Sequence[object]],
        **kwargs: object,
    ) -> object:
        """Capture a bulk submission without emulating connector internals."""

        self.executed_many.append((sql, params, kwargs))
        self.rowcount = len(params)
        return None

    def fetchone(self) -> Sequence[object] | None:
        row = self.rows[self.index] if self.index < len(self.rows) else None
        self.index += 1
        return row

    def fetchall(self) -> list[object]:
        return list(self.all_rows)

    def close(self) -> object:
        self.closed = True
        return None


class FakeConnection:
    def __init__(
        self,
        rows: list[Sequence[object] | None],
        all_rows: list[Sequence[object]] | None = None,
        rowcount: int = 1,
    ) -> None:
        self.cursor_instance = FakeCursor(rows, all_rows, rowcount)
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> object:
        self.committed = True
        return None

    def rollback(self) -> object:
        self.rolled_back = True
        return None

    def close(self) -> object:
        self.closed = True
        return None


def test_snowflake_job_manager_claims_available_job() -> None:
    connection = FakeConnection([("job_1", "RUNNING", "worker_1")])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    result = manager.claim_job("job_1", "graph", "worker_1", 900, {"safe": True})

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert result.claimed
    assert result.status == "RUNNING"
    assert executed_sql == [
        build_create_pending_job_statement(),
        build_claim_job_statement(900),
        build_select_claimed_job_statement(),
    ]
    assert connection.committed
    assert not connection.closed
    manager.close()
    assert connection.closed


def test_snowflake_job_manager_returns_not_claimed_when_row_is_not_owned() -> None:
    connection = FakeConnection([None])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    result = manager.claim_job("job_1", "graph", "worker_1", 900, {})

    assert not result.claimed
    assert result.job_id == "job_1"
    assert connection.committed


def test_snowflake_job_manager_submits_files_in_one_idempotent_merge() -> None:
    """Verify queue submission persists provenance without changing existing statuses."""

    connection = FakeConnection([])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)
    files = [
        InputFile(
            id="file_1",
            path=Path("incoming/a.pdf"),
            source_uri="@DB.SCHEMA.DOC_STAGE/incoming/a.pdf",
            checksum="checksum-a",
            mime_type="application/pdf",
            size_bytes=123,
        ),
        InputFile(
            id="file_2",
            path=Path("incoming/b.pdf"),
            source_uri="@DB.SCHEMA.DOC_STAGE/incoming/b.pdf",
            checksum="checksum-b",
            mime_type="application/pdf",
            size_bytes=456,
        ),
    ]

    submitted = manager.submit_job_files("job_1", "graph", files, {"safe": True})

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert submitted == 2
    assert executed_sql == [
        build_create_pending_job_statement(),
        build_create_job_file_submission_table_statement(),
        build_merge_job_file_submission_statement(),
    ]
    assert len(connection.cursor_instance.executed_many) == 1
    insert_sql, rows, kwargs = connection.cursor_instance.executed_many[0]
    assert insert_sql == build_insert_job_file_submission_statement()
    assert [row[3] for row in rows] == ["file_1", "file_2"]
    assert [row[5] for row in rows] == ["incoming/a.pdf", "incoming/b.pdf"]
    assert kwargs == {}
    assert "WHEN MATCHED THEN UPDATE" in build_merge_job_file_submission_statement()
    assert (
        "STATUS ="
        not in build_merge_job_file_submission_statement()
        .split("WHEN MATCHED THEN UPDATE", 1)[1]
        .split("WHEN NOT MATCHED", 1)[0]
    )
    assert connection.committed
    assert not connection.closed
    manager.close()
    assert connection.closed


def test_snowflake_job_manager_rejects_empty_submission_before_connecting() -> None:
    """Avoid creating a parent job that no worker could meaningfully process."""

    def factory(**_kwargs: object) -> FakeConnection:
        raise AssertionError("empty submission must not connect")

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    with pytest.raises(ValueError, match="no input files"):
        manager.submit_job_files("job_1", "graph", [], {})


def test_snowflake_job_manager_completes_and_fails_running_job() -> None:
    complete_connection = FakeConnection([])
    fail_connection = FakeConnection([])
    connections = [complete_connection, fail_connection]

    def factory(**_kwargs: object) -> FakeConnection:
        return connections.pop(0)

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    manager.complete_job("job_1", "worker_1", {"files_processed": 1})
    manager.close()
    manager.fail_job("job_2", "worker_1", {"message": "bad"})
    manager.close()

    assert complete_connection.cursor_instance.executed[0][0] == build_complete_job_statement()
    assert fail_connection.cursor_instance.executed[0][0] == build_fail_job_statement()
    assert complete_connection.committed
    assert fail_connection.committed


def test_snowflake_job_manager_rejects_completion_after_lease_loss() -> None:
    connection = FakeConnection([], rowcount=0)

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    with pytest.raises(RuntimeError, match="no longer owned"):
        manager.complete_job("job_1", "stale_worker", {"files_processed": 1})

    assert connection.rolled_back
    assert not connection.committed


def test_snowflake_job_manager_heartbeats_running_job() -> None:
    connection = FakeConnection([])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    manager.heartbeat_job("job_1", "worker_1", 900)

    assert connection.cursor_instance.executed[0] == (
        build_heartbeat_job_statement(900),
        ["job_1", "worker_1"],
    )
    assert connection.committed


def test_snowflake_job_manager_claims_file_batch() -> None:
    connection = FakeConnection(
        [],
        all_rows=[
            (
                "job_1",
                "graph",
                "file_1",
                "@DB.SCHEMA.DOC_STAGE/input/a.pdf",
                "checksum-a",
                "CLAIMED",
                "worker_1",
                2,
            )
        ],
    )

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    claims = manager.claim_job_files("job_1", "graph", "worker_1", 900, 25)

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert [claim.file_id for claim in claims] == ["file_1"]
    assert claims[0].source_uri == "@DB.SCHEMA.DOC_STAGE/input/a.pdf"
    assert claims[0].attempts == 2
    assert executed_sql == [
        build_mark_job_running_statement(),
        build_claim_job_files_statement(900, 25),
        build_select_claimed_job_files_statement(25),
    ]
    claim_params = connection.cursor_instance.executed[1][1]
    select_params = connection.cursor_instance.executed[2][1]
    assert claim_params is not None
    assert select_params is not None
    assert claim_params[0] == "worker_1"
    assert str(claim_params[1]).startswith("claim:worker_1:")
    assert claim_params[2:] == ["job_1", "graph", "job_1", "graph", "job_1", "graph"]
    assert select_params == ["job_1", "graph", "worker_1", claim_params[1]]
    assert connection.committed


@pytest.mark.parametrize(
    ("counts", "expected"),
    [((2, 2), True), ((3, 2), False)],
)
def test_snowflake_job_manager_detects_complete_file_queue_ownership(
    counts: tuple[int, int],
    expected: bool,
) -> None:
    """Only a worker owning every queue row may publish graph-wide artifacts."""

    connection = FakeConnection([counts])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    owns_queue = manager.worker_owns_entire_file_queue(
        "job_1",
        "graph",
        "worker_1",
        ["file_1", "file_2"],
    )

    assert owns_queue is expected
    assert connection.cursor_instance.executed == [
        (
            build_complete_file_queue_ownership_statement(2),
            # Bound to the graph, not the job: the permission a complete claim
            # grants replaces the whole graph, so the proof has to cover it.
            ["worker_1", "file_1", "file_2", "graph"],
        )
    ]
    assert not connection.closed
    manager.close()
    assert connection.closed


def test_snowflake_job_manager_completes_and_fails_claimed_files() -> None:
    complete_connection = FakeConnection([])
    fail_connection = FakeConnection([])
    drain_connection = FakeConnection([])
    connections = [complete_connection, fail_connection, drain_connection]

    def factory(**_kwargs: object) -> FakeConnection:
        return connections.pop(0)

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    manager.complete_job_files(
        "job_1",
        "graph",
        "worker_1",
        [
            JobFileResult(
                file_id="file_1",
                rows_written=9,
                row_counts={"chunks": 2, "nodes": 7},
                stage="written",
                ocr_provider="mineru_internal",
                llm_provider="vllm_local",
                embedding_model="Snowflake/snowflake-arctic-embed-l-v2.0",
                embedding_dimension=1024,
                audit={"quality_ok": True, "write_scope": "file_batch"},
            )
        ],
    )
    manager.close()
    manager.fail_job_files(
        "job_1",
        "graph",
        "worker_1",
        ["file_2"],
        {"message": "bad"},
    )
    manager.close()
    manager.complete_job_if_file_queue_drained("job_1", "graph", {"files_processed": 1})
    manager.close()

    assert complete_connection.cursor_instance.executed_many[0][0] == (
        build_complete_job_file_statement()
    )
    assert complete_connection.cursor_instance.executed_many[0][1][0] == [
        9,
        '{"chunks":2,"nodes":7}',
        "written",
        "mineru_internal",
        "vllm_local",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        1024,
        (
            '{"file_id":"file_1","metadata":{"quality_ok":true,'
            '"write_scope":"file_batch"},"providers":{"embedding_dimension":1024,'
            '"embedding_model":"Snowflake/snowflake-arctic-embed-l-v2.0",'
            '"llm":"vllm_local","ocr":"mineru_internal"},"row_counts":{"chunks":2,'
            '"nodes":7},"rows_written":9,"stage":"written"}'
        ),
        "job_1",
        "graph",
        "file_1",
        "worker_1",
    ]
    assert complete_connection.cursor_instance.executed_many[0][2] == {"num_statements": 1}
    assert fail_connection.cursor_instance.executed_many[0][0] == build_fail_job_file_statement()
    assert fail_connection.cursor_instance.executed_many[0][2] == {"num_statements": 1}
    assert drain_connection.cursor_instance.executed[0][0] == (
        build_complete_job_if_file_queue_drained_statement()
    )


def test_snowflake_job_manager_heartbeats_claimed_files() -> None:
    connection = FakeConnection([])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    manager.heartbeat_job_files("job_1", "graph", "worker_1", ["file_1", "file_2"], 900)

    assert connection.cursor_instance.executed[0] == (
        build_heartbeat_job_files_statement(900, 2),
        ["job_1", "graph", "worker_1", "file_1", "file_2"],
    )
    assert connection.committed


def test_snowflake_job_manager_updates_claimed_file_progress() -> None:
    connection = FakeConnection([])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    manager.update_job_file_progress(
        "job_1",
        "graph",
        "worker_1",
        ["file_1", "file_2"],
        "graph_extraction:started",
    )

    assert connection.cursor_instance.executed[0] == (
        build_update_job_file_progress_statement(2),
        ["graph_extraction:started", "job_1", "graph", "worker_1", "file_1", "file_2"],
    )
    assert connection.committed


def test_snowflake_job_file_progress_sink_targets_claimed_rows() -> None:
    manager = FakeProgressManager()
    sink = SnowflakeJobFileProgressSink(
        cast(SnowflakeJobManager, manager),
        "job_1",
        "graph",
        "worker_1",
        ["file_1", "file_2"],
    )

    sink.emit(
        ProgressEvent(
            job_id="job_1",
            graph_id="graph",
            stage="ocr",
            status="started",
            file_id="file_1",
        )
    )
    sink.emit(
        ProgressEvent(
            job_id="job_1",
            graph_id="graph",
            stage="merge",
            status="completed",
        )
    )
    sink.emit(
        ProgressEvent(
            job_id="job_1",
            graph_id="graph",
            stage="ocr",
            status="started",
            file_id="unclaimed",
        )
    )

    # STAGE names a stage, not a stage-and-status pair. The app groups file rows
    # by this column and merges the job row's live counts onto the stage of the
    # same name, so a composite value renders one stage as two rows.
    assert manager.updates == [
        ("job_1", "graph", "worker_1", ["file_1"], "ocr"),
        ("job_1", "graph", "worker_1", ["file_1", "file_2"], "merge"),
    ]


def test_claim_statement_rejects_non_positive_lease() -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        build_claim_job_statement(0)


def test_claim_job_files_statement_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        build_claim_job_files_statement(0, 100)
    with pytest.raises(ValueError, match="batch_size"):
        build_claim_job_files_statement(900, 0)


def test_job_file_claim_sql_uses_parent_job_state_and_claim_token() -> None:
    claim_sql = build_claim_job_files_statement(900, 25)
    select_sql = build_select_claimed_job_files_statement(25)
    drain_sql = build_complete_job_if_file_queue_drained_statement()

    assert "STAGE = ?" in claim_sql
    assert "AND EXISTS (SELECT 1 FROM KG_JOB" in claim_sql
    assert "STATUS = 'RUNNING'" in claim_sql
    assert "AND WORKER_ID = ? AND STAGE = ?" in select_sql
    assert "STATUS IN ('QUEUED', 'CLAIMED')" in drain_sql
    assert "'failed_files'" in drain_sql
    assert "'FAILED', 'SUCCEEDED'" in drain_sql
    assert "target.STATUS NOT IN ('SUCCEEDED', 'FAILED')" in build_mark_job_running_statement()
    assert "LEASE_EXPIRES_AT >= CURRENT_TIMESTAMP()" in build_complete_job_statement()
    assert "LEASE_UNTIL >= CURRENT_TIMESTAMP()" in build_complete_job_file_statement()


def test_heartbeat_statements_reject_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        build_heartbeat_job_statement(0)
    with pytest.raises(ValueError, match="lease_seconds"):
        build_heartbeat_job_files_statement(0, 1)
    with pytest.raises(ValueError, match="file_count"):
        build_heartbeat_job_files_statement(900, 0)
    with pytest.raises(ValueError, match="file_count"):
        build_update_job_file_progress_statement(0)


def _config() -> SnowflakeConnectionConfig:
    return SnowflakeConnectionConfig(
        account="account",
        host=None,
        user="user",
        password="password",
        authenticator=None,
        private_key_path=None,
        database="DB",
        schema_name="SCHEMA",
        role="ROLE",
        warehouse="WH",
    )


class FakeProgressManager:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str, str, list[str], str]] = []

    def update_job_file_progress(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        file_ids: list[str],
        stage: str,
    ) -> None:
        self.updates.append((job_id, graph_id, worker_id, file_ids, stage))


class SequencedCursor(FakeCursor):
    """Return a different result set per fetchall, for before/after comparisons."""

    def __init__(self, result_sets: list[list[Sequence[object]]]) -> None:
        super().__init__([])
        self.result_sets = result_sets
        self.fetch_index = 0

    def fetchall(self) -> list[object]:
        if self.fetch_index < len(self.result_sets):
            rows = self.result_sets[self.fetch_index]
            self.fetch_index += 1
            return list(rows)
        return []


class SequencedConnection(FakeConnection):
    cursor_instance: SequencedCursor

    def __init__(self, result_sets: list[list[Sequence[object]]]) -> None:
        super().__init__([])
        self.cursor_instance = SequencedCursor(result_sets)

    def cursor(self) -> SequencedCursor:
        return self.cursor_instance


def test_retry_statement_respects_the_attempt_budget() -> None:
    """A poisoned document must not be retried forever by repeating the command."""

    statement = build_retry_failed_job_files_statement(3)

    assert "STATUS = 'QUEUED'" in statement
    assert "STATUS = 'FAILED'" in statement
    assert "COALESCE(ATTEMPTS, 0) < 3" in statement
    assert "ERROR = NULL" in statement
    assert "LEASE_UNTIL = NULL" in statement


def test_retry_statement_rejects_a_meaningless_budget() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        build_retry_failed_job_files_statement(0)


def test_job_reset_targets_settled_jobs_only() -> None:
    """Claiming accepts PENDING or FAILED, so a SUCCEEDED job must be reopened.

    A drained run marks the job SUCCEEDED. Without this reset a requeued file
    queue would never be claimed, because the file claim requires a RUNNING job.
    """

    statement = build_reset_job_for_retry_statement()

    assert "STATUS = 'PENDING'" in statement
    assert "STATUS IN ('SUCCEEDED', 'FAILED')" in statement
    assert "LEASE_OWNER = NULL" in statement


def test_retry_requeues_failed_files_and_reports_exhausted_rows() -> None:
    connection = SequencedConnection(
        [
            [("FAILED", 10)],
            [("QUEUED", 8), ("FAILED", 2)],
        ]
    )

    def factory(**_kwargs: object) -> SequencedConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    result = manager.retry_failed_job_files("job_1", "graph_1", 3)

    assert result.files_requeued == 8
    assert result.files_exhausted == 2
    assert connection.committed
    statements = [sql for sql, _ in connection.cursor_instance.executed]
    assert any("STATUS = 'QUEUED'" in sql for sql in statements)
    assert any("STATUS = 'PENDING'" in sql for sql in statements)


def test_queue_status_counts_are_reported_per_status() -> None:
    connection = FakeConnection([], all_rows=[("DONE", 7), ("FAILED", 3)])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    assert manager.count_job_files_by_status("job_1", "graph_1") == {"DONE": 7, "FAILED": 3}


class _RecordingProgressManager:
    """Job manager stand-in that records graph-level progress writes."""

    def __init__(self, fail: bool = False) -> None:
        self.progress_payloads: list[dict[str, object]] = []
        self.updates: list[tuple[object, ...]] = []
        self.fail = fail

    def update_job_progress(
        self,
        job_id: str,
        graph_id: str,
        lease_owner: str,
        progress: dict[str, object],
    ) -> None:
        if self.fail:
            raise RuntimeError("warehouse unavailable")
        self.progress_payloads.append(progress)

    def update_job_file_progress(self, *args: object, **kwargs: object) -> None:
        self.updates.append(args)


def _extraction_event(status: str, completed: int) -> ProgressEvent:
    return ProgressEvent(
        job_id="job",
        graph_id="graph",
        stage="graph_extraction",
        status=status,
        counts={"batches_completed": completed, "batches_total": 21},
    )


def test_progress_writes_are_throttled_but_keep_every_transition() -> None:
    """Progress reporting must not become a real share of a run's query load.

    A stage can emit per batch or per candidate. Dropping a stage transition,
    though, would leave a finished stage rendered as still running.
    """

    manager = _RecordingProgressManager()
    sink = SnowflakeJobFileProgressSink(
        cast(Any, manager),
        job_id="job",
        graph_id="graph",
        worker_id="worker",
        claimed_file_ids=["f1"],
    )

    sink.emit(_extraction_event("progress", 1))
    sink.emit(_extraction_event("progress", 2))
    sink.emit(_extraction_event("completed", 21))

    recorded = [(p["stage"], p["status"]) for p in manager.progress_payloads]
    assert recorded == [
        ("graph_extraction", "progress"),
        ("graph_extraction", "completed"),
    ]
    assert manager.progress_payloads[-1]["counts"] == {
        "batches_completed": 21,
        "batches_total": 21,
    }


def test_a_failed_progress_write_never_ends_the_run() -> None:
    """The lease heartbeat is authoritative for liveness, not observability."""

    manager = _RecordingProgressManager(fail=True)
    sink = SnowflakeJobFileProgressSink(
        cast(Any, manager),
        job_id="job",
        graph_id="graph",
        worker_id="worker",
        claimed_file_ids=["f1"],
    )

    sink.emit(_extraction_event("progress", 1))

    assert manager.progress_payloads == []


def test_progress_is_recorded_for_a_run_that_takes_no_job_lease() -> None:
    """A file-queue run leaves KG_JOB.LEASE_OWNER null.

    Requiring the lease owner to match made the write a no-op rather than an
    error, so nothing surfaced and the app kept reporting file-level zeros while
    the worker was plainly busy.
    """

    statement = build_job_progress_statement()

    assert "LEASE_OWNER IS NULL OR LEASE_OWNER = ?" in statement


def test_file_stage_and_job_progress_agree_on_one_stage_name() -> None:
    """The two halves the run page joins must name the same stage identically.

    The app groups KG_JOB_FILE rows by STAGE to derive each stage's state, then
    merges the live counts from KG_JOB.PROGRESS onto the stage of the same name.
    While the file column stored ``graph_extraction:progress`` the names could
    never match, so a single running stage was drawn twice: a stale file-derived
    row at 0/9 titled "Graph Extraction:Progress", and a live duplicate.
    """

    manager = _RecordingProgressManager()
    sink = SnowflakeJobFileProgressSink(
        cast(Any, manager),
        job_id="job",
        graph_id="graph",
        worker_id="worker",
        claimed_file_ids=["f1"],
    )

    sink.emit(_extraction_event("progress", 3))

    file_stages = {update[-1] for update in manager.updates}
    payload_stages = {str(payload["stage"]) for payload in manager.progress_payloads}
    assert file_stages == payload_stages == {"graph_extraction"}


class _TransactionLoggingCursor:
    """Record statements in the same ordered log as its connection's transaction events."""

    def __init__(
        self,
        log: list[tuple[str, object]],
        rows: list[Sequence[object] | None],
        all_rows: list[Sequence[object]],
        fail_after: int | None,
    ) -> None:
        self.log = log
        self.rows = rows
        self.all_rows = all_rows
        self.index = 0
        self.executed = 0
        self.fail_after = fail_after
        self.closed = False

    def execute(self, sql: str, params: Sequence[object] | None = None) -> object:
        """Log the statement and optionally emulate a mid-claim connector failure."""

        self.executed += 1
        self.log.append(("execute", sql))
        if self.fail_after is not None and self.executed > self.fail_after:
            raise RuntimeError("connector lost the session")
        return None

    def fetchone(self) -> Sequence[object] | None:
        """Return the next prepared row for a confirming single-row read."""

        row = self.rows[self.index] if self.index < len(self.rows) else None
        self.index += 1
        return row

    def fetchall(self) -> list[object]:
        """Return the prepared rows for a confirming batch read."""

        return list(self.all_rows)

    def close(self) -> object:
        """Record cursor release without emulating connector internals."""

        self.closed = True
        return None


class _TransactionLoggingConnection:
    """Expose the connector autocommit hook and order every transaction event."""

    def __init__(
        self,
        rows: list[Sequence[object] | None] | None = None,
        all_rows: list[Sequence[object]] | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.log: list[tuple[str, object]] = []
        self.cursor_instance = _TransactionLoggingCursor(
            self.log,
            rows or [],
            all_rows or [],
            fail_after,
        )

    def cursor(self) -> _TransactionLoggingCursor:
        """Return the single cursor whose statements share the connection log."""

        return self.cursor_instance

    def autocommit(self, enabled: bool) -> None:
        """Record the connector's transaction-mode toggle."""

        self.log.append(("autocommit", enabled))

    def commit(self) -> object:
        """Record a commit."""

        self.log.append(("commit", None))
        return None

    def rollback(self) -> object:
        """Record a rollback."""

        self.log.append(("rollback", None))
        return None

    def close(self) -> object:
        """Record connection release."""

        self.log.append(("close", None))
        return None


def test_job_claim_opens_a_transaction_before_its_first_statement() -> None:
    """The insert, the lease claim, and the confirming read must commit as one unit."""

    connection = _TransactionLoggingConnection(rows=[("job_1", "RUNNING", "worker_1")])

    def factory(**_kwargs: object) -> _TransactionLoggingConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=cast(Any, factory))

    result = manager.claim_job("job_1", "graph", "worker_1", 900, {"safe": True})

    assert result.claimed is True
    assert connection.log[0] == ("autocommit", False)
    assert ("commit", None) in connection.log
    assert connection.log[-1] == ("autocommit", True)


def test_file_claim_opens_a_transaction_before_its_first_statement() -> None:
    """Claiming file rows and reading them back must commit as one unit."""

    connection = _TransactionLoggingConnection(all_rows=[])

    def factory(**_kwargs: object) -> _TransactionLoggingConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=cast(Any, factory))

    assert manager.claim_job_files("job_1", "graph", "worker_1", 900, 4) == []
    assert connection.log[0] == ("autocommit", False)
    assert ("commit", None) in connection.log
    assert connection.log[-1] == ("autocommit", True)


def test_a_failed_claim_confirmation_releases_the_rows_it_claimed() -> None:
    """A read that fails after the claim must not strand rows on an unaware worker."""

    connection = _TransactionLoggingConnection(fail_after=2)

    def factory(**_kwargs: object) -> _TransactionLoggingConnection:
        return connection

    manager = SnowflakeJobManager(_config(), connector_factory=cast(Any, factory))

    with pytest.raises(RuntimeError):
        manager.claim_job_files("job_1", "graph", "worker_1", 900, 4)

    assert connection.log[0] == ("autocommit", False)
    assert ("rollback", None) in connection.log
    assert ("commit", None) not in connection.log
