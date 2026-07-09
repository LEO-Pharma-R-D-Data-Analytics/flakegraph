from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from kg_processor.adapters.jobs.snowflake import (
    SnowflakeJobFileProgressSink,
    SnowflakeJobManager,
    build_claim_job_files_statement,
    build_claim_job_statement,
    build_complete_job_file_statement,
    build_complete_job_if_file_queue_drained_statement,
    build_complete_job_statement,
    build_create_pending_job_statement,
    build_fail_job_file_statement,
    build_fail_job_statement,
    build_heartbeat_job_files_statement,
    build_heartbeat_job_statement,
    build_mark_job_running_statement,
    build_select_claimed_job_files_statement,
    build_select_claimed_job_statement,
    build_update_job_file_progress_statement,
)
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
from kg_processor.application.progress import ProgressEvent
from kg_processor.domain.jobs import JobFileResult


class FakeCursor:
    def __init__(
        self,
        rows: list[Sequence[object] | None],
        all_rows: list[Sequence[object]] | None = None,
    ) -> None:
        self.rows = rows
        self.all_rows = all_rows or []
        self.index = 0
        self.executed: list[tuple[str, Sequence[object] | None]] = []
        self.closed = False

    def execute(self, sql: str, params: Sequence[object] | None = None) -> object:
        self.executed.append((sql, params))
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
    ) -> None:
        self.cursor_instance = FakeCursor(rows, all_rows)
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


def test_snowflake_job_manager_completes_and_fails_running_job() -> None:
    complete_connection = FakeConnection([])
    fail_connection = FakeConnection([])
    connections = [complete_connection, fail_connection]

    def factory(**_kwargs: object) -> FakeConnection:
        return connections.pop(0)

    manager = SnowflakeJobManager(_config(), connector_factory=factory)

    manager.complete_job("job_1", "worker_1", {"files_processed": 1})
    manager.fail_job("job_2", "worker_1", {"message": "bad"})

    assert complete_connection.cursor_instance.executed[0][0] == build_complete_job_statement()
    assert fail_connection.cursor_instance.executed[0][0] == build_fail_job_statement()
    assert complete_connection.committed
    assert fail_connection.committed


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
    assert connection.cursor_instance.executed[1][1] == [
        "worker_1",
        "job_1",
        "graph",
        "job_1",
        "graph",
    ]
    assert connection.committed


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
    manager.fail_job_files(
        "job_1",
        "graph",
        "worker_1",
        ["file_2"],
        {"message": "bad"},
    )
    manager.complete_job_if_file_queue_drained("job_1", "graph", {"files_processed": 1})

    assert complete_connection.cursor_instance.executed[0][0] == (
        build_complete_job_file_statement()
    )
    assert complete_connection.cursor_instance.executed[0][1] == [
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
    assert fail_connection.cursor_instance.executed[0][0] == build_fail_job_file_statement()
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

    assert manager.updates == [
        ("job_1", "graph", "worker_1", ["file_1"], "ocr:started"),
        ("job_1", "graph", "worker_1", ["file_1", "file_2"], "merge:completed"),
    ]


def test_claim_statement_rejects_non_positive_lease() -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        build_claim_job_statement(0)


def test_claim_job_files_statement_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="lease_seconds"):
        build_claim_job_files_statement(0, 100)
    with pytest.raises(ValueError, match="batch_size"):
        build_claim_job_files_statement(900, 0)


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
