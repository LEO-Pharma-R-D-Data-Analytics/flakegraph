"""Regression contracts for Snowflake publication and app orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import flakegraph_app.backends.snowflake as snowflake_app
import pytest
from flakegraph_app.backends.snowflake import SnowflakeBackend
from flakegraph_app.models import (
    IngestionRequest,
    OutputDestination,
    ProviderSelection,
    RunSnapshot,
    RuntimeMode,
    SourceKind,
    StorageKind,
)

from kg_processor.adapters.jobs.snowflake import (
    build_complete_job_if_file_queue_drained_statement,
    build_mark_job_running_statement,
)
from kg_processor.adapters.writers.snowflake_bulk import validate_put_result


def test_put_status_rejects_skipped_stage_objects() -> None:
    """A retry must not COPY an older same-name object after Snowflake skips PUT."""

    cursor = SimpleNamespace(
        description=("PUT",),
        fetchall=lambda: [("src", "target", 1, 1, "NONE", "NONE", "SKIPPED", "")],
    )

    with pytest.raises(RuntimeError, match="SKIPPED"):
        validate_put_result(cursor)


def test_native_job_sql_cannot_revive_cancelled_jobs() -> None:
    """Worker startup and queue-drain completion both preserve cancellation."""

    assert "target.STATUS <> 'CANCELLED'" in build_mark_job_running_statement()
    drained = build_complete_job_if_file_queue_drained_statement()
    assert "STATUS = 'RUNNING'" in drained


def test_app_submission_streams_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not duplicate a corpus-sized LIST result into a second million-row list."""

    monkeypatch.setattr(snowflake_app, "_SUBMISSION_BATCH_SIZE", 2)
    monkeypatch.setattr(snowflake_app, "build_run_config", lambda _request: {})
    monkeypatch.setattr(
        snowflake_app,
        "build_spcs_launch",
        lambda _config: SimpleNamespace(
            runtime_config={},
            spec_yaml="kind: Job\n",
            spec_stage="@DB.GRAPH.SPECS",
            spec_file="run.yaml",
            execute_sql="EXECUTE JOB SERVICE test",
            service_identifier="DB.GRAPH.FLAKEGRAPH_APP_TEST",
        ),
    )
    session = _SubmissionSession(5)
    backend = SnowflakeBackend(session)
    expected = RunSnapshot("job-1", "graph-1", "pending", None, None)
    monkeypatch.setattr(backend, "status", lambda *_args, **_kwargs: expected)

    result = backend.submit(_request(tmp_path))

    assert result == expected
    assert session.batch_sizes == [2, 2, 1]
    assert session.list_queries == 1


def test_app_cancel_marks_parent_terminal_before_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent queue-drain completion sees CANCELLED before file leases change."""

    session = _RecordingSession()
    backend = SnowflakeBackend(session)
    monkeypatch.setattr(
        backend,
        "status",
        lambda *_args, **_kwargs: RunSnapshot("job-1", "graph-1", "cancelled", None, None),
    )

    backend.cancel("job-1")

    updates = [sql for sql in session.sql_calls if sql.startswith("UPDATE KG_JOB")]
    assert updates[0].startswith("UPDATE KG_JOB SET STATUS = 'CANCELLED'")
    assert "STATUS IN ('QUEUED', 'CLAIMED')" in updates[1]
    # Cancelling is the owner's to do, so authorisation precedes both writes.
    assert session.sql_calls.index(updates[0]) > 0


def test_native_edge_projection_selects_only_endpoints_in_review_nodes() -> None:
    """Edges and nodes come from one coherent native Snowflake projection."""

    session = _GraphSession()
    graph = SnowflakeBackend(session).load_graph("current", "graph-1")

    edge_query = next(sql for sql in session.sql_calls if "WITH REVIEW_NODES" in sql)
    assert "SOURCE.ID = E.SOURCE_NODE_ID" in edge_query
    assert "TARGET.ID = E.TARGET_NODE_ID" in edge_query
    assert graph.edges == [{"id": "edge-1", "source_node_id": "node-1", "target_node_id": "node-2"}]


class _Row:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def as_dict(self) -> dict[str, object]:
        return self.values


class _Query:
    def __init__(self, rows: list[_Row]) -> None:
        self.rows = rows

    def collect(self) -> list[_Row]:
        return self.rows

    def to_local_iterator(self) -> Any:
        return iter(self.rows)


class _Writer:
    def __init__(self, session: _SubmissionSession, rows: list[tuple[str, ...]]) -> None:
        self.session = session
        self.rows = rows

    def mode(self, mode: str) -> _Writer:
        assert mode == "overwrite"
        return self

    def save_as_table(self, _name: str, *, table_type: str) -> None:
        # Stored procedures reject temporary tables; transient works in both contexts.
        assert table_type == "transient"
        self.session.batch_sizes.append(len(self.rows))


class _Frame:
    def __init__(self, writer: _Writer) -> None:
        self.write = writer


class _FileApi:
    def put_stream(self, *_args: object, **_kwargs: object) -> object:
        return SimpleNamespace(status="UPLOADED")


class _SubmissionSession:
    def __init__(self, object_count: int) -> None:
        self.object_count = object_count
        self.batch_sizes: list[int] = []
        self.list_queries = 0
        self.file = _FileApi()

    def sql(self, sql: str, params: object = None) -> _Query:
        del params
        if sql.startswith("LIST "):
            self.list_queries += 1
            return _Query(
                [
                    _Row(
                        {
                            "name": f"DB/GRAPH/DOCS/file-{index}.pdf",
                            "size": 10,
                            "md5": f"checksum-{index}",
                        }
                    )
                    for index in range(self.object_count)
                ]
            )
        return _Query([])

    def create_dataframe(self, rows: Any, schema: object) -> _Frame:
        assert schema
        values = list(rows)
        return _Frame(_Writer(self, values))


class _RecordingSession:
    def __init__(self) -> None:
        self.sql_calls: list[str] = []

    def sql(self, sql: str, params: object = None) -> _Query:
        del params
        self.sql_calls.append(sql)
        return _Query([])


class _GraphSession(_RecordingSession):
    def sql(self, sql: str, params: object = None) -> _Query:
        del params
        self.sql_calls.append(sql)
        if "UNION ALL" in sql:
            return _Query([])
        if "FROM KG_NODE WHERE" in sql and "WITH REVIEW_NODES" not in sql:
            return _Query([_Row({"ID": "node-1"}), _Row({"ID": "node-2"})])
        if "WITH REVIEW_NODES" in sql:
            return _Query(
                [
                    _Row(
                        {
                            "ID": "edge-1",
                            "SOURCE_NODE_ID": "node-1",
                            "TARGET_NODE_ID": "node-2",
                        }
                    )
                ]
            )
        if "KG_RUN_REPORT" in sql:
            return _Query([])
        return _Query([])


def _request(tmp_path: Path) -> IngestionRequest:
    return IngestionRequest(
        runtime=RuntimeMode.SNOWFLAKE,
        job_id="job-1",
        graph_id="graph-1",
        source_kind=SourceKind.SNOWFLAKE_STAGE,
        source={"stage": "@DB.GRAPH.DOCS", "prefix": "incoming"},
        ocr=ProviderSelection("snowflake_cortex"),
        llm=ProviderSelection("snowflake_cortex", model="model"),
        embedding=ProviderSelection("snowflake_cortex", model="embed", dimension=8),
        output=OutputDestination(StorageKind.LOCAL, tmp_path),
    )
