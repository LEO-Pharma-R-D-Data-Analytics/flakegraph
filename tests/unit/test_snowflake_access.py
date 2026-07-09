from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from kg_processor.application.snowflake_access import (
    REQUIRED_SNOWFLAKE_COLUMNS,
    REQUIRED_SNOWFLAKE_TABLES,
    run_snowflake_access_check,
)
from kg_processor.config.settings import Settings


class FakeCursor:
    def __init__(
        self,
        visible_tables: set[str] | None = None,
        visible_columns: dict[str, set[str]] | None = None,
        fail_document_stage: bool = False,
    ) -> None:
        self.visible_tables = visible_tables or set(REQUIRED_SNOWFLAKE_TABLES)
        self.visible_columns = visible_columns or {
            table: set(REQUIRED_SNOWFLAKE_COLUMNS[table]) for table in self.visible_tables
        }
        self.fail_document_stage = fail_document_stage
        self.executed: list[tuple[str, Sequence[object] | None]] = []
        self.fetchone_result: Sequence[object] | None = None
        self.fetchall_result: list[Sequence[object]] = []
        self.closed = False

    def execute(self, sql: str, params: Sequence[object] | None = None) -> object:
        self.executed.append((sql, params))
        self.fetchone_result = None
        self.fetchall_result = []
        if sql.startswith("SELECT CURRENT_ACCOUNT"):
            self.fetchone_result = [
                "EXAMPLE_ACCOUNT",
                "AWS_EU_CENTRAL_1",
                "USER",
                "KG_PROCESSOR_ROLE",
                "KG_DB",
                "GRAPH",
                "WH",
            ]
        elif "INFORMATION_SCHEMA.TABLES" in sql:
            self.fetchall_result = [(table,) for table in sorted(self.visible_tables)]
        elif "INFORMATION_SCHEMA.COLUMNS" in sql:
            self.fetchall_result = [
                (table, column)
                for table, columns in sorted(self.visible_columns.items())
                for column in sorted(columns)
            ]
        elif sql.startswith("SHOW WAREHOUSES"):
            self.fetchall_result = [("created", "WH")]
        elif sql.startswith("LIST @KG_DB.GRAPH.KG_DOCS"):
            if self.fail_document_stage:
                raise RuntimeError("stage not authorized")
            self.fetchall_result = []
        elif sql.startswith("LIST @KG_DB.GRAPH.KG_LOAD_STAGE") or sql.startswith(
            "LIST @KG_DB.GRAPH.KG_SERVICE_SPECS"
        ):
            self.fetchall_result = []
        elif sql.startswith("SHOW COMPUTE POOLS"):
            self.fetchall_result = [("KG_PROCESSOR_CPU_POOL",)]
        elif sql.startswith("SHOW IMAGE REPOSITORIES"):
            self.fetchall_result = [("KG_IMAGES",)]
        elif sql.startswith("SELECT AI_COMPLETE"):
            self.fetchone_result = [{"structured_output": {"ok": True}}]
        elif sql.startswith("SELECT AI_EMBED"):
            self.fetchone_result = [[0.1, 0.2, 0.3]]
        else:
            raise AssertionError(f"Unexpected SQL: {sql}")
        return None

    def fetchone(self) -> Sequence[object] | None:
        return self.fetchone_result

    def fetchall(self) -> list[object]:
        return list(self.fetchall_result)

    def close(self) -> object:
        self.closed = True
        return None


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_instance = cursor
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> object:
        return None

    def rollback(self) -> object:
        return None

    def close(self) -> object:
        self.closed = True
        return None


def test_snowflake_access_check_reports_configured_objects_and_cortex_access() -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    report = run_snowflake_access_check(_settings(), connector_factory=factory)

    assert report.ok
    assert {check.name for check in report.checks} == {
        "connection_context",
        "warehouse",
        "target_tables",
        "target_table_columns",
        "document_stage",
        "bulk_stage",
        "service_spec_stage",
        "compute_pool",
        "image_repository",
        "cortex_llm",
        "cortex_embedding",
    }
    assert any(sql.startswith("LIST @KG_DB.GRAPH.KG_DOCS/incoming") for sql, _ in cursor.executed)
    assert any(sql.startswith("SELECT AI_COMPLETE") for sql, _ in cursor.executed)
    assert connection.closed
    assert cursor.closed


def test_snowflake_access_check_reports_missing_table_and_stage_error() -> None:
    cursor = FakeCursor(
        visible_tables=set(REQUIRED_SNOWFLAKE_TABLES) - {"KG_NODE"},
        fail_document_stage=True,
    )
    connection = FakeConnection(cursor)

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    report = run_snowflake_access_check(_settings(), connector_factory=factory)

    assert not report.ok
    target_tables = next(check for check in report.checks if check.name == "target_tables")
    document_stage = next(check for check in report.checks if check.name == "document_stage")
    assert target_tables.details["missing"] == ["KG_NODE"]
    assert not document_stage.ok
    assert document_stage.details["message"] == "stage not authorized"


def test_snowflake_access_check_reports_missing_required_column() -> None:
    visible_columns = {
        table: set(columns) for table, columns in REQUIRED_SNOWFLAKE_COLUMNS.items()
    }
    visible_columns["KG_NODE"].remove("NAME")
    cursor = FakeCursor(visible_columns=visible_columns)
    connection = FakeConnection(cursor)

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    report = run_snowflake_access_check(_settings(), connector_factory=factory)

    assert not report.ok
    target_columns = next(check for check in report.checks if check.name == "target_table_columns")
    assert target_columns.details["missing"] == {"KG_NODE": ["NAME"]}


def test_snowflake_access_check_reports_connection_failure() -> None:
    def factory(**_kwargs: object) -> FakeConnection:
        raise RuntimeError("login failed")

    report = run_snowflake_access_check(_settings(), connector_factory=factory)

    assert not report.ok
    assert report.checks[0].name == "connect"
    assert report.checks[0].details["message"] == "login failed"


def test_snowflake_access_check_rejects_system_compute_pool_for_spcs_job() -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    report = run_snowflake_access_check(
        _settings({"snowflake": {"compute_pool": "SYSTEM_COMPUTE_POOL_GPU"}}),
        connector_factory=factory,
    )

    compute_pool = next(check for check in report.checks if check.name == "compute_pool")
    assert not compute_pool.ok
    assert compute_pool.details["requires_dedicated_pool"] is True
    assert "not compatible" in compute_pool.message


def _settings(overrides: dict[str, Any] | None = None) -> Settings:
    base: dict[str, Any] = {
        "runtime": {"runtime": "spcs"},
        "job": {
            "job_id": "job-123",
            "graph_id": "graph-123",
            "use_lease": True,
            "lease_owner": "worker-1",
        },
        "files": {"source": "snowflake_stage", "stage_prefix": "incoming"},
        "ocr": {"provider": "snowflake_cortex"},
        "llm": {"provider": "snowflake_cortex", "model": "llama3.3-70b"},
        "embedding": {
            "provider": "snowflake_cortex",
            "model": "snowflake-arctic-embed-l-v2.0",
            "dimension": 1024,
        },
        "writer": {"provider": "snowflake_bulk"},
        "cache": {"provider": "snowflake"},
        "snowflake": {
            "account": "EXAMPLE_ACCOUNT",
            "database": "KG_DB",
            "schema": "GRAPH",
            "role": "KG_PROCESSOR_ROLE",
            "warehouse": "WH",
            "stage": "@KG_DB.GRAPH.KG_DOCS",
            "bulk_stage": "@KG_DB.GRAPH.KG_LOAD_STAGE",
            "image_repository": "KG_DB.GRAPH.KG_IMAGES",
            "compute_pool": "KG_PROCESSOR_CPU_POOL",
            "service_spec_stage": "@KG_DB.GRAPH.KG_SERVICE_SPECS",
        },
    }
    if overrides:
        _deep_update(base, overrides)
    return Settings.load(overrides=base)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
