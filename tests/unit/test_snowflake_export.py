from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pytest
from snowflake_fakes import FakeConnection, FakeCursor

from kg_processor.application.snowflake_export import export_snowflake_graph
from kg_processor.application.snowflake_schema import snowflake_schema_columns
from kg_processor.config.settings import Settings

_ROWS: dict[str, list[object]] = {
    "KG_NODE": [
        ("node-1", "g1", "aikido", "Aikido", "CONCEPT", None, None, None, None, None, 1, 0.5),
    ],
    "KG_EDGE": [
        (
            "edge-1",
            "g1",
            "node-1",
            "node-2",
            "RELATED_TO",
            None,
            1.0,
            0.9,
            None,
            None,
            None,
            1,
            None,
        ),
    ],
}


class ExportCursor(FakeCursor):
    """Answer each table read with that table's rows, so every export has data to write."""

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        timeout: int | None = None,
    ) -> object:
        super().execute(sql, params, timeout=timeout)
        self.result_sets = [rows for table, rows in _ROWS.items() if f".{table} " in sql]
        return None


def _connection() -> FakeConnection:
    return FakeConnection(cursor=ExportCursor())


def _settings() -> Settings:
    return Settings.model_validate(
        {
            "snowflake": {
                "account": "EXAMPLE_ACCOUNT",
                "user": "someone@example.com",
                "database": "EXAMPLE_DB",
                "schema": "EXAMPLE_SCHEMA",
                "role": "EXAMPLE_ROLE",
                "warehouse": "EXAMPLE_WH",
            }
        }
    )


def test_export_writes_every_inspection_artifact(tmp_path: Path) -> None:
    """Inspection reads a fixed table set, so every one must exist after export.

    A missing parquet file is read as an empty table rather than an error, which
    would silently understate a graph during gold evaluation.
    """

    connection = _connection()
    output = tmp_path / "kg"

    result = export_snowflake_graph(
        _settings(), "g1", output, connector_factory=lambda **_: connection
    )

    expected = {
        "documents",
        "pages",
        "blocks",
        "assets",
        "chunks",
        "nodes",
        "edges",
        "edge_observations",
        "evidence",
        "entity_sources",
        "communities",
        "community_findings",
    }
    assert {path.stem for path in output.glob("*.parquet")} == expected
    assert result["tables"]["nodes"] == 1
    assert result["tables"]["edges"] == 1
    assert connection.closed


def test_export_uses_lower_case_artifact_columns(tmp_path: Path) -> None:
    """Inspection addresses columns by lower-case name; Snowflake returns upper."""

    output = tmp_path / "kg"

    export_snowflake_graph(_settings(), "g1", output, connector_factory=lambda **_: _connection())

    nodes = pd.read_parquet(output / "nodes.parquet")

    assert "name" in nodes.columns
    assert "NAME" not in nodes.columns
    assert nodes.loc[0, "name"] == "Aikido"


def test_export_excludes_bookkeeping_columns(tmp_path: Path) -> None:
    """UPDATED_AT is a storage detail and is not part of the graph contract."""

    output = tmp_path / "kg"

    export_snowflake_graph(_settings(), "g1", output, connector_factory=lambda **_: _connection())

    nodes = pd.read_parquet(output / "nodes.parquet")

    assert "updated_at" not in nodes.columns
    assert "UPDATED_AT" in snowflake_schema_columns()["KG_NODE"]


def test_export_scopes_every_read_to_the_requested_graph(tmp_path: Path) -> None:
    """A shared schema holds many graphs, so an unscoped read would merge them."""

    connection = _connection()

    export_snowflake_graph(
        _settings(), "g1", tmp_path / "kg", connector_factory=lambda **_: connection
    )

    statements = connection.cursor_instance.executed

    assert statements
    for sql, params in statements:
        assert "WHERE GRAPH_ID = ?" in sql
        assert params == ["g1"]


def test_export_rejects_empty_graph_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="graph id"):
        export_snowflake_graph(
            _settings(), "", tmp_path / "kg", connector_factory=lambda **_: _connection()
        )


@pytest.mark.parametrize(
    "target",
    [
        {"database": 'EXAMPLE_DB.PUBLIC.KG_NODE WHERE 1=1 OR "'},
        {"schema": "EXAMPLE_SCHEMA; DROP TABLE KG_NODE; --"},
    ],
)
def test_export_rejects_a_name_that_is_not_an_identifier(
    tmp_path: Path,
    target: dict[str, str],
) -> None:
    """Database and schema reach the statement as text and cannot be bound."""

    settings = Settings.model_validate(
        {
            "snowflake": {
                "account": "EXAMPLE_ACCOUNT",
                "user": "someone@example.com",
                "database": "EXAMPLE_DB",
                "schema": "EXAMPLE_SCHEMA",
                **target,
            }
        }
    )

    with pytest.raises(ValueError, match="must be an unquoted identifier"):
        export_snowflake_graph(
            settings, "g1", tmp_path / "kg", connector_factory=lambda **_: _connection()
        )


def test_export_accepts_a_lower_case_configured_name(tmp_path: Path) -> None:
    """Snowflake resolves unquoted identifiers case-insensitively."""

    settings = Settings.model_validate(
        {
            "snowflake": {
                "account": "EXAMPLE_ACCOUNT",
                "user": "someone@example.com",
                "database": "example_db",
                "schema": "example_schema",
            }
        }
    )
    connection = _connection()

    export_snowflake_graph(
        settings, "g1", tmp_path / "kg", connector_factory=lambda **_: connection
    )

    assert all(
        "EXAMPLE_DB.EXAMPLE_SCHEMA." in sql for sql, _ in connection.cursor_instance.executed
    )
