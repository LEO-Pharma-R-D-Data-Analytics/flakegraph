from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from kg_processor.adapters.files.snowflake_stage import SnowflakeStageFileSource
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig


class FakeCursor:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, Sequence[object] | None]] = []
        self.closed = False

    def execute(self, sql: str, params: Sequence[object] | None = None) -> object:
        self.executed.append((sql, params))
        return None

    def fetchone(self) -> Sequence[object] | None:
        return None

    def fetchall(self) -> list[object]:
        return self.rows

    def close(self) -> object:
        self.closed = True
        return None


class FakeConnection:
    def __init__(self, rows: list[object]) -> None:
        self.cursor_instance = FakeCursor(rows)
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


def test_snowflake_stage_file_source_lists_and_filters_supported_files() -> None:
    connection = FakeConnection(
        [
            {
                "name": "DB.SCHEMA.DOC_STAGE/input/a.pdf",
                "size": 42,
                "md5": "checksum-a",
            },
            ("DB.SCHEMA.DOC_STAGE/input/notes.txt", 12, "checksum-b"),
            ("DB.SCHEMA.DOC_STAGE/input/image.png", 10, "checksum-c"),
        ]
    )

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    source = SnowflakeStageFileSource(
        _config(),
        "@DB.SCHEMA.DOC_STAGE",
        prefix="input",
        include_globs=["**/*.pdf", "*.txt"],
        connector_factory=factory,
    )

    files = source.list_files()

    assert [file.path for file in files] == [Path("input/a.pdf"), Path("input/notes.txt")]
    assert files[0].source_uri == "@DB.SCHEMA.DOC_STAGE/input/a.pdf"
    assert files[0].mime_type == "application/pdf"
    assert connection.cursor_instance.executed == [("LIST @DB.SCHEMA.DOC_STAGE/input", None)]
    assert connection.cursor_instance.closed
    assert connection.closed


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
