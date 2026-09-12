from __future__ import annotations

import json
from collections.abc import Sequence

from kg_processor.adapters.cache.snowflake import (
    SnowflakeCache,
    build_extraction_cache_merge_statement,
    build_ocr_cache_merge_statement,
)
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
from kg_processor.domain.documents import ParsedDocument, ParsedPage
from kg_processor.domain.graph import ExtractedEntity, ExtractionResult
from kg_processor.ports.cache import ExtractionCacheKey, OcrCacheKey


class FakeCursor:
    def __init__(self, rows: list[Sequence[object] | None]) -> None:
        self.rows = rows
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
        return []

    def close(self) -> object:
        self.closed = True
        return None


class FakeConnection:
    def __init__(self, rows: list[Sequence[object] | None]) -> None:
        self.cursor_instance = FakeCursor(rows)
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


def test_snowflake_cache_reads_and_writes_ocr_document() -> None:
    document = ParsedDocument(
        file_id="file_1",
        checksum="checksum",
        source_uri="file:///sample.txt",
        mime_type="text/plain",
        pages=[ParsedPage(page_number=1, markdown="Alice", raw_text="Alice")],
    )
    connection = FakeConnection([(json.dumps(document.model_dump(mode="json")),)])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    cache = SnowflakeCache(_config(), connector_factory=factory)
    key = OcrCacheKey(
        id="ocr_1",
        file_id="file_1",
        checksum="checksum",
        ocr_provider="builtin_text",
        options_hash="options",
    )

    assert cache.get_ocr_document(key) == document
    cache.put_ocr_document(key, document)

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert executed_sql[0] == "SELECT PARSED_DOCUMENT FROM KG_OCR_CACHE WHERE ID = ?"
    assert executed_sql[1] == build_ocr_cache_merge_statement()
    assert connection.committed
    assert not connection.closed
    cache.close()
    assert connection.closed


def test_snowflake_cache_reads_and_writes_extraction_result() -> None:
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Alice",
                type="PERSON",
                description="Alice is mentioned.",
                source_chunk_id="chunk_1",
            )
        ],
        relations=[],
    )
    connection = FakeConnection([(extraction.model_dump(mode="json"),)])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    cache = SnowflakeCache(_config(), connector_factory=factory)
    key = ExtractionCacheKey(
        id="extraction_1",
        graph_id="graph",
        chunk_batch_hash="chunks",
        llm_provider="fake",
        model="fake",
        options_hash="options",
    )

    assert cache.get_extraction_result(key) == extraction
    cache.put_extraction_result(key, extraction)

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert executed_sql[0] == "SELECT RESULT FROM KG_EXTRACTION_CACHE WHERE ID = ?"
    assert executed_sql[1] == build_extraction_cache_merge_statement()
    assert connection.committed


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
