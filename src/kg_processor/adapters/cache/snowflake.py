"""Snowflake-backed OCR/extraction cache adapter."""

from __future__ import annotations

import json
from collections.abc import Sequence

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    connect_snowflake,
)
from kg_processor.domain.documents import ParsedDocument
from kg_processor.domain.graph import ExtractionResult
from kg_processor.ports.cache import ExtractionCacheKey, OcrCacheKey


class SnowflakeCache:
    """Persists OCR and extraction cache entries in Snowflake VARIANT columns."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        self.config = config
        self.connector_factory = connector_factory

    def get_ocr_document(self, key: OcrCacheKey) -> ParsedDocument | None:
        """Fetch and validate a cached OCR document from KG_OCR_CACHE."""

        value = self._fetch_variant("KG_OCR_CACHE", "PARSED_DOCUMENT", key.id)
        return ParsedDocument.model_validate(value) if value is not None else None

    def put_ocr_document(self, key: OcrCacheKey, document: ParsedDocument) -> None:
        """Merge a parsed OCR document into KG_OCR_CACHE."""

        self._execute_merge(
            build_ocr_cache_merge_statement(),
            [
                key.id,
                key.file_id,
                key.checksum,
                key.ocr_provider,
                key.options_hash,
                _json(document.model_dump(mode="json")),
            ],
        )

    def get_extraction_result(self, key: ExtractionCacheKey) -> ExtractionResult | None:
        """Fetch and validate a cached LLM extraction result."""

        value = self._fetch_variant("KG_EXTRACTION_CACHE", "RESULT", key.id)
        return ExtractionResult.model_validate(value) if value is not None else None

    def put_extraction_result(self, key: ExtractionCacheKey, result: ExtractionResult) -> None:
        """Merge an extraction result into KG_EXTRACTION_CACHE."""

        self._execute_merge(
            build_extraction_cache_merge_statement(),
            [
                key.id,
                key.graph_id,
                key.chunk_batch_hash,
                key.llm_provider,
                key.model,
                key.options_hash,
                _json(result.model_dump(mode="json")),
            ],
        )

    def _fetch_variant(self, table: str, column: str, cache_id: str) -> object | None:
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT {column} FROM {table} WHERE ID = ?", [cache_id])
            row = cursor.fetchone()
        finally:
            cursor.close()
            connection.close()
        if row is None or len(row) == 0:
            return None
        return _variant_value(row[0])

    def _execute_merge(self, sql: str, params: Sequence[object]) -> None:
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()


def build_ocr_cache_merge_statement() -> str:
    """Return SQL that upserts one OCR cache entry."""

    return (
        "MERGE INTO KG_OCR_CACHE target USING ("
        "SELECT ? AS ID, ? AS FILE_ID, ? AS CHECKSUM, ? AS OCR_PROVIDER, "
        "? AS OPTIONS_HASH, PARSE_JSON(?) AS PARSED_DOCUMENT"
        ") source ON target.ID = source.ID "
        "WHEN MATCHED THEN UPDATE SET FILE_ID = source.FILE_ID, "
        "CHECKSUM = source.CHECKSUM, OCR_PROVIDER = source.OCR_PROVIDER, "
        "OPTIONS_HASH = source.OPTIONS_HASH, PARSED_DOCUMENT = source.PARSED_DOCUMENT, "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHEN NOT MATCHED THEN INSERT "
        "(ID, FILE_ID, CHECKSUM, OCR_PROVIDER, OPTIONS_HASH, PARSED_DOCUMENT) "
        "VALUES (source.ID, source.FILE_ID, source.CHECKSUM, source.OCR_PROVIDER, "
        "source.OPTIONS_HASH, source.PARSED_DOCUMENT)"
    )


def build_extraction_cache_merge_statement() -> str:
    """Return SQL that upserts one extraction cache entry."""

    return (
        "MERGE INTO KG_EXTRACTION_CACHE target USING ("
        "SELECT ? AS ID, ? AS GRAPH_ID, ? AS CHUNK_BATCH_HASH, ? AS LLM_PROVIDER, "
        "? AS MODEL, ? AS OPTIONS_HASH, PARSE_JSON(?) AS RESULT"
        ") source ON target.ID = source.ID "
        "WHEN MATCHED THEN UPDATE SET GRAPH_ID = source.GRAPH_ID, "
        "CHUNK_BATCH_HASH = source.CHUNK_BATCH_HASH, LLM_PROVIDER = source.LLM_PROVIDER, "
        "MODEL = source.MODEL, OPTIONS_HASH = source.OPTIONS_HASH, RESULT = source.RESULT, "
        "UPDATED_AT = CURRENT_TIMESTAMP() "
        "WHEN NOT MATCHED THEN INSERT "
        "(ID, GRAPH_ID, CHUNK_BATCH_HASH, LLM_PROVIDER, MODEL, OPTIONS_HASH, RESULT) "
        "VALUES (source.ID, source.GRAPH_ID, source.CHUNK_BATCH_HASH, source.LLM_PROVIDER, "
        "source.MODEL, source.OPTIONS_HASH, source.RESULT)"
    )


def _variant_value(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)
