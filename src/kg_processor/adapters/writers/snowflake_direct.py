"""Direct Snowflake writer using parameterized row-level MERGE statements.

This writer favors clarity and smoke-testability. The bulk writer reuses the
same row mapping and table column specs for production-sized loads.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    load_snowflake_connector,
)
from kg_processor.application.redaction import redact_sensitive_data
from kg_processor.application.snowflake_schema import (
    render_snowflake_schema_sql,
    split_sql_statements,
)
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.ids import stable_id

_IDENTIFIER_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

__all__ = [
    "ColumnSpec",
    "SnowflakeConnectionConfig",
    "SnowflakeDirectWriter",
    "build_merge_statement",
    "build_reindex_delete_statements",
    "build_snowflake_rows",
]


@dataclass(frozen=True)
class ColumnSpec:
    """Snowflake column name plus the casting behavior required for its values."""

    name: str
    kind: str = "scalar"


TABLE_COLUMNS: dict[str, tuple[ColumnSpec, ...]] = {
    # This table map is the source of truth for both direct MERGE statements and
    # bulk staged loads. Adding a persisted field here gives both writer paths
    # the same Snowflake casting behavior.
    "KG_DOCUMENT": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("CHECKSUM"),
        ColumnSpec("SOURCE_URI"),
        ColumnSpec("MIME_TYPE"),
        ColumnSpec("SIZE_BYTES"),
        ColumnSpec("OCR_PROVIDER"),
    ),
    "KG_PAGE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("MARKDOWN"),
        ColumnSpec("RAW_TEXT"),
        ColumnSpec("DETECTED_LANGUAGE"),
    ),
    "KG_BLOCK": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("KIND"),
        ColumnSpec("TEXT"),
        ColumnSpec("BBOX", "array"),
        ColumnSpec("METADATA", "variant"),
    ),
    "KG_ASSET": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("KIND"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("URI"),
        ColumnSpec("METADATA", "variant"),
    ),
    "KG_CHUNK": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("DOCUMENT_ID"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("CHUNK_INDEX"),
        ColumnSpec("CONTENT"),
        ColumnSpec("START_OFFSET"),
        ColumnSpec("END_OFFSET"),
        ColumnSpec("TOKEN_COUNT"),
        ColumnSpec("CONTENT_HASH"),
        ColumnSpec("SECTION_PATH", "array"),
        ColumnSpec("BLOCK_IDS", "array"),
        ColumnSpec("ASSET_IDS", "array"),
        ColumnSpec("OCR_GENERATION_ID"),
        ColumnSpec("EMBEDDING", "vector"),
    ),
    "KG_NODE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("NORMALIZED_NAME"),
        ColumnSpec("NAME"),
        ColumnSpec("PRIMARY_TYPE"),
        ColumnSpec("TYPES", "array"),
        ColumnSpec("DESCRIPTION"),
        ColumnSpec("EMBEDDING", "vector"),
        ColumnSpec("SOURCE_CHUNK_IDS", "array"),
        ColumnSpec("DEGREE"),
        ColumnSpec("RANK"),
    ),
    "KG_EDGE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("SOURCE_NODE_ID"),
        ColumnSpec("TARGET_NODE_ID"),
        ColumnSpec("RELATION_TYPE"),
        ColumnSpec("DESCRIPTION"),
        ColumnSpec("WEIGHT"),
        ColumnSpec("SOURCE_FILE_ID"),
        ColumnSpec("SOURCE_CHUNK_IDS", "array"),
        ColumnSpec("EMBEDDING", "vector"),
    ),
    "KG_EVIDENCE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("SUBJECT_ID"),
        ColumnSpec("SUBJECT_KIND"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("CHUNK_ID"),
        ColumnSpec("PAGE_NUMBER"),
        ColumnSpec("START_OFFSET"),
        ColumnSpec("END_OFFSET"),
        ColumnSpec("QUOTE"),
    ),
    "KG_ENTITY_SOURCE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("NODE_ID"),
        ColumnSpec("FILE_ID"),
        ColumnSpec("PER_FILE_DESCRIPTION"),
        ColumnSpec("MENTION_COUNT"),
    ),
    "KG_COMMUNITY": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("STABLE_KEY"),
        ColumnSpec("LEVEL"),
        ColumnSpec("TITLE"),
        ColumnSpec("SUMMARY"),
        ColumnSpec("RATING"),
        ColumnSpec("RATING_EXPLANATION"),
        ColumnSpec("MEMBER_NODE_IDS", "array"),
        ColumnSpec("SUGGESTED_QUESTIONS", "array"),
        ColumnSpec("EMBEDDING", "vector"),
    ),
    "KG_COMMUNITY_FINDING": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("COMMUNITY_ID"),
        ColumnSpec("SUMMARY"),
        ColumnSpec("EXPLANATION"),
    ),
    "KG_RUN_REPORT": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("JOB_ID"),
        ColumnSpec("RUN_ID"),
        ColumnSpec("REPORT", "variant"),
    ),
    "KG_GRAPH_METRICS": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("JOB_ID"),
        ColumnSpec("RUN_ID"),
        ColumnSpec("METRICS", "variant"),
    ),
    "KG_EXTRACTION_TRACE": (
        ColumnSpec("ID"),
        ColumnSpec("GRAPH_ID"),
        ColumnSpec("JOB_ID"),
        ColumnSpec("RUN_ID"),
        ColumnSpec("STEP_INDEX"),
        ColumnSpec("STAGE"),
        ColumnSpec("PAYLOAD", "variant"),
    ),
}

_GRAPH_SNAPSHOT_DELETE_ORDER = (
    "KG_EXTRACTION_TRACE",
    "KG_RUN_REPORT",
    "KG_GRAPH_METRICS",
    "KG_COMMUNITY_FINDING",
    "KG_COMMUNITY",
    "KG_ENTITY_SOURCE",
    "KG_EVIDENCE",
    "KG_EDGE",
    "KG_NODE",
    "KG_CHUNK",
    "KG_BLOCK",
    "KG_ASSET",
    "KG_PAGE",
    "KG_DOCUMENT",
)

_JOB_SCOPED_TABLES = {
    "KG_EXTRACTION_TRACE",
    "KG_RUN_REPORT",
    "KG_GRAPH_METRICS",
}

_FILE_BATCH_DELETE_ORDER = (
    ("KG_ENTITY_SOURCE", "FILE_ID"),
    ("KG_EVIDENCE", "FILE_ID"),
    ("KG_EDGE", "SOURCE_FILE_ID"),
    ("KG_CHUNK", "FILE_ID"),
    ("KG_BLOCK", "FILE_ID"),
    ("KG_ASSET", "FILE_ID"),
    ("KG_PAGE", "FILE_ID"),
    ("KG_DOCUMENT", "FILE_ID"),
)


class SnowflakeDirectWriter:
    """Writes graph batches with parameterized row-level Snowflake MERGE statements."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        embedding_dimension: int,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        """Store connector dependencies and the vector dimension for casts."""

        self.config = config
        self.embedding_dimension = embedding_dimension
        self.connector_factory = connector_factory or load_snowflake_connector()

    def write(self, batch: GraphWriteBatch) -> None:
        """Create schema, delete stale rows, and merge each graph artifact row."""

        connection = self.connector_factory(**self.config.connect_kwargs())
        cursor = connection.cursor()
        try:
            for statement in split_sql_statements(
                render_snowflake_schema_sql(self.embedding_dimension)
            ):
                cursor.execute(statement)
            for sql, params in build_reindex_delete_statements(batch):
                cursor.execute(sql, params)
            for table_name, rows in build_snowflake_rows(batch).items():
                for row in rows:
                    sql, params = build_merge_statement(
                        table_name,
                        row,
                        TABLE_COLUMNS[table_name],
                        self.embedding_dimension,
                    )
                    cursor.execute(sql, params)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()


def build_snowflake_rows(batch: GraphWriteBatch) -> dict[str, list[dict[str, object]]]:
    """Map a graph write batch into upper-case Snowflake table rows."""

    job_id = str(batch.run_report.get("job_id", ""))
    run_id = _run_id(batch)
    run_report = cast(dict[str, Any], redact_sensitive_data(batch.run_report))
    graph_metrics = cast(dict[str, Any], redact_sensitive_data(batch.graph_metrics))
    extraction_trace = cast(list[dict[str, Any]], redact_sensitive_data(batch.extraction_trace))
    # Domain models use Pythonic lower-case fields. Snowflake rows are upper-case
    # to match generated DDL and avoid quoted identifier surprises.
    return {
        "KG_DOCUMENT": [_document_row(batch.graph_id, row) for row in batch.documents],
        "KG_PAGE": [_page_row(batch.graph_id, row) for row in batch.pages],
        "KG_BLOCK": [_block_row(batch.graph_id, row) for row in batch.blocks],
        "KG_ASSET": [_asset_row(batch.graph_id, row) for row in batch.assets],
        "KG_CHUNK": [
            _upper_model(row.model_dump(mode="json"), batch.graph_id) for row in batch.chunks
        ],
        "KG_NODE": [_upper_model(row.model_dump(mode="json")) for row in batch.nodes],
        "KG_EDGE": [_upper_model(row.model_dump(mode="json")) for row in batch.edges],
        "KG_EVIDENCE": [_upper_model(row.model_dump(mode="json")) for row in batch.evidence],
        "KG_ENTITY_SOURCE": [
            _upper_model(row.model_dump(mode="json")) for row in batch.entity_sources
        ],
        "KG_COMMUNITY": [_upper_model(row.model_dump(mode="json")) for row in batch.communities],
        "KG_COMMUNITY_FINDING": [
            {**_upper_model(row.model_dump(mode="json")), "GRAPH_ID": batch.graph_id}
            for row in batch.community_findings
        ],
        "KG_RUN_REPORT": [
            {
                "ID": stable_id("run_report", batch.graph_id, job_id, run_id),
                "GRAPH_ID": batch.graph_id,
                "JOB_ID": job_id,
                "RUN_ID": run_id,
                "REPORT": run_report,
            }
        ],
        "KG_GRAPH_METRICS": [
            {
                "ID": stable_id("graph_metrics", batch.graph_id, job_id, run_id),
                "GRAPH_ID": batch.graph_id,
                "JOB_ID": job_id,
                "RUN_ID": run_id,
                "METRICS": graph_metrics,
            }
        ],
        "KG_EXTRACTION_TRACE": [
            {
                "ID": stable_id(
                    "trace",
                    batch.graph_id,
                    job_id,
                    run_id,
                    index,
                    row.get("stage", ""),
                ),
                "GRAPH_ID": batch.graph_id,
                "JOB_ID": job_id,
                "RUN_ID": run_id,
                "STEP_INDEX": index,
                "STAGE": str(row.get("stage", "")),
                "PAYLOAD": row,
            }
            for index, row in enumerate(extraction_trace)
        ],
    }


def build_reindex_delete_statements(batch: GraphWriteBatch) -> list[tuple[str, list[object]]]:
    """Build delete statements for snapshot rewrites or per-file reindexing."""

    if batch.write_scope == "file_batch":
        return _file_batch_delete_statements(batch)
    job_id = str(batch.run_report.get("job_id", ""))
    statements: list[tuple[str, list[object]]] = []
    for table_name in _GRAPH_SNAPSHOT_DELETE_ORDER:
        table = _validate_identifier(table_name)
        if table_name in _JOB_SCOPED_TABLES:
            statements.append(
                (
                    f"DELETE FROM {table} WHERE GRAPH_ID = ? AND JOB_ID = ?",
                    [batch.graph_id, job_id],
                )
            )
        else:
            statements.append((f"DELETE FROM {table} WHERE GRAPH_ID = ?", [batch.graph_id]))
    return statements


def _file_batch_delete_statements(batch: GraphWriteBatch) -> list[tuple[str, list[object]]]:
    job_id = str(batch.run_report.get("job_id", ""))
    run_id = _run_id(batch)
    file_ids = [file_id for file_id in batch.reindex_file_ids if file_id.strip()]
    statements: list[tuple[str, list[object]]] = [
        (
            "DELETE FROM KG_EXTRACTION_TRACE WHERE GRAPH_ID = ? AND JOB_ID = ? AND RUN_ID = ?",
            [batch.graph_id, job_id, run_id],
        ),
        (
            "DELETE FROM KG_RUN_REPORT WHERE GRAPH_ID = ? AND JOB_ID = ? AND RUN_ID = ?",
            [batch.graph_id, job_id, run_id],
        ),
        (
            "DELETE FROM KG_GRAPH_METRICS WHERE GRAPH_ID = ? AND JOB_ID = ? AND RUN_ID = ?",
            [batch.graph_id, job_id, run_id],
        ),
    ]
    if not file_ids:
        return statements
    for table_name, file_column in _FILE_BATCH_DELETE_ORDER:
        table = _validate_identifier(table_name)
        column = _validate_identifier(file_column)
        placeholders = ", ".join("?" for _ in file_ids)
        statements.append(
            (
                f"DELETE FROM {table} WHERE GRAPH_ID = ? AND {column} IN ({placeholders})",
                [batch.graph_id, *file_ids],
            )
        )
    return statements


def build_merge_statement(
    table_name: str,
    row: Mapping[str, object],
    columns: Sequence[ColumnSpec],
    embedding_dimension: int,
) -> tuple[str, list[object]]:
    """Build a parameterized MERGE statement for a single Snowflake row."""

    table = _validate_identifier(table_name)
    select_parts: list[str] = []
    params: list[object] = []
    for column in columns:
        value = row.get(column.name)
        expression, expression_params = _select_expression(column, value, embedding_dimension)
        select_parts.append(f"{expression} AS {column.name}")
        params.extend(expression_params)

    source_sql = "SELECT " + ", ".join(select_parts)
    update_columns = [column.name for column in columns if column.name != "ID"]
    update_sql = ", ".join(f"{name} = source.{name}" for name in update_columns)
    column_names = ", ".join(column.name for column in columns)
    value_names = ", ".join(f"source.{column.name}" for column in columns)
    sql = (
        f"MERGE INTO {table} target USING ({source_sql}) source ON target.ID = source.ID "
        f"WHEN MATCHED THEN UPDATE SET {update_sql}, UPDATED_AT = CURRENT_TIMESTAMP() "
        f"WHEN NOT MATCHED THEN INSERT ({column_names}) VALUES ({value_names})"
    )
    return sql, params


def _select_expression(
    column: ColumnSpec,
    value: object,
    embedding_dimension: int,
) -> tuple[str, list[object]]:
    if value is None:
        return "NULL", []
    if column.kind == "vector":
        return f"PARSE_JSON(?)::VECTOR(FLOAT, {embedding_dimension})", [_json(value)]
    if column.kind == "array":
        return "PARSE_JSON(?)::ARRAY", [_json(value)]
    if column.kind == "variant":
        return "PARSE_JSON(?)", [_json(value)]
    return "?", [value]


def _document_row(graph_id: str, row: Mapping[str, object]) -> dict[str, object]:
    return {
        "ID": str(row["file_id"]),
        "GRAPH_ID": graph_id,
        "FILE_ID": str(row["file_id"]),
        "CHECKSUM": str(row["checksum"]),
        "SOURCE_URI": str(row["source_uri"]),
        "MIME_TYPE": str(row["mime_type"]),
        "SIZE_BYTES": row.get("size_bytes"),
        "OCR_PROVIDER": row.get("ocr_provider"),
    }


def _page_row(graph_id: str, row: Mapping[str, object]) -> dict[str, object]:
    file_id = str(row["file_id"])
    raw_page_number = row["page_number"]
    if not isinstance(raw_page_number, int | str):
        raise ValueError("page_number must be an integer-compatible value")
    page_number = int(raw_page_number)
    return {
        "ID": stable_id("page", graph_id, file_id, page_number),
        "GRAPH_ID": graph_id,
        "FILE_ID": file_id,
        "PAGE_NUMBER": page_number,
        "MARKDOWN": row.get("markdown"),
        "RAW_TEXT": row.get("raw_text"),
        "DETECTED_LANGUAGE": row.get("detected_language"),
    }


def _asset_row(graph_id: str, row: Mapping[str, object]) -> dict[str, object]:
    return {
        "ID": str(row["id"]),
        "GRAPH_ID": graph_id,
        "FILE_ID": str(row["file_id"]),
        "KIND": str(row["kind"]),
        "PAGE_NUMBER": row.get("page_number"),
        "URI": row.get("uri"),
        "METADATA": row.get("metadata") or {},
    }


def _block_row(graph_id: str, row: Mapping[str, object]) -> dict[str, object]:
    return {
        "ID": str(row["id"]),
        "GRAPH_ID": graph_id,
        "FILE_ID": str(row["file_id"]),
        "PAGE_NUMBER": row.get("page_number"),
        "KIND": str(row["kind"]),
        "TEXT": row.get("text"),
        "BBOX": row.get("bbox"),
        "METADATA": row.get("metadata") or {},
    }


def _upper_model(row: Mapping[str, object], graph_id: str | None = None) -> dict[str, object]:
    converted = {_camel_to_screaming(key): value for key, value in row.items()}
    if graph_id is not None:
        converted["GRAPH_ID"] = graph_id
    return converted


def _camel_to_screaming(value: str) -> str:
    chars: list[str] = []
    for char in value:
        if char.isupper() and chars:
            chars.append("_")
        chars.append(char.upper())
    return "".join(chars)


def _validate_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Unsafe Snowflake identifier: {value}")
    return value


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _run_id(batch: GraphWriteBatch) -> str:
    value = batch.run_report.get("run_id")
    if isinstance(value, str) and value.strip():
        return value
    return stable_id(
        "run",
        batch.graph_id,
        batch.run_report.get("job_id", ""),
        batch.write_scope,
        batch.reindex_file_ids,
    )
