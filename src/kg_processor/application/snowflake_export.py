"""Materialize a Snowflake-resident graph as local inspection artifacts.

Snowflake runs write the graph straight into ``KG_*`` tables, while every
inspection and evaluation tool reads the local parquet snapshot. Without a
bridge, a graph built on Snowflake cannot be compared against a gold fixture or
against a local run, which makes benchmark parity unverifiable on the one
runtime where the pipeline is most likely to be operated.

The export deliberately reads tables rather than reconstructing domain objects.
The Snowflake schema is generated from the same field definitions the artifact
writer uses, so column names already line up, and a row-level copy keeps the
export correct as the graph model evolves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    SnowflakeCursor,
    connect_snowflake,
    validate_snowflake_identifier,
)
from kg_processor.application.snowflake_schema import snowflake_schema_columns
from kg_processor.config.settings import Settings

# Artifact table name -> Snowflake table name. The artifact side is the parquet
# file stem that inspection reads; the Snowflake side is where a run wrote it.
_ARTIFACT_TABLES: dict[str, str] = {
    "documents": "KG_DOCUMENT",
    "pages": "KG_PAGE",
    "blocks": "KG_BLOCK",
    "assets": "KG_ASSET",
    "chunks": "KG_CHUNK",
    "nodes": "KG_NODE",
    "edges": "KG_EDGE",
    "edge_observations": "KG_EDGE_OBSERVATION",
    "evidence": "KG_EVIDENCE",
    "entity_sources": "KG_ENTITY_SOURCE",
    "communities": "KG_COMMUNITY",
    "community_findings": "KG_COMMUNITY_FINDING",
}

# Bookkeeping columns exist in Snowflake but are not part of the graph contract.
_EXCLUDED_COLUMNS = frozenset({"UPDATED_AT", "CREATED_AT"})


def export_snowflake_graph(
    settings: Settings,
    graph_id: str,
    output_path: Path,
    connector_factory: ConnectorFactory | None = None,
) -> dict[str, Any]:
    """Write one graph's Snowflake tables to a local artifact directory.

    Returns per-table row counts so a caller can tell an empty graph apart from
    a graph that was never written, which otherwise look identical on disk.
    """

    import pandas as pd  # noqa: PLC0415 — optional at import time, required here

    if not graph_id:
        raise ValueError("graph export requires a graph id")

    config = SnowflakeConnectionConfig.from_settings(settings.snowflake)
    schema_columns = snowflake_schema_columns()
    counts: dict[str, int] = {}
    frames: dict[str, Any] = {}

    connection = connect_snowflake(config, connector_factory)
    try:
        cursor = connection.cursor()
        try:
            for artifact_name, table in _ARTIFACT_TABLES.items():
                columns = [
                    name
                    for name in schema_columns[table]
                    if name.upper() not in _EXCLUDED_COLUMNS
                ]
                frame = _read_table(cursor, pd, config, table, columns, graph_id)
                frames[artifact_name] = frame
                counts[artifact_name] = len(frame.index)
        finally:
            cursor.close()
    finally:
        connection.close()

    _write_artifacts(output_path, frames)
    return {
        "graph_id": graph_id,
        "output": str(output_path),
        "tables": counts,
        "rows": sum(counts.values()),
    }


def _read_table(
    cursor: SnowflakeCursor,
    pd: Any,
    config: SnowflakeConnectionConfig,
    table: str,
    columns: list[str],
    graph_id: str,
) -> Any:
    """Read one graph-scoped table into a frame using artifact column names.

    Columns are taken from the schema generator rather than the result cursor so
    the projection stays pinned to the declared contract, and a column added to
    Snowflake out of band cannot silently change the artifact shape.

    The database and schema come from configuration and cannot be bound as
    parameters, so they are validated as unquoted identifiers before they reach
    the statement text.
    """

    qualified = ".".join(
        (
            _qualifier(config.database, "database"),
            _qualifier(config.schema_name, "schema"),
            validate_snowflake_identifier(table),
        )
    )
    projection = ", ".join(validate_snowflake_identifier(column) for column in columns)
    cursor.execute(f"SELECT {projection} FROM {qualified} WHERE GRAPH_ID = ?", [graph_id])
    rows = cursor.fetchall()
    frame = pd.DataFrame(list(rows), columns=columns)
    # Inspection reads lower-case artifact columns; Snowflake returns them upper.
    frame.columns = [name.lower() for name in frame.columns]
    return frame


def _qualifier(value: str, label: str) -> str:
    """Normalize and validate one configured name-resolution component.

    Snowflake resolves unquoted identifiers case-insensitively, so a configured
    lower-case name is upper-cased to match the identifier contract rather than
    being rejected.
    """

    try:
        return validate_snowflake_identifier(value.upper())
    except ValueError:
        raise ValueError(
            f"Snowflake {label} must be an unquoted identifier, got: {value}"
        ) from None


def _write_artifacts(output_path: Path, frames: dict[str, Any]) -> None:
    """Publish the snapshot only after every table has been written.

    A partially written directory would be read by inspection as a complete but
    truncated graph, so the whole set is staged first and moved into place.
    """

    import shutil  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    temp_path.mkdir(parents=False, exist_ok=False)
    try:
        for name, frame in frames.items():
            frame.to_parquet(temp_path / f"{name}.parquet", index=False)
        if output_path.exists():
            shutil.rmtree(output_path)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
