"""Graph artifact readers shared by local and remote explorer backends."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from flakegraph_app.models import GraphDataset

# Tables whose rows a review surface renders. Documents and chunks are counted
# for the summary metrics but never displayed, and chunk rows carry the full text
# of the corpus — on one real graph they were 70% of everything transferred to
# open it — so they are counted rather than fetched.
GRAPH_REVIEW_ROW_LIMITS = {
    "KG_NODE": 2_500,
    "KG_EDGE": 5_000,
    "KG_COMMUNITY": 1_000,
    "KG_EVIDENCE": 5_000,
}
GRAPH_COUNTED_TABLES = ("KG_DOCUMENT", "KG_CHUNK")
# Every table that records something about one graph. Removing a graph has to
# clear all of them, or rows survive a deletion with nothing in the application
# still pointing at them. Written out rather than derived because the deployed
# Streamlit bundle ships only this package and cannot import the schema module;
# a test pins this list to that schema so a table added later cannot be missed.
GRAPH_DATA_TABLES: tuple[str, ...] = (
    "KG_ASSET",
    "KG_BLOCK",
    "KG_BULK_LOAD",
    "KG_CHUNK",
    "KG_COMMUNITY",
    "KG_COMMUNITY_FINDING",
    "KG_DOCUMENT",
    "KG_EDGE",
    "KG_EDGE_OBSERVATION",
    "KG_ENRICHMENT_CACHE",
    "KG_ENTITY_SOURCE",
    "KG_EVIDENCE",
    "KG_EXTRACTION_CACHE",
    "KG_EXTRACTION_TRACE",
    "KG_GRAPH",
    "KG_GRAPH_ACL",
    "KG_GRAPH_METRICS",
    "KG_JOB",
    "KG_JOB_FILE",
    "KG_NODE",
    "KG_PAGE",
    "KG_RUN_REPORT",
)
EMBEDDING_TABLES = frozenset({"KG_NODE", "KG_EDGE", "KG_CHUNK", "KG_COMMUNITY"})
# A local run reads these metrics from graph_metrics.json; a Snowflake run has
# them in a table. Two Snowflake readers exist — Snowpark inside the deployed app
# and the connector outside it — so the statement is shared rather than written
# twice. Omitting it made the consumption view report every Snowflake graph as
# having no recorded usage while that usage sat in the table.
GRAPH_METRICS_QUERY = (
    "SELECT METRICS FROM KG_GRAPH_METRICS WHERE GRAPH_ID = ? ORDER BY UPDATED_AT DESC LIMIT 1"
)


def review_projection(table: str, *, alias: str | None = None) -> str:
    """Return a bounded-review projection without requesting absent columns."""

    qualifier = f"{alias}." if alias else ""
    projection = f"{qualifier}*"
    return f"{projection} EXCLUDE EMBEDDING" if table in EMBEDDING_TABLES else projection


def load_local_graph(path: Path) -> GraphDataset:
    """Load bounded review rows while retaining full local artifact counts."""

    if not (path / "nodes.parquet").exists() or not (path / "edges.parquet").exists():
        raise ValueError(f"Graph directory must contain nodes.parquet and edges.parquet: {path}")
    run_report = _read_json(path / "run_report.json")
    nodes, node_count = _read_parquet(
        path / "nodes.parquet",
        GRAPH_REVIEW_ROW_LIMITS["KG_NODE"],
    )
    node_ids = {str(node.get("id")) for node in nodes}
    edges, edge_count = _read_parquet(
        path / "edges.parquet",
        GRAPH_REVIEW_ROW_LIMITS["KG_EDGE"],
        node_ids=node_ids,
    )
    communities, community_count = _read_parquet(
        path / "communities.parquet",
        GRAPH_REVIEW_ROW_LIMITS["KG_COMMUNITY"],
    )
    evidence, evidence_count = _read_parquet(
        path / "evidence.parquet",
        GRAPH_REVIEW_ROW_LIMITS["KG_EVIDENCE"],
    )
    document_count = _count_parquet(path / "documents.parquet")
    chunk_count = _count_parquet(path / "chunks.parquet")
    run_report["app_full_counts"] = {
        "node": node_count,
        "edge": edge_count,
        "community": community_count,
        "evidence": evidence_count,
        "document": document_count,
        "chunk": chunk_count,
    }
    graph_id = str(run_report.get("graph_id") or (nodes[0].get("graph_id") if nodes else path.name))
    return GraphDataset(
        graph_id=graph_id,
        nodes=nodes,
        edges=edges,
        communities=communities,
        evidence=evidence,
        run_report=run_report,
        graph_metrics=_read_json(path / "graph_metrics.json"),
    )


def load_snowflake_graph(config_path: Path, graph_id: str) -> GraphDataset:
    """Load a bounded explorer projection using a run's remote Snowflake target."""

    # The Snowflake-hosted app imports this module on Python 3.11. Keep the core
    # processing package, which targets a newer worker runtime, out of module import.
    from kg_processor.adapters.snowflake import (  # noqa: PLC0415
        SnowflakeConnectionConfig,
        load_snowflake_connector,
    )
    from kg_processor.config.settings import Settings  # noqa: PLC0415

    settings = Settings.load(config_path)
    config = SnowflakeConnectionConfig.from_settings(settings.snowflake)
    connection = load_snowflake_connector()(**config.connect_kwargs())
    cursor: Any = connection.cursor()
    try:
        tables: dict[str, list[dict[str, Any]]] = {}
        full_counts: dict[str, int] = {}
        cursor.execute(
            graph_count_query(GRAPH_REVIEW_ROW_LIMITS),
            [graph_id] * len(GRAPH_REVIEW_ROW_LIMITS),
        )
        full_counts = {
            str(table).removeprefix("KG_").lower(): int(count) for table, count in cursor.fetchall()
        }
        for table, limit in GRAPH_REVIEW_ROW_LIMITS.items():
            if table == "KG_EDGE":
                continue
            order = "DEGREE DESC NULLS LAST, ID" if table == "KG_NODE" else "ID"
            cursor.execute(
                f"SELECT {review_projection(table)} FROM {table} WHERE GRAPH_ID = ? "
                f"ORDER BY {order} LIMIT {limit}",
                [graph_id],
            )
            columns = [str(item[0]) for item in cursor.description]
            tables[table] = [
                normalize_row(dict(zip(columns, row, strict=True))) for row in cursor.fetchall()
            ]
        cursor.execute(
            connector_edge_projection_query(
                GRAPH_REVIEW_ROW_LIMITS["KG_NODE"],
                GRAPH_REVIEW_ROW_LIMITS["KG_EDGE"],
            ),
            [graph_id, graph_id],
        )
        edge_columns = [str(item[0]) for item in cursor.description]
        edges = [
            normalize_row(dict(zip(edge_columns, row, strict=True))) for row in cursor.fetchall()
        ]
        cursor.execute(
            "SELECT REPORT FROM KG_RUN_REPORT WHERE GRAPH_ID = ? ORDER BY UPDATED_AT DESC LIMIT 1",
            [graph_id],
        )
        report_row = cursor.fetchone()
        report = _variant_object(report_row[0]) if report_row else {}
        report["app_full_counts"] = full_counts
        cursor.execute(GRAPH_METRICS_QUERY, [graph_id])
        metrics_row = cursor.fetchone()
        return GraphDataset(
            graph_id=graph_id,
            nodes=tables["KG_NODE"],
            edges=edges,
            communities=tables["KG_COMMUNITY"],
            evidence=tables["KG_EVIDENCE"],
            run_report=report,
            graph_metrics=_variant_object(metrics_row[0]) if metrics_row else {},
        )
    finally:
        cursor.close()
        connection.close()


def normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert database and dataframe scalar values into portable Python data."""

    return {
        str(key).lower(): _json_value(value)
        for key, value in row.items()
        if str(key).lower() != "embedding"
    }


def graph_count_query(tables: Mapping[str, int]) -> str:
    """Build one constant-table UNION query for full explorer summary counts."""

    return " UNION ALL ".join(
        f"SELECT '{table}' AS TABLE_NAME, COUNT(*) AS ROW_COUNT FROM {table} WHERE GRAPH_ID = ?"
        for table in tables
    )


def connector_edge_projection_query(node_limit: int, edge_limit: int) -> str:
    """Select only edges whose endpoints belong to the deterministic node sample."""

    return (
        "WITH SELECTED_NODES AS ("
        "SELECT ID FROM KG_NODE WHERE GRAPH_ID = ? "
        f"ORDER BY DEGREE DESC NULLS LAST, ID LIMIT {node_limit}"
        ") "
        "SELECT EDGE.* EXCLUDE EMBEDDING FROM KG_EDGE AS EDGE WHERE EDGE.GRAPH_ID = ? "
        "AND EDGE.SOURCE_NODE_ID IN (SELECT ID FROM SELECTED_NODES) "
        "AND EDGE.TARGET_NODE_ID IN (SELECT ID FROM SELECTED_NODES) "
        f"ORDER BY EDGE.ID LIMIT {edge_limit}"
    )


def _count_parquet(path: Path) -> int:
    """Return a Parquet table's row count without reading any of its rows."""

    if not path.exists():
        return 0
    import pyarrow.dataset as ds  # noqa: PLC0415

    return int(ds.dataset(path, format="parquet").count_rows())  # type: ignore[no-untyped-call]


def _read_parquet(
    path: Path,
    limit: int,
    *,
    node_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return bounded rows and the full Parquet count without loading the table."""

    if not path.exists():
        return [], 0
    import pyarrow.dataset as ds  # noqa: PLC0415

    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    full_count = dataset.count_rows()
    columns = [name for name in dataset.schema.names if name != "embedding"]
    predicate = None
    if node_ids is not None:
        if not node_ids or full_count == 0:
            return [], full_count
        source_field = ds.field("source_node_id")  # type: ignore[attr-defined,no-untyped-call]
        target_field = ds.field("target_node_id")  # type: ignore[attr-defined,no-untyped-call]
        predicate = source_field.isin(node_ids) & target_field.isin(node_ids)
    table = dataset.scanner(
        columns=columns,
        filter=predicate,
        batch_size=max(1, min(limit, 1_024)),
    ).head(limit)
    return (
        [
            normalize_row({str(key): value for key, value in row.items()})
            for row in table.to_pylist()
        ],
        full_count,
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read an optional JSON object while rejecting unexpected root shapes."""

    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return _json_value(value) if isinstance(value, dict) else {}


def _variant_object(value: object) -> dict[str, Any]:
    """Normalize a Snowflake VARIANT report into the explorer's mapping contract."""

    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        parsed = json.loads(value)
        return {str(key): item for key, item in parsed.items()} if isinstance(parsed, dict) else {}
    return {}


def variant_sequence(value: Any) -> list[Any]:
    """Return a list for a value that may arrive as a JSON-encoded array.

    Local artifacts store these columns as real lists, while Snowflake returns
    its ARRAY columns as JSON text. Consumers that assume one shape silently
    misreport the other: ``len()`` counts characters instead of members, and an
    isinstance check drops every row.
    """

    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return list(parsed) if isinstance(parsed, list | tuple) else []
    return []


def _json_value(value: Any) -> Any:  # noqa: PLR0911 - each portable value shape exits clearly.
    """Normalize common Pandas, NumPy, temporal, and nested values recursively."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return _json_value(scalar)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist())
    return str(value)
