from __future__ import annotations

import inspect
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from kg_processor.adapters import snowflake as shared_snowflake
from kg_processor.adapters.jobs import snowflake as jobs_snowflake
from kg_processor.adapters.writers import snowflake_bulk, snowflake_direct, snowflake_manifest
from kg_processor.adapters.writers.snowflake_direct import (
    ColumnSpec,
    SnowflakeConnectionConfig,
    SnowflakeDirectWriter,
    build_edge_reconciliation_statements,
    build_merge_statement,
    build_node_reconciliation_statements,
    build_reindex_delete_statements,
    build_snowflake_rows,
)
from kg_processor.application.snowflake_schema import (
    render_snowflake_schema_sql,
    split_sql_statements,
)
from kg_processor.config.preflight import run_preflight
from kg_processor.config.settings import Settings
from kg_processor.domain.graph import (
    Chunk,
    Community,
    CommunityFinding,
    EntitySource,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphWriteBatch,
)
from kg_processor.domain.ids import stable_id


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, Sequence[object] | None]] = []
        self.closed = False

    def execute(self, sql: str, params: Sequence[object] | None = None) -> object:
        self.executed.append((sql, params))
        return None

    def fetchone(self) -> Sequence[object] | None:
        return None

    def fetchall(self) -> list[object]:
        return []

    def close(self) -> object:
        self.closed = True
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()
        self.committed = False
        self.rolled_back = False
        self.closed = False
        self.autocommit_calls: list[bool] = []

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> object:
        self.committed = True
        return None

    def rollback(self) -> object:
        self.rolled_back = True
        return None

    def autocommit(self, enabled: bool) -> object:
        self.autocommit_calls.append(enabled)
        return None

    def close(self) -> object:
        self.closed = True
        return None


def test_snowflake_schema_uses_configured_vector_dimension() -> None:
    sql = render_snowflake_schema_sql(1536)

    assert "VECTOR(FLOAT, 1536)" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_JOB" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_JOB_FILE" in sql
    assert "WORKER_ID STRING" in sql
    assert "LEASE_UNTIL TIMESTAMP_NTZ" in sql
    assert "ROWS_WRITTEN NUMBER" in sql
    assert "ROW_COUNTS VARIANT" in sql
    assert "AUDIT VARIANT" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_NODE" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_BLOCK" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_ASSET" in sql
    assert "BBOX ARRAY" in sql
    assert "METADATA VARIANT" in sql
    assert "DOCUMENT_ID STRING NOT NULL" in sql
    assert "SECTION_PATH ARRAY" in sql
    assert "BLOCK_IDS ARRAY" in sql
    assert "ASSET_IDS ARRAY" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_GRAPH_METRICS" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_EXTRACTION_TRACE" in sql
    assert "RATING_EXPLANATION STRING" in sql
    assert "SUGGESTED_QUESTIONS ARRAY" in sql
    assert "RUN_ID STRING NOT NULL" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_OCR_CACHE" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_EXTRACTION_CACHE" in sql
    assert "CREATE TABLE IF NOT EXISTS KG_BULK_LOAD" in sql
    assert len(split_sql_statements(sql)) >= 19


def test_snowflake_schema_rejects_invalid_vector_dimension() -> None:
    with pytest.raises(ValueError, match="VECTOR dimension"):
        render_snowflake_schema_sql(5000)


def test_merge_statement_casts_arrays_variants_and_vectors() -> None:
    sql, params = build_merge_statement(
        "KG_NODE",
        {
            "ID": "node_1",
            "GRAPH_ID": "graph",
            "TYPES": ["PERSON"],
            "EMBEDDING": [0.1, 0.2],
            "SOURCE_CHUNK_IDS": ["chunk_1"],
        },
        (
            ColumnSpec("ID"),
            ColumnSpec("GRAPH_ID"),
            ColumnSpec("TYPES", "array"),
            ColumnSpec("EMBEDDING", "vector"),
            ColumnSpec("SOURCE_CHUNK_IDS", "array"),
        ),
        2,
    )

    assert "PARSE_JSON(?)::VECTOR(FLOAT, 2) AS EMBEDDING" in sql
    assert "PARSE_JSON(?)::ARRAY AS TYPES" in sql
    assert params == ["node_1", "graph", '["PERSON"]', "[0.1,0.2]", '["chunk_1"]']


def test_build_snowflake_rows_maps_graph_batch() -> None:
    batch = _sample_batch()

    rows = build_snowflake_rows(batch)

    assert rows["KG_DOCUMENT"][0]["ID"] == stable_id("document", "graph", "file_1")
    other_graph_rows = build_snowflake_rows(batch.model_copy(update={"graph_id": "other_graph"}))
    assert other_graph_rows["KG_DOCUMENT"][0]["ID"] != rows["KG_DOCUMENT"][0]["ID"]
    assert rows["KG_PAGE"][0]["GRAPH_ID"] == "graph"
    assert rows["KG_BLOCK"][0]["ID"] == "block_1"
    assert rows["KG_BLOCK"][0]["BBOX"] == [0.0, 1.0, 2.0, 3.0]
    assert rows["KG_BLOCK"][0]["METADATA"] == {"layout": "body"}
    assert rows["KG_ASSET"][0]["ID"] == "asset_1"
    assert rows["KG_ASSET"][0]["GRAPH_ID"] == "graph"
    assert rows["KG_ASSET"][0]["METADATA"] == {"layout": "figure"}
    assert rows["KG_CHUNK"][0]["DOCUMENT_ID"] == rows["KG_DOCUMENT"][0]["ID"]
    assert rows["KG_CHUNK"][0]["SECTION_PATH"] == ["Intro"]
    assert rows["KG_CHUNK"][0]["BLOCK_IDS"] == ["block_1"]
    assert rows["KG_CHUNK"][0]["ASSET_IDS"] == ["asset_1"]
    assert rows["KG_CHUNK"][0]["OCR_GENERATION_ID"] == "ocr-run-1"
    assert rows["KG_CHUNK"][0]["EMBEDDING"] == [0.1, 0.2]
    assert rows["KG_NODE"][0]["SOURCE_CHUNK_IDS"] == ["chunk_1"]
    assert rows["KG_COMMUNITY"][0]["RATING_EXPLANATION"] == "Important Alice cluster."
    assert rows["KG_COMMUNITY"][0]["SUGGESTED_QUESTIONS"] == ["Who is Alice linked to?"]
    assert rows["KG_COMMUNITY_FINDING"][0]["GRAPH_ID"] == "graph"
    report = rows["KG_RUN_REPORT"][0]["REPORT"]
    assert isinstance(report, dict)
    assert report["job_id"] == "job"
    assert rows["KG_RUN_REPORT"][0]["RUN_ID"] == "run_1"
    assert rows["KG_GRAPH_METRICS"][0]["RUN_ID"] == "run_1"
    metrics = rows["KG_GRAPH_METRICS"][0]["METRICS"]
    assert isinstance(metrics, dict)
    counts = metrics["counts"]
    assert isinstance(counts, dict)
    assert counts["nodes"] == 1
    assert rows["KG_EXTRACTION_TRACE"][0]["STAGE"] == "ocr"
    assert rows["KG_EXTRACTION_TRACE"][0]["RUN_ID"] == "run_1"


def test_snowflake_row_mapping_redacts_sensitive_variant_artifacts() -> None:
    batch = _sample_batch().model_copy(
        update={
            "run_report": {
                "job_id": "job",
                "graph_id": "graph",
                "run_id": "run_1",
                "api_key": "report-secret",
            },
            "graph_metrics": {
                "counts": {"nodes": 1},
                "connection_string": "AccountKey=metric-secret",
            },
            "extraction_trace": [
                {
                    "stage": "llm",
                    "provider_metadata": {
                        "provider": "azure_openai",
                        "oauth_token": "trace-token",
                    },
                }
            ],
        }
    )

    rows = build_snowflake_rows(batch)
    report = cast(dict[str, Any], rows["KG_RUN_REPORT"][0]["REPORT"])
    metrics = cast(dict[str, Any], rows["KG_GRAPH_METRICS"][0]["METRICS"])
    trace_payload = cast(dict[str, Any], rows["KG_EXTRACTION_TRACE"][0]["PAYLOAD"])
    provider_metadata = cast(dict[str, Any], trace_payload["provider_metadata"])

    assert report["api_key"] == "***"
    assert metrics["connection_string"] == "***"
    assert provider_metadata["oauth_token"] == "***"
    assert provider_metadata["provider"] == "azure_openai"


def test_reindex_delete_statements_remove_graph_snapshot_and_current_job_rows() -> None:
    batch = _sample_batch()

    statements = build_reindex_delete_statements(batch)

    assert statements[:3] == [
        ("DELETE FROM KG_EXTRACTION_TRACE WHERE GRAPH_ID = ? AND JOB_ID = ?", ["graph", "job"]),
        ("DELETE FROM KG_RUN_REPORT WHERE GRAPH_ID = ? AND JOB_ID = ?", ["graph", "job"]),
        ("DELETE FROM KG_GRAPH_METRICS WHERE GRAPH_ID = ? AND JOB_ID = ?", ["graph", "job"]),
    ]
    assert ("DELETE FROM KG_NODE WHERE GRAPH_ID = ?", ["graph"]) in statements
    assert ("DELETE FROM KG_BLOCK WHERE GRAPH_ID = ?", ["graph"]) in statements
    assert ("DELETE FROM KG_ASSET WHERE GRAPH_ID = ?", ["graph"]) in statements
    assert ("DELETE FROM KG_DOCUMENT WHERE GRAPH_ID = ?", ["graph"]) in statements


def test_reindex_delete_statements_for_file_batch_are_file_scoped() -> None:
    """Ensure incremental deletion cannot remove artifacts from unrelated source files.

    Run diagnostics remain scoped by run identity.
    """

    batch = _sample_batch().model_copy(
        update={
            "write_scope": "file_batch",
            "reindex_file_ids": ["file_1"],
        }
    )

    statements = build_reindex_delete_statements(batch)

    assert (
        "DELETE FROM KG_EXTRACTION_TRACE WHERE GRAPH_ID = ? AND JOB_ID = ? AND RUN_ID = ?",
        ["graph", "job", "run_1"],
    ) in statements
    assert (
        "DELETE FROM KG_DOCUMENT WHERE GRAPH_ID = ? AND FILE_ID IN (?)",
        ["graph", "file_1"],
    ) in statements
    assert (
        "DELETE FROM KG_BLOCK WHERE GRAPH_ID = ? AND FILE_ID IN (?)",
        ["graph", "file_1"],
    ) in statements
    assert (
        "DELETE FROM KG_ASSET WHERE GRAPH_ID = ? AND FILE_ID IN (?)",
        ["graph", "file_1"],
    ) in statements
    assert (
        "DELETE FROM KG_EDGE_OBSERVATION WHERE GRAPH_ID = ? AND FILE_ID IN (?)",
        ["graph", "file_1"],
    ) in statements
    assert not any(sql == "DELETE FROM KG_NODE WHERE GRAPH_ID = ?" for sql, _ in statements)
    assert ("DELETE FROM KG_COMMUNITY WHERE GRAPH_ID = ?", ["graph"]) in statements
    assert (
        "DELETE FROM KG_COMMUNITY_FINDING WHERE GRAPH_ID = ?",
        ["graph"],
    ) in statements


def test_snowflake_direct_writer_executes_schema_and_merges() -> None:
    batch = _sample_batch()
    connection = FakeConnection()

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    writer = SnowflakeDirectWriter(
        SnowflakeConnectionConfig(
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
        ),
        embedding_dimension=2,
        connector_factory=factory,
    )

    writer.write(batch)

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert connection.committed
    assert connection.autocommit_calls == [False, True]
    assert connection.closed
    assert any(sql.startswith("CREATE TABLE IF NOT EXISTS KG_DOCUMENT") for sql in executed_sql)
    assert any(sql.startswith("MERGE INTO KG_NODE") for sql in executed_sql)
    delete_index = next(
        index for index, sql in enumerate(executed_sql) if sql.startswith("DELETE FROM KG_NODE")
    )
    merge_index = next(
        index for index, sql in enumerate(executed_sql) if sql.startswith("MERGE INTO KG_NODE")
    )
    assert delete_index < merge_index


def test_file_batch_writer_does_not_clobber_shared_node_identity_fields() -> None:
    batch = _sample_batch().model_copy(
        update={"write_scope": "file_batch", "reindex_file_ids": ["file_1"]}
    )
    connection = FakeConnection()

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    SnowflakeDirectWriter(
        SnowflakeConnectionConfig(
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
        ),
        embedding_dimension=2,
        connector_factory=factory,
    ).write(batch)

    node_merge = next(
        sql
        for sql, _params in connection.cursor_instance.executed
        if sql.startswith("MERGE INTO KG_NODE")
    )
    update_clause = node_merge.split("WHEN MATCHED THEN UPDATE SET ", 1)[1].split(
        "WHEN NOT MATCHED", 1
    )[0]
    for field in (
        "NAME",
        "PRIMARY_TYPE",
        "TYPES",
        "ALIASES",
        "DESCRIPTION",
        "SOURCE_CHUNK_IDS",
    ):
        assert f"{field} = source.{field}" not in update_clause


def test_snowflake_writer_preflight_requires_target(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "files": {"input_path": tmp_path},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "writer": {"provider": "snowflake_direct"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("snowflake_direct writer requires" in error for error in result.errors)


def test_edge_reconciliation_rebuilds_canonical_rows_from_observations() -> None:
    """Ensure Snowflake edges are rebuilt from remaining per-file assertions.

    Aggregate files, weights, and evidence counts must be recomputed.
    """

    statements = build_edge_reconciliation_statements(_sample_batch())

    assert len(statements) == 2
    aggregate_sql, aggregate_params = statements[0]
    assert "FROM KG_EDGE_OBSERVATION" in aggregate_sql
    assert "ARRAY_AGG(DISTINCT FILE_ID)" in aggregate_sql
    assert "COUNT(*) AS EVIDENCE_COUNT" in aggregate_sql
    assert aggregate_params == [10.0, "graph"]
    assert "NOT EXISTS" in statements[1][0]


def test_file_batch_node_reconciliation_uses_durable_per_file_support() -> None:
    """Rebuild shared node fields and remove unsupported nodes after a file reindex."""

    batch = _sample_batch().model_copy(update={"write_scope": "file_batch"})

    statements = build_node_reconciliation_statements(batch)

    assert len(statements) == 4
    assert "FROM KG_ENTITY_SOURCE source JOIN KG_EVIDENCE evidence" in statements[0][0]
    assert "SET DEGREE = 0, RANK = 0" in statements[1][0]
    assert "COUNT(DISTINCT EDGE_ID)" in statements[2][0]
    assert "NOT EXISTS" in statements[3][0]
    assert all("graph" in params for _sql, params in statements)


def test_file_batch_node_merge_preserves_fields_rebuilt_after_observation_merge() -> None:
    """Do not clobber shared node aggregates with one batch's partial values."""

    sql, _params = build_merge_statement(
        "KG_NODE",
        {"ID": "node", "GRAPH_ID": "graph", "DESCRIPTION": "partial", "DEGREE": 1},
        (
            ColumnSpec("ID"),
            ColumnSpec("GRAPH_ID"),
            ColumnSpec("DESCRIPTION"),
            ColumnSpec("DEGREE"),
        ),
        2,
        preserve_on_match={"DESCRIPTION", "DEGREE"},
    )

    update_clause = sql.split("WHEN MATCHED THEN UPDATE SET ", 1)[1].split("WHEN NOT MATCHED", 1)[0]
    assert "DESCRIPTION =" not in update_clause
    assert "DEGREE =" not in update_clause


def _sample_batch() -> GraphWriteBatch:
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith works at Acme Corp.",
        start_offset=0,
        end_offset=32,
        token_count=6,
        content_hash="hash",
        section_path=["Intro"],
        block_ids=["block_1"],
        asset_ids=["asset_1"],
        ocr_generation_id="ocr-run-1",
        embedding=[0.1, 0.2],
    )
    node = GraphNode(
        id="node_1",
        graph_id="graph",
        normalized_name="alicesmith",
        name="Alice Smith",
        primary_type="PERSON",
        types=["PERSON"],
        description="Alice Smith is mentioned.",
        embedding=[0.1, 0.2],
        source_chunk_ids=["chunk_1"],
        degree=1,
        rank=1.0,
    )
    edge = GraphEdge(
        id="edge_1",
        graph_id="graph",
        source_node_id="node_1",
        target_node_id="node_2",
        relation_type="works_at",
        description="Alice works at Acme.",
        weight=1.0,
        source_file_id="file_1",
        source_chunk_ids=["chunk_1"],
        embedding=[0.1, 0.2],
    )
    evidence = Evidence(
        id="evidence_1",
        graph_id="graph",
        subject_id="node_1",
        subject_kind="node",
        file_id="file_1",
        chunk_id="chunk_1",
        page_number=1,
        start_offset=0,
        end_offset=32,
        quote="Alice Smith works at Acme Corp.",
    )
    source = EntitySource(
        id="source_1",
        graph_id="graph",
        node_id="node_1",
        file_id="file_1",
        per_file_description="Alice Smith is mentioned.",
        mention_count=1,
    )
    community = Community(
        id="community_1",
        graph_id="graph",
        stable_key="stable",
        level=0,
        title="Alice",
        summary="Alice community",
        rating=5,
        rating_explanation="Important Alice cluster.",
        member_node_ids=["node_1"],
        suggested_questions=["Who is Alice linked to?"],
        embedding=[0.1, 0.2],
    )
    finding = CommunityFinding(
        id="finding_1",
        community_id="community_1",
        summary="Finding",
        explanation="Explanation",
    )
    return GraphWriteBatch(
        graph_id="graph",
        documents=[
            {
                "file_id": "file_1",
                "checksum": "checksum",
                "source_uri": "file:///sample.txt",
                "mime_type": "text/plain",
                "size_bytes": 32,
                "ocr_provider": "builtin_text",
            }
        ],
        pages=[
            {
                "file_id": "file_1",
                "page_number": 1,
                "markdown": "Alice Smith works at Acme Corp.",
                "raw_text": "Alice Smith works at Acme Corp.",
                "detected_language": "en",
            }
        ],
        blocks=[
            {
                "id": "block_1",
                "graph_id": "graph",
                "file_id": "file_1",
                "page_number": 1,
                "kind": "paragraph",
                "text": "Alice Smith works at Acme Corp.",
                "bbox": [0.0, 1.0, 2.0, 3.0],
                "metadata": {"layout": "body"},
            }
        ],
        assets=[
            {
                "id": "asset_1",
                "graph_id": "graph",
                "file_id": "file_1",
                "kind": "image",
                "page_number": 1,
                "uri": "file:///asset.png",
                "metadata": {"layout": "figure"},
            }
        ],
        chunks=[chunk],
        nodes=[node],
        edges=[edge],
        evidence=[evidence],
        entity_sources=[source],
        communities=[community],
        community_findings=[finding],
        run_report={"job_id": "job", "graph_id": "graph", "run_id": "run_1"},
        graph_metrics={"counts": {"nodes": 1}},
        extraction_trace=[{"stage": "ocr", "file_id": "file_1"}],
    )


def test_direct_writer_releases_the_session_when_no_cursor_can_be_opened() -> None:
    """A connection that outlives a failed write holds a Snowflake session open."""

    class _CursorlessConnection(FakeConnection):
        def cursor(self) -> FakeCursor:
            """Fail the way a connector reports an unusable session."""

            raise RuntimeError("session is no longer usable")

    connection = _CursorlessConnection()

    def factory(**_kwargs: object) -> _CursorlessConnection:
        return connection

    writer = SnowflakeDirectWriter(
        SnowflakeConnectionConfig(
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
        ),
        embedding_dimension=2,
        connector_factory=cast(Any, factory),
    )

    with pytest.raises(RuntimeError, match="session is no longer usable"):
        writer.write(_sample_batch())

    assert connection.closed


def test_one_autocommit_helper_serves_every_snowflake_adapter() -> None:
    """Duplicated transaction-mode helpers drift and diverge adapters' commit semantics."""

    consumers = [jobs_snowflake, snowflake_direct, snowflake_bulk, snowflake_manifest]
    definers = [
        module.__name__
        for module in [shared_snowflake, *consumers]
        if re.search(r"^def \w*autocommit\w*\(", inspect.getsource(module), re.MULTILINE)
    ]

    assert definers == [shared_snowflake.__name__]
    for module in consumers:
        assert module.set_snowflake_autocommit is shared_snowflake.set_snowflake_autocommit
