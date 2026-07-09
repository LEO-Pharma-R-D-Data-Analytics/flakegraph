from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
from kg_processor.adapters.writers.snowflake_bulk import (
    SnowflakeBulkWriter,
    build_bulk_merge_statement,
    build_copy_into_load_table_statement,
    build_create_load_table_statement,
    build_load_table_name,
    build_put_statement,
    write_bulk_load_files,
)
from kg_processor.adapters.writers.snowflake_direct import (
    TABLE_COLUMNS,
    build_snowflake_rows,
)
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


def test_bulk_load_files_write_string_parquet_rows(tmp_path: Path) -> None:
    rows_by_table = build_snowflake_rows(_sample_batch())

    load_files = write_bulk_load_files(
        rows_by_table,
        tmp_path,
        "@DB.SCHEMA.KG_LOAD_STAGE",
        "kg_processor/graph/job/LOAD123",
        "LOAD123",
    )

    node_file = next(item for item in load_files if item.table_name == "KG_NODE")
    frame = pd.read_parquet(node_file.local_path)
    assert node_file.load_table_name == "KG_LOAD_LOAD123_KG_NODE"
    assert node_file.stage_file_location == (
        "@DB.SCHEMA.KG_LOAD_STAGE/kg_processor/graph/job/LOAD123/KG_NODE.parquet"
    )
    assert frame.loc[0, "TYPES"] == '["PERSON"]'
    assert frame.loc[0, "EMBEDDING"] == "[0.1,0.2]"
    assert frame.loc[0, "DEGREE"] == "1"
    block_file = next(item for item in load_files if item.table_name == "KG_BLOCK")
    block_frame = pd.read_parquet(block_file.local_path)
    assert block_frame.loc[0, "BBOX"] == "[0.0,1.0,2.0,3.0]"
    assert block_frame.loc[0, "METADATA"] == '{"layout":"body"}'
    asset_file = next(item for item in load_files if item.table_name == "KG_ASSET")
    asset_frame = pd.read_parquet(asset_file.local_path)
    assert asset_frame.loc[0, "METADATA"] == '{"layout":"figure"}'
    chunk_file = next(item for item in load_files if item.table_name == "KG_CHUNK")
    chunk_frame = pd.read_parquet(chunk_file.local_path)
    assert chunk_frame.loc[0, "SECTION_PATH"] == '["Intro"]'
    assert chunk_frame.loc[0, "BLOCK_IDS"] == '["block_1"]'
    assert chunk_frame.loc[0, "ASSET_IDS"] == '["asset_1"]'
    community_file = next(item for item in load_files if item.table_name == "KG_COMMUNITY")
    community_frame = pd.read_parquet(community_file.local_path)
    assert community_frame.loc[0, "RATING_EXPLANATION"] == "Important Alice cluster."
    assert community_frame.loc[0, "SUGGESTED_QUESTIONS"] == '["Who is Alice linked to?"]'


def test_bulk_load_files_split_large_tables_by_target_size(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {
            "ID": f"node_{index}",
            "GRAPH_ID": "graph",
            "NORMALIZED_NAME": f"node_{index}",
            "NAME": f"Node {index}",
            "PRIMARY_TYPE": "CONCEPT",
            "TYPES": ["CONCEPT"],
            "DESCRIPTION": "large staged payload " * 10,
        }
        for index in range(5)
    ]

    load_files = write_bulk_load_files(
        {"KG_NODE": rows},
        tmp_path,
        "@DB.SCHEMA.KG_LOAD_STAGE",
        "kg_processor/graph/job/LOAD123",
        "LOAD123",
        target_file_size_bytes=350,
    )

    assert len(load_files) > 1
    assert {load_file.load_table_name for load_file in load_files} == {"KG_LOAD_LOAD123_KG_NODE"}
    assert sum(load_file.row_count for load_file in load_files) == len(rows)
    assert load_files[0].local_path.name == "KG_NODE_00001.parquet"
    assert load_files[-1].stage_file_location.endswith(f"/{load_files[-1].local_path.name}")


def test_bulk_sql_builders_cast_from_staging_columns(tmp_path: Path) -> None:
    put_path = tmp_path / "KG_NODE.parquet"
    put_path.write_text("placeholder", encoding="utf-8")

    assert build_load_table_name("KG_NODE", "load_1") == "KG_LOAD_LOAD_1_KG_NODE"
    assert build_create_load_table_statement(
        "KG_LOAD_LOAD_1_KG_NODE",
        TABLE_COLUMNS["KG_NODE"],
    ).startswith("CREATE TEMP TABLE KG_LOAD_LOAD_1_KG_NODE")
    assert build_put_statement(
        put_path,
        "@DB.SCHEMA.KG_LOAD_STAGE/prefix",
    ).startswith("PUT 'file://")
    assert build_copy_into_load_table_statement(
        "KG_LOAD_LOAD_1_KG_NODE",
        "@DB.SCHEMA.KG_LOAD_STAGE/prefix/KG_NODE.parquet",
    ) == (
        "COPY INTO KG_LOAD_LOAD_1_KG_NODE "
        "FROM @DB.SCHEMA.KG_LOAD_STAGE/prefix/KG_NODE.parquet "
        "FILE_FORMAT = (TYPE = PARQUET USE_LOGICAL_TYPE = TRUE) "
        "MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE"
    )
    merge_sql = build_bulk_merge_statement(
        "KG_NODE",
        "KG_LOAD_LOAD_1_KG_NODE",
        TABLE_COLUMNS["KG_NODE"],
        2,
    )
    assert "PARSE_JSON(EMBEDDING)::VECTOR(FLOAT, 2) AS EMBEDDING" in merge_sql
    assert "PARSE_JSON(TYPES)::ARRAY AS TYPES" in merge_sql
    assert "DEGREE::NUMBER AS DEGREE" in merge_sql
    block_merge_sql = build_bulk_merge_statement(
        "KG_BLOCK",
        "KG_LOAD_LOAD_1_KG_BLOCK",
        TABLE_COLUMNS["KG_BLOCK"],
        2,
    )
    assert "PAGE_NUMBER::NUMBER AS PAGE_NUMBER" in block_merge_sql
    assert "PARSE_JSON(BBOX)::ARRAY AS BBOX" in block_merge_sql
    assert "PARSE_JSON(METADATA) AS METADATA" in block_merge_sql
    asset_merge_sql = build_bulk_merge_statement(
        "KG_ASSET",
        "KG_LOAD_LOAD_1_KG_ASSET",
        TABLE_COLUMNS["KG_ASSET"],
        2,
    )
    assert "PAGE_NUMBER::NUMBER AS PAGE_NUMBER" in asset_merge_sql
    assert "PARSE_JSON(METADATA) AS METADATA" in asset_merge_sql
    chunk_merge_sql = build_bulk_merge_statement(
        "KG_CHUNK",
        "KG_LOAD_LOAD_1_KG_CHUNK",
        TABLE_COLUMNS["KG_CHUNK"],
        2,
    )
    assert "PARSE_JSON(SECTION_PATH)::ARRAY AS SECTION_PATH" in chunk_merge_sql
    assert "PARSE_JSON(BLOCK_IDS)::ARRAY AS BLOCK_IDS" in chunk_merge_sql
    assert "PARSE_JSON(ASSET_IDS)::ARRAY AS ASSET_IDS" in chunk_merge_sql
    community_merge_sql = build_bulk_merge_statement(
        "KG_COMMUNITY",
        "KG_LOAD_LOAD_1_KG_COMMUNITY",
        TABLE_COLUMNS["KG_COMMUNITY"],
        2,
    )
    assert "PARSE_JSON(SUGGESTED_QUESTIONS)::ARRAY AS SUGGESTED_QUESTIONS" in (community_merge_sql)


def test_snowflake_bulk_writer_executes_put_copy_and_merge(tmp_path: Path) -> None:
    batch = _sample_batch()
    connection = FakeConnection()

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    writer = SnowflakeBulkWriter(
        _config(),
        embedding_dimension=2,
        bulk_stage="@DB.SCHEMA.KG_LOAD_STAGE",
        connector_factory=factory,
        local_temp_dir=tmp_path,
        load_id="LOAD123",
    )

    writer.write(batch)

    executed_sql = [sql for sql, _params in connection.cursor_instance.executed]
    assert connection.committed
    assert connection.closed
    assert any(sql.startswith("CREATE TABLE IF NOT EXISTS KG_DOCUMENT") for sql in executed_sql)
    assert any(sql.startswith("CREATE TEMP TABLE KG_LOAD_LOAD123_KG_BLOCK") for sql in executed_sql)
    assert any(sql.startswith("CREATE TEMP TABLE KG_LOAD_LOAD123_KG_ASSET") for sql in executed_sql)
    assert any(sql.startswith("CREATE TEMP TABLE KG_LOAD_LOAD123_KG_NODE") for sql in executed_sql)
    assert any(sql.startswith("PUT 'file://") for sql in executed_sql)
    assert any(sql.startswith("COPY INTO KG_LOAD_LOAD123_KG_NODE") for sql in executed_sql)
    assert any(sql.startswith("MERGE INTO KG_NODE") for sql in executed_sql)
    assert sum(1 for sql in executed_sql if sql.startswith("MERGE INTO KG_NODE")) == 1
    delete_index = next(
        index for index, sql in enumerate(executed_sql) if sql.startswith("DELETE FROM KG_NODE")
    )
    merge_index = next(
        index for index, sql in enumerate(executed_sql) if sql.startswith("MERGE INTO KG_NODE")
    )
    assert delete_index < merge_index
    assert executed_sql[-1] == ("REMOVE @DB.SCHEMA.KG_LOAD_STAGE/kg_processor/graph/job/LOAD123")


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
                "source_uri": "file:///file.txt",
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
        run_report={"job_id": "job", "nodes": 1},
        graph_metrics={"counts": {"nodes": 1}},
        extraction_trace=[{"stage": "ocr", "file_id": "file_1"}],
    )


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
