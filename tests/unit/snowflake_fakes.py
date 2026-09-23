"""One Snowflake connector fake for every adapter unit test.

The fakes model exactly the DB-API surface the adapters drive. A method the code
under test calls that is not modelled here fails with AttributeError instead of
quietly succeeding, so a new connector call cannot pass a test by accident.
"""

from __future__ import annotations

from collections.abc import Sequence

from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
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

CONFIG = SnowflakeConnectionConfig(
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

# The row a real PUT returns: source, target, sizes, compressions, STATUS, message.
UPLOADED_PUT_ROW = ("src", "dst", 1, 1, "NONE", "NONE", "UPLOADED", "")


class FakeCursor:
    """Record every statement and answer reads from rows prepared up front.

    ``rows`` feeds ``fetchone`` one row per call; ``result_sets`` feeds
    ``fetchall`` one list per call. Both answer ``None`` or ``[]`` once spent.
    """

    def __init__(
        self,
        rows: Sequence[Sequence[object] | None] | None = None,
        *,
        result_sets: Sequence[Sequence[object]] | None = None,
        rowcount: int = 0,
    ) -> None:
        self.rows: list[Sequence[object] | None] = list(rows or [])
        self.result_sets: list[list[object]] = [list(rows) for rows in result_sets or []]
        self.rowcount = rowcount
        self.executed: list[tuple[str, Sequence[object] | None]] = []
        self.executed_many: list[tuple[str, Sequence[Sequence[object]], dict[str, object]]] = []
        self.timeouts: list[int | None] = []
        self.log: list[tuple[str, object]] = []
        self.closed = False

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        timeout: int | None = None,
    ) -> object:
        self.executed.append((sql, params))
        self.timeouts.append(timeout)
        self.log.append(("execute", sql))
        # A PUT always answers with its upload status, which the bulk writers
        # read back before they COPY, so it must not be an empty result here.
        if sql.lstrip().startswith("PUT "):
            self.result_sets.insert(0, [UPLOADED_PUT_ROW])
        return None

    def executemany(
        self,
        sql: str,
        params: Sequence[Sequence[object]],
        **kwargs: object,
    ) -> object:
        self.executed_many.append((sql, params, kwargs))
        self.rowcount = len(params)
        return None

    def fetchone(self) -> Sequence[object] | None:
        return self.rows.pop(0) if self.rows else None

    def fetchall(self) -> list[object]:
        return self.result_sets.pop(0) if self.result_sets else []

    def close(self) -> object:
        self.closed = True
        return None


class FakeConnection:
    """Hand out one cursor and log transaction events in order with its statements.

    ``autocommit`` is a method because that is what the Snowflake connector
    exposes and what ``snowflake_transaction`` looks for before toggling it.
    """

    def __init__(
        self,
        rows: Sequence[Sequence[object] | None] | None = None,
        *,
        result_sets: Sequence[Sequence[object]] | None = None,
        rowcount: int = 0,
        cursor: FakeCursor | None = None,
    ) -> None:
        self.cursor_instance = cursor or FakeCursor(
            rows, result_sets=result_sets, rowcount=rowcount
        )
        self.log = self.cursor_instance.log
        self.autocommit_calls: list[bool] = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def autocommit(self, enabled: bool) -> None:
        self.autocommit_calls.append(enabled)
        self.log.append(("autocommit", enabled))

    def commit(self) -> object:
        self.committed = True
        self.log.append(("commit", None))
        return None

    def rollback(self) -> object:
        self.rolled_back = True
        self.log.append(("rollback", None))
        return None

    def close(self) -> object:
        self.closed = True
        self.log.append(("close", None))
        return None


def sample_batch() -> GraphWriteBatch:
    """One document's worth of every graph table, as the writers receive it."""

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
