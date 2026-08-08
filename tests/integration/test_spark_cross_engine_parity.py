"""Cross-engine parity contracts executed on a real Spark runtime.

Local and Spark finalization publish into the same tables, so identity, weight,
rank, and rating must be computed identically. These contracts run the Spark
expressions against the reference Python rules on values chosen to expose any
independent reimplementation.
"""

# PySpark is an optional dependency, so its imports stay inside the guarded
# contracts that only run when the distributed extra and a JVM are installed.
# ruff: noqa: PLC0415

from __future__ import annotations

import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.application.community_reports import structural_rating
from kg_processor.application.graph_dataset import GraphDatasetReader
from kg_processor.application.graph_merge import normalize_entity_name, normalize_relation_type
from kg_processor.application.spark_finalization import (
    _GRAPHFRAMES_COORDINATE,
    SparkFinalizationRequest,
    SparkGraphFinalizer,
    _normalized_name,
    _normalized_relation_type,
    _stable_id,
    _structural_rating_column,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    RelationObservation,
)
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.domain.stages import ExtractedDocumentShard, PreparedDocumentShard

# Values chosen so a rule reimplemented in SQL diverges from the reference rule:
# decomposed accents, case folding that changes length, ligatures, non-Latin
# scripts, whitespace runs, punctuation-only values, and oversized labels.
TRICKY_VALUES: tuple[str, ...] = (
    "Café Latte",
    "Café Latte",
    "  Spaced   Name  ",
    "line\tbreak\nname",
    "STRASSE",
    "Straße",
    "ﬁnance",
    "İstanbul",
    "Next.js",
    "Müller",
    "東京",
    "-- /// --",
    "",
    "Works-At",
    "works at",
    "very-long predicate " * 12,
    "x" * 200,
)
_GRAPH_ID = "parity-graph"
_ONTOLOGY_PROFILE: dict[str, Any] = {
    "name": "parity",
    "description": "Minimal profile exercising self-loop policy.",
    "mode": "hybrid",
    "entity_types": [{"name": "ENTITY", "description": "Any named thing."}],
    "relation_types": [
        {
            "name": "INTERACTS_WITH",
            "description": "One entity interacts with another.",
            "source_types": ["ENTITY"],
            "target_types": ["ENTITY"],
            "allow_self_loop": True,
        }
    ],
}


def _java_runtime_available() -> bool:
    """Distinguish macOS's Java launcher stub from an installed JVM."""

    try:
        return (
            subprocess.run(
                ["java", "-version"],
                check=False,
                capture_output=True,
                timeout=5,
            ).returncode
            == 0
        )
    except FileNotFoundError, subprocess.TimeoutExpired:
        return False


def _spark_runtime_available() -> bool:
    """Require both optional Python packages and a working Java runtime."""

    return (
        find_spec("pyspark") is not None
        and find_spec("graphframes") is not None
        and _java_runtime_available()
    )


pytestmark = pytest.mark.skipif(
    not _spark_runtime_available(),
    reason="the distributed-spark extra and an installed Java 17+ runtime are required",
)


@pytest.fixture(scope="module")
def spark_session() -> Any:
    """Provide the local Spark session shared by every contract in this module.

    One JVM serves the whole test process and resolves declared packages only
    while a context starts, so this session carries what a full finalization run
    needs and stays available for the finalizer to adopt. Finalization releases
    the session it used, which is why this fixture never stops it.
    """

    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[2]")
        .appName("flakegraph-cross-engine-parity")
        .config("spark.ui.enabled", "false")
        .config("spark.jars.packages", _GRAPHFRAMES_COORDINATE)
        # Local executors live in this process, and a developer hostname can
        # resolve to a VPN or Wi-Fi interface that is not a valid loopback route.
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )


def test_spark_canonical_identity_matches_the_local_rules(spark_session: Any) -> None:
    """Both engines must derive the same canonical keys and IDs for one surface."""

    from pyspark.sql import functions as F

    frame = spark_session.createDataFrame(
        [(index, value) for index, value in enumerate(TRICKY_VALUES)],
        "row_index int, value string",
    )
    computed = (
        frame.withColumn("normalized_name", _normalized_name(F.col("value")))
        .withColumn("relation_label", _normalized_relation_type(F.col("value")))
        .withColumn(
            "node_id",
            _stable_id(
                "node",
                F.lit(_GRAPH_ID),
                F.col("normalized_name"),
                F.lit("ENTITY"),
            ),
        )
        .collect()
    )

    assert len(computed) == len(TRICKY_VALUES)
    for row in computed:
        value = TRICKY_VALUES[row.row_index]
        assert row.normalized_name == normalize_entity_name(value)
        assert row.relation_label == normalize_relation_type(value)
        assert row.node_id == stable_id("node", _GRAPH_ID, normalize_entity_name(value), "ENTITY")


def test_spark_community_rating_matches_the_local_rule(spark_session: Any) -> None:
    """Ratings published by either engine must agree for identical structure."""

    from pyspark.sql import functions as F

    cases = [
        (3, 3, 2, 4, 1.0),
        (10, 5, 40, 4, 0.0),
        (1, 1, 0, 0, 0.5),
        (2, 3, 3, 6, 1.0),
    ]
    frame = spark_session.createDataFrame(
        cases,
        "member_count int, endpoint_pair_count int, reported_relations int, "
        "quotes int, confidence double",
    )
    computed = frame.withColumn(
        "structural_rating",
        _structural_rating_column(
            F.col("member_count"),
            F.col("endpoint_pair_count"),
            F.col("reported_relations"),
            F.col("quotes"),
            F.col("confidence"),
        ),
    ).collect()

    assert len(computed) == len(cases)
    for row in computed:
        rating, explanation = structural_rating(
            row.member_count,
            row.endpoint_pair_count,
            row.reported_relations,
            row.quotes,
            row.confidence,
        )
        assert row.structural_rating.rating == rating
        assert row.structural_rating.explanation == explanation
        assert 0.0 <= row.structural_rating.rating <= 10.0


def test_spark_edges_sum_observation_weight_and_reject_self_loops(
    spark_session: Any,
    tmp_path: Path,
) -> None:
    """Canonical edges carry summed observation weight, rank, and no self-loops.

    Finalization adopts the module session, so this contract runs last and leaves
    no session behind.
    """

    assert not spark_session.sparkContext._jsc.sc().isStopped()

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    run_id = "parity-run"
    first_text = "Alpha interacts with Beta."
    second_text = "alpha interacts with beta again."
    chunks = [
        _chunk("chunk-1", "file-1", first_text),
        _chunk("chunk-2", "file-1", second_text, start_offset=len(first_text)),
    ]
    confidence = 0.6
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        document_rows=[
            {
                "id": "document-1",
                "graph_id": _GRAPH_ID,
                "file_id": "file-1",
                "checksum": "checksum-1",
                "source_uri": "fixture.txt",
                "mime_type": "text/plain",
                "size_bytes": len(first_text) + len(second_text),
                "ocr_provider": "builtin_text",
            }
        ],
        page_rows=[],
        block_rows=[],
        asset_rows=[],
        chunks=chunks,
        document_context_entities=[],
    )
    extracted = ExtractedDocumentShard(
        prepared=prepared,
        observations=ExtractionObservations(
            entities=[
                _mention("mention-alpha-1", "Alpha", "chunk-1", start_offset=0),
                _mention(
                    "mention-beta-1",
                    "Beta",
                    "chunk-1",
                    start_offset=first_text.index("Beta"),
                ),
                _mention("mention-alpha-2", "alpha", "chunk-2", start_offset=0),
                _mention(
                    "mention-beta-2",
                    "beta",
                    "chunk-2",
                    start_offset=second_text.index("beta"),
                ),
            ],
            relations=[
                _relation(
                    "relation-1",
                    "mention-alpha-1",
                    "mention-beta-1",
                    "chunk-1",
                    first_text,
                    confidence,
                ),
                _relation(
                    "relation-2",
                    "mention-alpha-2",
                    "mention-beta-2",
                    "chunk-2",
                    second_text,
                    confidence,
                ),
                # Two spellings of one entity resolve to a single canonical node.
                # The resulting loop is an assembly artifact, not an assertion the
                # ontology's self-loop allowance should preserve.
                _relation(
                    "relation-3",
                    "mention-alpha-1",
                    "mention-alpha-2",
                    "chunk-1",
                    first_text,
                    confidence,
                ),
            ],
            chunk_count=2,
            window_count=1,
        ),
    )
    store.put(
        f"{run_id}/prepared_document/prepared.json",
        prepared.model_dump_json().encode(),
        "application/json",
    )
    store.put(
        f"{run_id}/extracted_document/extracted.json",
        extracted.model_dump_json().encode(),
        "application/json",
    )
    settings = Settings.load(
        env={},
        overrides={
            "runtime": {"runtime": "kubernetes"},
            "job": {"graph_id": _GRAPH_ID},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
            "cache": {"provider": "none"},
            "ontology": {"profile": _ONTOLOGY_PROFILE},
            "distributed": {
                "artifact_uri": root.as_uri(),
                "finalization_engine": "spark",
                "spark_master": "local[2]",
                "spark_executor_instances": 2,
                "spark_executor_cores": 1,
                "spark_executor_memory": "1g",
            },
        },
    )

    manifest = SparkGraphFinalizer(settings).finalize(
        SparkFinalizationRequest(run_id=run_id, graph_id=_GRAPH_ID, attempt=1)
    )
    batch = GraphDatasetReader(store).read(manifest)

    assert len(batch.edges) == 1
    edge = batch.edges[0]
    assert edge.source_node_id != edge.target_node_id
    assert edge.weight == pytest.approx(2 * confidence)
    assert edge.weight <= settings.graph.relation_weight_max
    assert edge.evidence_count == 2
    assert {node.id for node in batch.nodes} == {edge.source_node_id, edge.target_node_id}
    assert all(node.rank == float(node.degree) for node in batch.nodes)
    assert all(node.degree == 1 for node in batch.nodes)
    assert manifest.metrics["hard_quality_error_count"] == 0


def _chunk(chunk_id: str, file_id: str, content: str, *, start_offset: int = 0) -> Chunk:
    """Build a valid source chunk with the offsets evidence joins consume."""

    return Chunk(
        id=chunk_id,
        graph_id=_GRAPH_ID,
        file_id=file_id,
        document_id=file_id,
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=start_offset,
        end_offset=start_offset + len(content),
        token_count=len(content.split()),
        content_hash=sha256_hex(content),
    )


def _mention(entity_id: str, name: str, chunk_id: str, *, start_offset: int) -> EntityMention:
    """Build one grounded mention of a canonical entity."""

    return EntityMention(
        id=entity_id,
        name=name,
        type="ENTITY",
        description=f"A mention of {name}.",
        source_chunk_id=chunk_id,
        quote=name,
        start_offset=start_offset,
        end_offset=start_offset + len(name),
    )


def _relation(
    relation_id: str,
    source_entity_id: str,
    target_entity_id: str,
    chunk_id: str,
    quote: str,
    confidence: float,
) -> RelationObservation:
    """Build one grounded relation observation between accepted mentions."""

    return RelationObservation(
        id=relation_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type="INTERACTS_WITH",
        description="One entity interacts with another.",
        source_chunk_id=chunk_id,
        quote=quote,
        start_offset=0,
        end_offset=len(quote),
        confidence=confidence,
    )
