"""End-to-end contract test for partitioned Spark graph finalization."""

from __future__ import annotations

import json
import subprocess
from importlib.util import find_spec
from pathlib import Path

import pytest

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.application.graph_dataset import GraphDatasetReader
from kg_processor.application.spark_finalization import (
    SparkFinalizationRequest,
    SparkGraphFinalizer,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    RelationObservation,
)
from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.domain.graph import Chunk, GraphWriteBatch
from kg_processor.domain.ids import sha256_hex
from kg_processor.domain.stages import ExtractedDocumentShard, PreparedDocumentShard


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
    """Require both optional Python packages and a working Java runtime.

    Spark is intentionally an optional deployment backend. The ordinary development
    environment should skip these integration contracts, while environments installed
    with ``--extra distributed-spark`` exercise them whenever Java is available.
    """

    return (
        find_spec("pyspark") is not None
        and find_spec("graphframes") is not None
        and _java_runtime_available()
    )


pytestmark = pytest.mark.skipif(
    not _spark_runtime_available(),
    reason="the distributed-spark extra and an installed Java 17+ runtime are required",
)


def _assert_extraction_contract(extraction: dict[str, object]) -> None:
    """Check a fleet run reports extraction the way a local run does.

    Rejection and repair reasons are the only record of why a document lost
    entities or relations, and a fleet run is the one an operator can least
    easily inspect. Reporting them under different keys, or summing a different
    set of stages, would leave two runs over one corpus incomparable.
    """

    assert extraction["entity_calls"] == 1
    assert extraction["relation_calls"] == 1
    assert extraction["verification_calls"] == 1
    assert extraction["document_context_calls"] == 1
    # Verification raises reasons that a local run does not fold in, so its nine
    # rejections must not appear here either.
    assert extraction["record_actions"] == {
        "domain_or_range_violation": 4,
        "duplicate": 1,
        "ungrounded_quote": 5,
    }
    assert extraction["chunk_count"] == 2
    assert extraction["window_count"] == 1
    assert extraction["entity_mentions"] == 7
    assert extraction["relation_observations"] == 1


def test_spark_finalizer_writes_manifest_without_collecting_graph_tables(
    tmp_path: Path,
) -> None:
    """Resolve aliases, canonicalize endpoints, and publish readable Parquet tables."""

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    run_id = "spark-run"
    graph_id = "spark-graph"
    first_chunk_text = (
        "Judo influenced Brazilian Jiu-Jitsu. Elman simple recurrent network is also called SRN."
    )
    second_chunk_text = (
        "The name judo refers to the same martial art. "
        "Scene Representation Networks (SRN) render scenes. "
        "DA may abbreviate either Dario Amodei or Diogo Almeida."
    )
    chunks = [
        _chunk(
            "chunk-1",
            "file-1",
            first_chunk_text,
            start_offset=120,
        ),
        _chunk("chunk-2", "file-2", second_chunk_text),
    ]
    prepared = PreparedDocumentShard(
        file_ids=["file-1", "file-2"],
        files_seen=2,
        documents_processed=2,
        document_rows=[
            {
                "id": "document-1",
                "graph_id": graph_id,
                "file_id": "file-1",
                "checksum": "checksum-1",
                "source_uri": "fixture.txt",
                "mime_type": "text/plain",
                "size_bytes": 42,
                "ocr_provider": "builtin_text",
            }
        ],
        page_rows=[
            {
                "id": "page-1",
                "graph_id": graph_id,
                "file_id": "file-1",
                "page_number": 1,
                "markdown": "Judo influenced Brazilian Jiu-Jitsu.",
                "raw_text": "Judo influenced Brazilian Jiu-Jitsu.",
                "detected_language": "en",
            }
        ],
        block_rows=[
            {
                "id": "block-1",
                "graph_id": graph_id,
                "file_id": "file-1",
                "page_number": 1,
                "kind": "paragraph",
                "text": "Judo influenced Brazilian Jiu-Jitsu.",
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "metadata": {"confidence": 0.99, "layout": {"column": 1}},
            }
        ],
        asset_rows=[
            {
                "id": "asset-1",
                "graph_id": graph_id,
                "file_id": "file-1",
                "kind": "figure",
                "page_number": 1,
                "uri": "asset://figure-1",
                "metadata": {"caption": "Technique"},
            }
        ],
        chunks=chunks,
        # This focal entity is deliberately absent from ordinary observations.
        # Spark must retain compacted document context at the final barrier.
        document_context_entities=[
            _mention(
                "entity-bjj",
                "Brazilian Jiu-Jitsu",
                "chunk-1",
                start_offset=16,
            )
        ],
    )
    extracted = ExtractedDocumentShard(
        prepared=prepared,
        observations=ExtractionObservations(
            entities=[
                _mention("entity-judo-1", "Judo", "chunk-1", start_offset=0),
                _mention("entity-judo-2", "judo", "chunk-2", start_offset=9),
                _mention(
                    "entity-elman-srn",
                    "Elman simple recurrent network",
                    "chunk-1",
                    start_offset=first_chunk_text.index("Elman"),
                    entity_type="MODEL",
                    aliases=["SRN"],
                ),
                _mention(
                    "entity-scene-srn",
                    "SRN",
                    "chunk-2",
                    start_offset=second_chunk_text.index("SRN"),
                    entity_type="MODEL",
                    aliases=["Scene Representation Networks"],
                ),
                _mention(
                    "entity-da",
                    "DA",
                    "chunk-2",
                    start_offset=second_chunk_text.index("DA"),
                    entity_type="PERSON",
                ),
                _mention(
                    "entity-dario",
                    "Dario Amodei",
                    "chunk-2",
                    start_offset=second_chunk_text.index("Dario Amodei"),
                    entity_type="PERSON",
                ),
                _mention(
                    "entity-diogo",
                    "Diogo Almeida",
                    "chunk-2",
                    start_offset=second_chunk_text.index("Diogo Almeida"),
                    entity_type="PERSON",
                ),
            ],
            relations=[
                RelationObservation(
                    id="relation-1",
                    source_entity_id="entity-judo-1",
                    target_entity_id="entity-bjj",
                    relation_type="INFLUENCED",
                    description="Judo influenced Brazilian Jiu-Jitsu.",
                    source_chunk_id="chunk-1",
                    quote="Judo influenced Brazilian Jiu-Jitsu.",
                    start_offset=0,
                    end_offset=len("Judo influenced Brazilian Jiu-Jitsu."),
                )
            ],
            chunk_count=2,
            window_count=1,
            # The counters a worker raises per window. A fleet run has no other
            # record of why records were dropped or repaired.
            trace=[
                {
                    "stage": "entity_extraction",
                    "record_actions": {"ungrounded_quote": 2, "duplicate": 1},
                },
                {
                    "stage": "document_context_extraction",
                    "record_actions": {"ungrounded_quote": 3},
                },
                {
                    "stage": "relation_extraction",
                    "record_actions": {"domain_or_range_violation": 4},
                },
                # Verification raises reasons of its own. A local run does not
                # fold them into record_actions, so neither may this one.
                {"stage": "relation_verification", "record_actions": {"invalid_schema": 9}},
            ],
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
            "job": {"graph_id": graph_id},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
            "cache": {"provider": "none"},
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
        SparkFinalizationRequest(run_id=run_id, graph_id=graph_id, attempt=1)
    )
    batch = GraphDatasetReader(store).read(manifest)

    assert manifest.tables["nodes"].row_count == 7
    assert manifest.tables["edges"].row_count == 1
    # GraphFrames label propagation may resolve tied labels differently as Spark
    # partition order changes. FlakeGraph's contract is the validity and internal
    # consistency of the resulting communities, not a vendor algorithm's exact
    # label count for this deliberately small, highly connected fixture.
    _assert_community_contract(batch, manifest, settings.graph.min_community_size)
    assert all(item.size_bytes > 0 for table in manifest.tables.values() for item in table.files)
    assert manifest.metrics["dangling_edges"] == 0
    assert manifest.metrics["evidence_coverage"] == 1.0
    assert manifest.metrics["community_co_mention_strategy"] == "bounded_chunk_anchor_star"
    assert manifest.metrics["hard_quality_error_count"] == 0
    # A fleet run has to report the same rejection reasons a local run does, or
    # the runs whose workers are hardest to inspect are the ones that explain
    # nothing. Counts are summed across every stage that grounds records.
    _assert_extraction_contract(manifest.metrics["extraction"])
    assert not any(manifest.metrics["hard_quality_errors"].values())
    assert len(batch.nodes) == 7
    assert len(batch.edges) == 1
    assert batch.documents[0]["checksum"] == "checksum-1"
    assert batch.pages[0]["raw_text"] == "Judo influenced Brazilian Jiu-Jitsu."
    assert json.loads(batch.blocks[0]["metadata"])["layout"] == {"column": 1}
    assert json.loads(batch.assets[0]["metadata"])["caption"] == "Technique"
    assert batch.edges[0].source_node_id in {node.id for node in batch.nodes}
    assert batch.edges[0].target_node_id in {node.id for node in batch.nodes}
    assert {node.normalized_name for node in batch.nodes} == {
        "judo",
        "brazilianjiujitsu",
        "elmansimplerecurrentnetwork",
        "srn",
        "da",
        "darioamodei",
        "diogoalmeida",
    }
    assert all(node.embedding and len(node.embedding) == 8 for node in batch.nodes)
    assert batch.communities[0].summary.startswith("Community around")
    assert batch.community_findings[0].summary == "Connected entities"
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    for evidence in batch.evidence:
        chunk = chunks_by_id[evidence.chunk_id]
        local_start = evidence.start_offset - chunk.start_offset
        local_end = evidence.end_offset - chunk.start_offset
        assert 0 <= local_start < local_end <= len(chunk.content)
        assert chunk.content[local_start:local_end] == evidence.quote


def test_spark_finalizer_publishes_schema_correct_empty_graph(tmp_path: Path) -> None:
    """A corpus with no accepted mentions must still publish every logical table."""

    root = tmp_path / "artifacts"
    store = LocalBlobStore(root.as_uri())
    store.initialize()
    prepared = PreparedDocumentShard(
        file_ids=["file-empty"],
        files_seen=1,
        documents_processed=1,
        document_rows=[{"id": "document-empty", "graph_id": "empty-graph"}],
        chunks=[_chunk("chunk-empty", "file-empty", "plain lowercase text")],
    )
    extracted = ExtractedDocumentShard(
        prepared=prepared,
        observations=ExtractionObservations(chunk_count=1, window_count=1),
    )
    store.put(
        "empty-run/prepared_document/prepared.json",
        prepared.model_dump_json().encode(),
        "application/json",
    )
    store.put(
        "empty-run/extracted_document/extracted.json",
        extracted.model_dump_json().encode(),
        "application/json",
    )
    settings = Settings.load(
        env={},
        overrides={
            "runtime": {"runtime": "kubernetes"},
            "job": {"graph_id": "empty-graph"},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {
                "provider": "local_artifacts",
                "output_path": str(tmp_path / "out"),
            },
            "cache": {"provider": "none"},
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
        SparkFinalizationRequest(run_id="empty-run", graph_id="empty-graph", attempt=1)
    )
    batch = GraphDatasetReader(store).read(manifest)

    assert set(manifest.tables) == {
        "documents",
        "pages",
        "blocks",
        "assets",
        "chunks",
        "nodes",
        "edges",
        "edge_observations",
        "evidence",
        "entity_sources",
        "communities",
        "community_findings",
    }
    assert manifest.metrics["input_entity_mentions"] == 0
    assert batch.nodes == []
    assert batch.edges == []


def _chunk(
    chunk_id: str,
    file_id: str,
    content: str,
    *,
    start_offset: int = 0,
) -> Chunk:
    """Build a valid source chunk with offsets used by evidence joins."""

    return Chunk(
        id=chunk_id,
        graph_id="spark-graph",
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


def _assert_community_contract(
    batch: GraphWriteBatch,
    manifest: GraphDatasetManifest,
    min_community_size: int,
) -> None:
    """Validate FlakeGraph's output contract without fixing GraphFrames labels."""

    community_count = manifest.tables["communities"].row_count
    assert community_count > 0
    assert manifest.tables["community_findings"].row_count == community_count
    assert len(batch.communities) == community_count
    assert len(batch.community_findings) == community_count
    node_ids = {node.id for node in batch.nodes}
    community_ids = {community.id for community in batch.communities}
    assert all(
        len(community.member_node_ids) >= min_community_size
        and set(community.member_node_ids) <= node_ids
        for community in batch.communities
    )
    assert {finding.community_id for finding in batch.community_findings} == community_ids


def _mention(
    entity_id: str,
    name: str,
    chunk_id: str,
    *,
    start_offset: int,
    entity_type: str = "MARTIAL_ART",
    aliases: list[str] | None = None,
) -> EntityMention:
    """Build one grounded mention for identity-resolution coverage."""

    return EntityMention(
        id=entity_id,
        name=name,
        type=entity_type,
        description=f"A mention of {name}.",
        source_chunk_id=chunk_id,
        quote=name,
        aliases=aliases or [],
        start_offset=start_offset,
        end_offset=start_offset + len(name),
    )
