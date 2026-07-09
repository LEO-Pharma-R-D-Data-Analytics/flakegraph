from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kg_processor.application.graph_quality import GraphQualityError
from kg_processor.application.inspect import inspect_local_graph
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.application.progress import ProgressEvent
from kg_processor.config.settings import Settings
from kg_processor.domain.documents import (
    InputFile,
    LayoutBlock,
    ParsedAsset,
    ParsedDocument,
    ParsedPage,
)
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractionResult, GraphWriteBatch
from kg_processor.domain.jobs import JobFileClaim
from kg_processor.factories import (
    build_cache,
    build_embedding_provider,
    build_file_source,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)
from kg_processor.ports.embeddings import EmbedOptions
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    GraphRepairRequest,
    LlmOptions,
)
from kg_processor.ports.ocr import OcrOptions


def test_pipeline_writes_local_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen. Acme Corp supports Contoso Health.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )

    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    )

    batch = pipeline.run()

    assert batch.run_report["files_processed"] == 1
    assert batch.nodes
    assert batch.edges
    assert batch.graph_metrics["quality"]["ok"]
    assert (output_dir / "nodes.parquet").exists()
    assert len(pd.read_parquet(output_dir / "nodes.parquet")) == len(batch.nodes)
    documents = pd.read_parquet(output_dir / "documents.parquet")
    pages = pd.read_parquet(output_dir / "pages.parquet")
    blocks = pd.read_parquet(output_dir / "blocks.parquet")
    assets = pd.read_parquet(output_dir / "assets.parquet")
    chunks = pd.read_parquet(output_dir / "chunks.parquet")
    communities = pd.read_parquet(output_dir / "communities.parquet")
    findings = pd.read_parquet(output_dir / "community_findings.parquet")
    assert set(["id", "graph_id", "file_id"]).issubset(documents.columns)
    assert documents.loc[0, "id"] == documents.loc[0, "file_id"]
    assert set(["id", "graph_id", "file_id", "page_number"]).issubset(pages.columns)
    assert set(["id", "graph_id", "file_id", "page_number", "kind", "metadata"]).issubset(
        blocks.columns
    )
    assert set(["id", "graph_id", "file_id", "kind", "metadata"]).issubset(assets.columns)
    assert set(["id", "graph_id", "document_id", "section_path"]).issubset(chunks.columns)
    assert set(["rating_explanation", "suggested_questions"]).issubset(communities.columns)
    assert set(["id", "graph_id", "community_id"]).issubset(findings.columns)
    assert documents.loc[0, "graph_id"] == "graph"
    assert pages.loc[0, "graph_id"] == "graph"
    assert chunks.loc[0, "graph_id"] == "graph"
    assert communities.loc[0, "rating_explanation"]
    assert findings.loc[0, "graph_id"] == "graph"


def test_pipeline_processes_original_manifest_corpus(tmp_path: Path) -> None:
    output_dir = tmp_path / "manifest-out"
    settings = Settings.load(
        Path("configs/local-manifest-smoke.yaml"),
        overrides={
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
            "graph": {
                "chunk_token_size": 100000,
                "chunk_token_overlap": 0,
                "max_chunks_per_llm_call": 2000,
                "max_entities_per_batch": 80,
                "max_relations_per_batch": 80,
                "gleaning_max_passes": 0,
            },
        },
    )

    batch = _build_pipeline(settings).run()
    inspection = inspect_local_graph(output_dir)

    assert batch.run_report["files_processed"] == 6
    assert batch.run_report["files_seen"] == 6
    assert len(batch.documents) == 6
    assert len(batch.pages) >= 12
    assert batch.chunks
    assert batch.nodes
    assert batch.evidence
    assert batch.graph_metrics["providers"] == {
        "ocr": "builtin_text",
        "llm": "fake",
        "embedding": "hash",
        "writer": "local_artifacts",
        "cache": "none",
    }
    assert batch.graph_metrics["quality"]["ok"] is True
    assert inspection["schema"]["ok"] is True
    assert inspection["quality"]["ok"] is True
    assert inspection["tables"]["documents"] == 6
    assert inspection["tables"]["pages"] >= 12
    assert {document["source_uri"] for document in batch.documents} == {
        "file://data/samples/martial-arts-schools.pptx",
        "file://data/samples/martial-arts-timeline.html",
        "file://data/samples/martial-arts-interview.docx",
        "file://data/samples/martial-arts-overview.pdf",
        "file://data/samples/martial-arts-lineages.pdf",
        "file://data/samples/smoke.txt",
    }


def test_pipeline_emits_structured_progress_events(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "progress-job", "graph_id": "progress-graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )
    progress = RecordingProgressSink()
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
        progress_sink=progress,
    )

    batch = pipeline.run()
    records = [event.as_log_record() for event in progress.events]

    assert batch.run_report["files_processed"] == 1
    assert all(record["event"] == "kg_processor.progress" for record in records)
    assert all(record["job_id"] == "progress-job" for record in records)
    assert all(record["graph_id"] == "progress-graph" for record in records)
    assert ("file_source", "started") in _stage_statuses(records)
    assert ("file_source", "completed") in _stage_statuses(records)
    assert ("ocr", "started") in _stage_statuses(records)
    assert ("ocr", "completed") in _stage_statuses(records)
    assert ("graph_extraction", "completed") in _stage_statuses(records)
    assert ("quality", "completed") in _stage_statuses(records)
    assert ("write", "started") in _stage_statuses(records)
    assert ("write", "completed") in _stage_statuses(records)
    ocr_completed = next(
        record
        for record in records
        if record["stage"] == "ocr" and record["status"] == "completed"
    )
    write_completed = next(
        record
        for record in records
        if record["stage"] == "write" and record["status"] == "completed"
    )
    ocr_metadata = ocr_completed["metadata"]
    write_counts = write_completed["counts"]
    assert isinstance(ocr_metadata, dict)
    assert isinstance(write_counts, dict)
    assert ocr_metadata["provider"] == "builtin_text"
    assert write_counts["nodes"] == len(batch.nodes)


def test_pipeline_persists_ocr_assets_and_chunk_asset_references(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=AssetOcrProvider(),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    )

    batch = pipeline.run()

    assert batch.assets == [
        {
            "id": "asset_1",
            "graph_id": "graph",
            "file_id": batch.documents[0]["file_id"],
            "kind": "image",
            "page_number": 1,
            "uri": "file:///figure-1.png",
            "metadata": {"caption": "Figure 1"},
        }
    ]
    assert batch.blocks[0]["id"] == "block_1"
    assert batch.blocks[0]["bbox"] is None
    assert batch.chunks[0].asset_ids == ["asset_1"]
    assert batch.chunks[0].block_ids == ["block_1"]
    assets = pd.read_parquet(output_dir / "assets.parquet")
    blocks = pd.read_parquet(output_dir / "blocks.parquet")
    chunks = pd.read_parquet(output_dir / "chunks.parquet")
    assert blocks.loc[0, "id"] == "block_1"
    assert blocks.loc[0, "metadata"] == "{}"
    assert assets.loc[0, "id"] == "asset_1"
    assert assets.loc[0, "metadata"] == '{"caption":"Figure 1"}'
    assert chunks.loc[0, "asset_ids"] == ["asset_1"]


def test_pipeline_reuses_local_cache_on_second_run(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
            "cache": {"provider": "local", "path": cache_dir},
        }
    )

    first = _build_pipeline(settings).run()
    second = _build_pipeline(settings).run()

    assert first.graph_metrics["cache"] == {
        "ocr_hits": 0,
        "ocr_misses": 1,
        "extraction_hit": False,
    }
    assert second.graph_metrics["cache"] == {
        "ocr_hits": 1,
        "ocr_misses": 0,
        "extraction_hit": True,
    }
    assert (cache_dir / "ocr").exists()
    assert (cache_dir / "extraction").exists()


def test_pipeline_resumes_extraction_from_completed_batch_checkpoint(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    input_dir.mkdir()
    (input_dir / "01-alice.txt").write_text(
        "Alice Smith documents a clinical process.",
        encoding="utf-8",
    )
    (input_dir / "02-bob.txt").write_text(
        "Bob Jones documents a manufacturing process.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "checkpoint-test", "graph_id": "checkpoint-graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {
                "chunk_token_size": 30,
                "chunk_token_overlap": 0,
                "max_chunks_per_llm_call": 1,
                "gleaning_max_passes": 0,
            },
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
            "cache": {"provider": "local", "path": cache_dir},
        }
    )

    first_provider = CheckpointResumeLlmProvider(fail_on_call=2)
    first_pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=first_provider,
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    )

    with pytest.raises(RuntimeError, match="batch 2 failed"):
        first_pipeline.run()

    extraction_cache_files = sorted((cache_dir / "extraction").glob("*.json"))
    assert len(extraction_cache_files) == 1
    assert [
        [_entity_name_from_content(content) for content in batch]
        for batch in first_provider.seen_batches
    ] == [
        ["Alice Smith"],
        ["Bob Jones"],
    ]
    assert not (output_dir / "nodes.parquet").exists()

    second_provider = CheckpointResumeLlmProvider()
    second_pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=second_provider,
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    )

    batch = second_pipeline.run()

    assert [
        [_entity_name_from_content(content) for content in batch]
        for batch in second_provider.seen_batches
    ] == [
        ["Bob Jones"],
    ]
    assert {node.name for node in batch.nodes} == {"Alice Smith", "Bob Jones"}
    assert batch.graph_metrics["cache"] == {
        "ocr_hits": 2,
        "ocr_misses": 0,
        "extraction_hit": False,
    }
    assert len(sorted((cache_dir / "extraction").glob("*.json"))) == 3
    assert (output_dir / "nodes.parquet").exists()


def test_pipeline_records_filter_decision_trace_events(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {
                "chunk_token_size": 30,
                "chunk_token_overlap": 5,
                "gleaning_max_passes": 0,
                "entity_blocklist": ["Acme Corp"],
            },
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )

    batch = _build_pipeline(settings).run()

    dropped = [
        event
        for event in batch.extraction_trace
        if event["stage"] == "filter_decision" and event["action"] == "dropped"
    ]
    assert any(
        event["kind"] == "entity"
        and event["reason"] == "blocklisted_entity"
        and event["name"] == "Acme Corp"
        for event in dropped
    )
    assert batch.graph_metrics["filtering"]["dropped_entities_by_reason"] == {
        "blocklisted_entity": 1
    }
    assert (output_dir / "extraction_trace.jsonl").exists()


def test_pipeline_records_merge_decision_trace_events(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp. Alice Smith supports Acme Corp.",
        encoding="utf-8",
    )
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )

    batch = _build_pipeline(settings).run()

    merge_events = [event for event in batch.extraction_trace if event["stage"] == "merge_decision"]
    assert merge_events
    assert batch.graph_metrics["merge"]["decision_actions"]["created"] > 0
    assert "canonical_node_created" in batch.graph_metrics["merge"]["decision_reasons"]
    assert (output_dir / "extraction_trace.jsonl").exists()


def test_pipeline_processes_only_claimed_job_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    skipped = input_dir / "skipped.txt"
    claimed = input_dir / "claimed.txt"
    skipped.write_text("Bob Jones works at Other Corp.", encoding="utf-8")
    claimed.write_text("Alice Smith works at Acme Corp in Copenhagen.", encoding="utf-8")
    settings = Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {"chunk_token_size": 30, "chunk_token_overlap": 5, "gleaning_max_passes": 0},
            "writer": {"provider": "local_artifacts", "output_path": output_dir},
        }
    )
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
        claimed_files=[
            JobFileClaim(
                job_id="test",
                graph_id="graph",
                file_id="app-file-claimed",
                source_uri=claimed.resolve().as_uri(),
            )
        ],
    )

    batch = pipeline.run()
    job_file_results = pipeline.job_file_results(batch)

    assert batch.run_report["files_processed"] == 1
    assert batch.documents[0]["file_id"] == "app-file-claimed"
    assert batch.documents[0]["source_uri"] == claimed.resolve().as_uri()
    assert batch.extraction_trace[0] == {
        "stage": "file_source",
        "files_available": 2,
        "files_seen": 1,
        "claimed_files": 1,
    }
    assert [result.file_id for result in job_file_results] == ["app-file-claimed"]
    assert job_file_results[0].rows_written > 0
    assert job_file_results[0].rows_written == sum(job_file_results[0].row_counts.values())
    assert job_file_results[0].row_counts["documents"] == 1
    assert job_file_results[0].row_counts["chunks"] == len(batch.chunks)
    assert job_file_results[0].audit == {
        "graph_id": "graph",
        "write_scope": "file_batch",
        "quality_ok": True,
    }
    assert job_file_results[0].ocr_provider == "builtin_text"
    assert skipped.resolve().as_uri() not in {str(row.get("source_uri")) for row in batch.documents}


def test_pipeline_fails_before_writing_when_quality_checks_fail(tmp_path: Path) -> None:
    settings = _quality_gate_settings(tmp_path, fail_on_quality_error=True)
    writer = RecordingWriter()
    pipeline = _build_pipeline_with_bad_embeddings(settings, writer)

    with pytest.raises(GraphQualityError, match="node_embedding_dimensions") as exc_info:
        pipeline.run()

    failed_checks = {check.name for check in exc_info.value.result.checks if not check.ok}
    assert "node_embedding_dimensions" in failed_checks
    assert writer.batch is None


def test_pipeline_can_write_failed_quality_for_diagnostics(tmp_path: Path) -> None:
    settings = _quality_gate_settings(tmp_path, fail_on_quality_error=False)
    writer = RecordingWriter()
    pipeline = _build_pipeline_with_bad_embeddings(settings, writer)

    batch = pipeline.run()

    assert writer.batch is batch
    assert batch.graph_metrics["quality"]["ok"] is False
    quality_events = [event for event in batch.extraction_trace if event["stage"] == "quality"]
    assert len(quality_events) == 1
    assert quality_events[0]["ok"] is False
    assert {check["name"] for check in quality_events[0]["failed_checks"]} >= {
        "node_embedding_dimensions",
        "edge_embedding_dimensions",
    }


def _build_pipeline(settings: Settings) -> KgProcessorPipeline:
    return KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
    )


def _build_pipeline_with_bad_embeddings(
    settings: Settings,
    writer: RecordingWriter,
) -> KgProcessorPipeline:
    return KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=WrongDimensionEmbeddingProvider(),
        writer=writer,
        cache=build_cache(settings),
    )


def _quality_gate_settings(
    tmp_path: Path,
    *,
    fail_on_quality_error: bool,
) -> Settings:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen. Acme Corp supports Contoso Health.",
        encoding="utf-8",
    )
    return Settings.load(
        overrides={
            "job": {"job_id": "test", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "graph": {
                "chunk_token_size": 30,
                "chunk_token_overlap": 5,
                "gleaning_max_passes": 0,
                "fail_on_quality_error": fail_on_quality_error,
            },
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )


class CheckpointResumeLlmProvider:
    """Small complete LLM provider used to verify extraction checkpoint behavior.

    The provider records the chunk text passed to `extract_graph`. On the resume
    run the first document should be absent from that record because it was read
    from the per-batch extraction cache created before the prior failure.
    """

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.seen_batches: list[list[str]] = []

    def extract_graph(self, chunks: list[Chunk], options: LlmOptions) -> ExtractionResult:
        self.seen_batches.append([chunk.content for chunk in chunks])
        if self.fail_on_call is not None and len(self.seen_batches) == self.fail_on_call:
            raise RuntimeError(f"batch {self.fail_on_call} failed")
        return ExtractionResult(
            entities=[_entity_for_chunk(chunk) for chunk in chunks],
            provider_metadata={"provider": "checkpoint_resume_test"},
        )

    def repair_graph_extraction(self, request: GraphRepairRequest) -> ExtractionResult:
        return request.invalid_result

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        return DescriptionMergeResult(
            description=" ".join(request.descriptions),
            provider_metadata={"provider": "checkpoint_resume_test"},
        )

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        return CommunitySummaryResult(
            title=request.title_seed,
            summary=f"Community centered on {request.title_seed}.",
            rating=5.0,
            rating_explanation="Deterministic test summary.",
            findings=[],
            suggested_questions=[],
            provider_metadata={"provider": "checkpoint_resume_test"},
        )


def _entity_for_chunk(chunk: Chunk) -> ExtractedEntity:
    name = _entity_name_from_content(chunk.content)
    start_offset = chunk.content.index(name)
    return ExtractedEntity(
        name=name,
        type="PERSON",
        description=f"{name} is mentioned in the document.",
        source_chunk_id=chunk.id,
        quote=name,
        start_offset=start_offset,
        end_offset=start_offset + len(name),
    )


def _entity_name_from_content(content: str) -> str:
    if "Alice Smith" in content:
        return "Alice Smith"
    if "Bob Jones" in content:
        return "Bob Jones"
    raise ValueError(f"Unexpected checkpoint test content: {content}")


class WrongDimensionEmbeddingProvider:
    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        return [[0.1] for _ in texts]


class AssetOcrProvider:
    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        text = file.path.read_text(encoding="utf-8")
        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=[
                ParsedPage(
                    page_number=1,
                    markdown=text,
                    raw_text=text,
                    blocks=[
                        LayoutBlock(
                            id="block_1",
                            page_number=1,
                            kind="paragraph",
                            text=text,
                        )
                    ],
                )
            ],
            assets=[
                ParsedAsset(
                    id="asset_1",
                    kind="image",
                    page_number=1,
                    uri="file:///figure-1.png",
                    metadata={"caption": "Figure 1"},
                )
            ],
            provider_metadata={"provider": "asset_ocr"},
        )


class RecordingWriter:
    def __init__(self) -> None:
        self.batch: GraphWriteBatch | None = None

    def write(self, batch: GraphWriteBatch) -> None:
        self.batch = batch


class RecordingProgressSink:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)


def _stage_statuses(records: list[dict[str, object]]) -> set[tuple[object, object]]:
    return {(record["stage"], record["status"]) for record in records}
