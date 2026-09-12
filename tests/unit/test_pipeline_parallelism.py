"""Concurrency contracts for provider-bound pipeline stages."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Lock
from typing import Any, cast

import pytest

from kg_processor.application.llm_extractors import LlmEntityExtractor
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.config.settings import Settings
from kg_processor.domain.extraction import EntityExtractionOutcome, EntityMention, ExtractionWindow
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import sha256_hex
from kg_processor.domain.ontology import OntologyProfile
from kg_processor.factories import (
    build_embedding_provider,
    build_file_source,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)


def test_pipeline_close_releases_each_unique_dependency_once(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "files": {"input_path": tmp_path},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    class _Closeable:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    dependency = _Closeable()
    shared = cast(Any, dependency)
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=shared,
        ocr=shared,
        llm=shared,
        embeddings=shared,
        writer=shared,
        cache=shared,
    )

    pipeline.close()

    assert dependency.close_calls == 1


def test_document_context_calls_run_concurrently_and_return_in_source_order(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    """Parallelize independent prefixes without making persisted order nondeterministic."""

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "job": {"job_id": "parallel-context", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "ontology": {"profile_path": "data/deep_learning_papers/ontology.yaml"},
            "graph": {"extraction_parallelism": 3},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
    )
    chunks = [_context_chunk(index) for index in range(3)]
    all_started = Event()
    state_lock = Lock()
    started = 0

    def extract_context(
        _extractor: LlmEntityExtractor,
        window: ExtractionWindow,
        _ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_entities: int,
    ) -> EntityExtractionOutcome:
        """Block each fake request until all document calls have entered concurrently."""

        nonlocal started
        _ = (model, timeout_seconds, max_entities)
        with state_lock:
            started += 1
            if started == len(chunks):
                all_started.set()
        if not all_started.wait(timeout=1.0):
            raise AssertionError("document-context requests were executed serially")
        index = int(window.document_id.rsplit("-", 1)[1])
        return EntityExtractionOutcome(
            entities=[
                EntityMention(
                    id=f"mention-{index}",
                    name=f"Paper {index}",
                    type="PAPER",
                    description=f"Paper {index}",
                    source_chunk_id=window.chunks[0].id,
                    quote=f"Paper {index}",
                    is_document_context=True,
                )
            ],
            trace={"window_index": index},
        )

    monkeypatch.setattr(  # type: ignore[attr-defined]
        LlmEntityExtractor,
        "extract_document_context_entities",
        extract_context,
    )

    entities, trace = pipeline._extract_document_context_entities(chunks)

    assert [entity.name for entity in entities] == ["Paper 0", "Paper 1", "Paper 2"]
    assert [event["window_index"] for event in trace] == [0, 1, 2]


def test_all_document_context_failures_are_reported_as_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "job": {"job_id": "failed-context", "graph_id": "graph"},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 16},
            "ontology": {"profile_path": "data/deep_learning_papers/ontology.yaml"},
            "graph": {"extraction_parallelism": 3},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )
    pipeline = KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
    )

    def fail_context(*_args: object, **_kwargs: object) -> EntityExtractionOutcome:
        raise ValueError("malformed optional context response")

    monkeypatch.setattr(
        LlmEntityExtractor,
        "extract_document_context_entities",
        fail_context,
    )

    entities, trace = pipeline._extract_document_context_entities(
        [_context_chunk(index) for index in range(3)]
    )

    assert entities == []
    assert [event["stage"] for event in trace[:-1]] == ["document_context_error"] * 3
    assert trace[-1] == {
        "stage": "document_context_summary",
        "status": "degraded",
        "windows": 3,
        "failed_windows": 3,
        "entities": 0,
    }


def _context_chunk(index: int) -> Chunk:
    """Build one deterministic prefix chunk for an independent source document."""

    content = f"Paper {index} presents a method."
    return Chunk(
        id=f"chunk-{index}",
        graph_id="graph",
        file_id=f"file-{index}",
        document_id=f"document-{index}",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=6,
        content_hash=sha256_hex(content),
    )
