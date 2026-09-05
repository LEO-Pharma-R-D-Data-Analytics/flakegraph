"""Behavior tests for distributed planning and stage execution boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Event
from typing import Any

import pytest
import yaml

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.application.distributed_planner import (
    DistributedRunPlanner,
    _source_staging_workers,
    distributed_processing_compatibility_config,
    distributed_processing_config_digest,
)
from kg_processor.application.distributed_worker import (
    DistributedWorker,
    _prepared_projection_for_extracted_artifact,
)
from kg_processor.application.pipeline import _reject_fully_discarded_window
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    ArtifactKind,
    ArtifactRef,
    RunDefinition,
    RunOverview,
    RunSnapshot,
    RunStatus,
    RunSummary,
    StoredArtifact,
    TaskCount,
    TaskDefinition,
    TaskLease,
    TaskProgress,
    TaskSnapshot,
    TaskStage,
    TaskStatus,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.extraction import EntityMention, ExtractionObservations
from kg_processor.domain.graph import Chunk, GraphWriteBatch
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.domain.stages import (
    DocumentContextShard,
    DocumentEntityInventoryShard,
    EntityWindowShard,
    ExtractedDocumentShard,
    ExtractionWindowShard,
    PreparedDocumentShard,
    RelationWindowShard,
    combine_extracted_shards,
)
from kg_processor.ports.task_store import TaskStoreUnavailableError


class MemoryDistributedStore:
    """Minimal deterministic port implementation used to test application behavior."""

    def __init__(self) -> None:
        self.run: RunDefinition | None = None
        self.tasks: list[TaskDefinition] = []
        self.artifacts: dict[str, StoredArtifact] = {}
        self.cancelled_run_ids: list[str] = []
        self.lease: TaskLease | None = None
        self.completed: list[tuple[str, str, list[str], list[TaskDefinition], str | None]] = []
        self.failed: list[tuple[str, str, dict[str, Any], timedelta]] = []
        self.heartbeats: list[tuple[str, str, timedelta]] = []
        self.progress_updates: list[tuple[str, str, TaskProgress]] = []
        self.initial_task_streams = 0

    def initialize(self) -> None:
        """Satisfy the idempotent setup port for this already-initialized fake."""

    def list_runs(self, limit: int = 100) -> list[RunOverview]:
        """Return no history because these tests query the active run directly."""

        _ = limit
        return []

    def retry_run(self, run_id: str) -> None:
        """Record no state because retry transitions are covered by PostgreSQL tests."""

        assert self.run is not None and self.run.id == run_id

    def delete_run_artifacts(self, run_id: str) -> int:
        """Delete and count this run's in-memory artifacts."""

        selected = [key for key, value in self.artifacts.items() if value.ref.run_id == run_id]
        for artifact_id in selected:
            del self.artifacts[artifact_id]
        return len(selected)

    def create_run(self, run: RunDefinition) -> None:
        self.run = run

    def add_tasks(self, run_id: str, tasks: list[TaskDefinition]) -> None:
        assert self.run is not None and self.run.id == run_id
        self.tasks = tasks

    def add_initial_tasks(self, run_id: str, tasks: Iterable[TaskDefinition]) -> None:
        """Consume the production planner's constant-memory initial-plan stream."""

        assert self.run is not None and self.run.id == run_id
        self.initial_task_streams += 1
        self.tasks = list(tasks)

    def activate_run(self, run_id: str) -> None:
        assert self.run is not None and self.run.id == run_id
        self.run = self.run.model_copy(update={"status": RunStatus.QUEUED})

    def cancel_run(self, run_id: str) -> None:
        self.cancelled_run_ids.append(run_id)

    def get_run(self, run_id: str) -> RunSnapshot:
        if self.run is None or self.run.id != run_id:
            raise KeyError(run_id)
        now = datetime.now(UTC)
        return RunSnapshot(
            run=self.run,
            tasks=[
                TaskSnapshot(task=task, status=TaskStatus.QUEUED, attempts=0) for task in self.tasks
            ],
            created_at=now,
            updated_at=now,
        )

    def get_run_summary(self, run_id: str) -> RunSummary:
        """Aggregate the fake's task state like the production status query."""

        if self.run is None or self.run.id != run_id:
            raise KeyError(run_id)
        now = datetime.now(UTC)
        counts: dict[tuple[TaskStage, TaskStatus], int] = {}
        for task in self.tasks:
            key = (task.stage, TaskStatus.QUEUED)
            counts[key] = counts.get(key, 0) + 1
        return RunSummary(
            run=self.run,
            task_counts=[
                TaskCount(stage=stage, status=status, count=count)
                for (stage, status), count in sorted(
                    counts.items(), key=lambda item: (item[0][0].value, item[0][1].value)
                )
            ],
            total_tasks=len(self.tasks),
            created_at=now,
            updated_at=now,
        )

    def put(
        self,
        run_id: str,
        kind: ArtifactKind,
        payload: bytes,
        media_type: str,
        metadata: dict[str, Any] | None = None,
        identity_key: str | None = None,
    ) -> ArtifactRef:
        checksum = sha256_hex(payload)
        artifact_id = (
            stable_id("artifact", run_id, kind.value, checksum, identity_key)
            if identity_key is not None
            else stable_id("artifact", run_id, kind.value, checksum)
        )
        ref = ArtifactRef(
            id=artifact_id,
            run_id=run_id,
            kind=kind,
            media_type=media_type,
            checksum=checksum,
            size_bytes=len(payload),
            metadata=metadata or {},
        )
        self.artifacts[artifact_id] = StoredArtifact(ref=ref, payload=payload)
        return ref

    def get(self, artifact_id: str) -> StoredArtifact:
        return self.artifacts[artifact_id]

    def get_many(self, artifact_ids: list[str]) -> list[StoredArtifact]:
        """Mirror ordered batch reads so worker tests exercise the optimized capability."""

        return [self.artifacts[artifact_id] for artifact_id in artifact_ids]

    def get_run_artifacts(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
    ) -> list[StoredArtifact]:
        """Return run outputs in stable identity order for local finalization tests."""

        return sorted(
            (
                artifact
                for artifact in self.artifacts.values()
                if artifact.ref.run_id == run_id and artifact.ref.kind in kinds
            ),
            key=lambda artifact: artifact.ref.id,
        )

    def claim_task(
        self,
        worker_id: str,
        stages: set[TaskStage],
        lease_duration: timedelta,
        expected_config_digest: str | None = None,
        skip_dependency_outputs_for_stages: set[TaskStage] | None = None,
    ) -> TaskLease | None:
        _ = lease_duration
        if (
            expected_config_digest is not None
            and self.run is not None
            and self.run.config_digest != expected_config_digest
        ):
            return None
        if self.lease is None or self.lease.worker_id != worker_id:
            return None
        if self.lease.task.stage not in stages:
            return None
        lease, self.lease = self.lease, None
        if lease.task.stage in (skip_dependency_outputs_for_stages or set()):
            lease = lease.model_copy(
                update={
                    "task": lease.task.model_copy(update={"dependency_ids": []}),
                    "dependency_outputs": {},
                }
            )
        return lease

    def heartbeat(self, task_id: str, worker_id: str, lease_duration: timedelta) -> None:
        self.heartbeats.append((task_id, worker_id, lease_duration))

    def report_task_progress(
        self,
        task_id: str,
        worker_id: str,
        progress: TaskProgress,
    ) -> None:
        """Record bounded in-task updates for finalization progress assertions."""

        self.progress_updates.append((task_id, worker_id, progress))

    def complete_task(
        self,
        task_id: str,
        worker_id: str,
        output_artifact_ids: list[str],
        follow_up_tasks: list[TaskDefinition] | None = None,
        barrier_task_id: str | None = None,
    ) -> None:
        self.completed.append(
            (task_id, worker_id, output_artifact_ids, follow_up_tasks or [], barrier_task_id)
        )

    def fail_task(
        self,
        task_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
    ) -> None:
        self.failed.append((task_id, worker_id, error, retry_delay))


class RecordingPipeline:
    """Record stage inputs while returning small valid domain artifacts."""

    def __init__(self) -> None:
        self.prepared_files: list[str] = []
        self.context_file_ids: list[list[str]] = []
        self.extracted_file_ids: list[list[str]] = []
        self.finalized_file_ids: list[list[str]] = []

    def prepare_documents(self, files: list[Any]) -> PreparedDocumentShard:
        self.prepared_files.extend(str(file.path) for file in files)
        chunks = [
            Chunk(
                id=f"chunk-{file.id}",
                file_id=file.id,
                document_id=file.id,
                page_number=1,
                chunk_index=0,
                content="public source",
                start_offset=0,
                end_offset=13,
                token_count=2,
                content_hash=sha256_hex("public source"),
            )
            for file in files
        ]
        return PreparedDocumentShard(
            file_ids=[file.id for file in files],
            files_seen=len(files),
            documents_processed=len(files),
            chunks=chunks,
        )

    def extract_document_context(self, prepared: PreparedDocumentShard) -> PreparedDocumentShard:
        """Record context-stage input and preserve the serializable shard."""

        self.context_file_ids.append(prepared.file_ids)
        return prepared

    def extract_window_entities(self, prepared: PreparedDocumentShard) -> EntityWindowShard:
        """Record a first-phase extraction without copying prepared metadata."""

        self.extracted_file_ids.append(prepared.file_ids)
        return EntityWindowShard(
            file_ids=prepared.file_ids,
            chunk_ids=[chunk.id for chunk in prepared.chunks],
        )

    def extract_window_relations(
        self,
        prepared: PreparedDocumentShard,
        document_entities: list[EntityMention],
    ) -> RelationWindowShard:
        """Record a second-phase extraction while accepting the shared inventory."""

        _ = document_entities
        self.extracted_file_ids.append(prepared.file_ids)
        return RelationWindowShard(
            file_ids=prepared.file_ids,
            chunk_ids=[chunk.id for chunk in prepared.chunks],
        )

    def finalize_document_shards(
        self,
        shards: list[ExtractedDocumentShard],
        *,
        write: bool = True,
    ) -> GraphWriteBatch:
        assert write is True
        self.finalized_file_ids.append(
            [file_id for shard in shards for file_id in shard.prepared.file_ids]
        )
        return _empty_batch("test-graph")


def test_planner_persists_sources_and_defers_window_fan_out(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.md"
    first.write_text("Alpha", encoding="utf-8")
    second.write_text("Beta", encoding="utf-8")
    settings = _settings(tmp_path)
    store = MemoryDistributedStore()

    snapshot = DistributedRunPlanner(
        settings,
        LocalFileSource(tmp_path),
        store,
        store,
    ).submit("run-test")

    assert snapshot.run.status == RunStatus.QUEUED
    assert store.initial_task_streams == 1
    assert len(store.artifacts) == 2
    assert len(store.tasks) == 3
    prepare = [task for task in store.tasks if task.stage == TaskStage.PREPARE_DOCUMENT]
    final = [task for task in store.tasks if task.stage == TaskStage.FINALIZE_GRAPH]
    assert len(prepare) == 2
    assert len(final) == 1
    assert final[0].dependency_ids == []
    assert final[0].payload == {"output": {"provider": "local_artifacts"}}
    assert "database_url" not in snapshot.run.config
    llm_config = snapshot.run.config["llm"]
    assert isinstance(llm_config, dict)
    assert llm_config["api_key"] == "***"


def test_planner_places_only_snowflake_secret_references_on_final_task(tmp_path: Path) -> None:
    """Route per-run fleet output without persisting a password in the task queue."""

    document = tmp_path / "document.txt"
    document.write_text("Alpha", encoding="utf-8")
    settings = _settings(tmp_path)
    settings.writer.provider = "snowflake_bulk"
    settings.snowflake.account = "account"
    settings.snowflake.user = "user"
    settings.snowflake.database = "DB"
    settings.snowflake.schema_name = "GRAPH"
    settings.snowflake.warehouse = "WH"
    settings.snowflake.bulk_stage = "@DB.GRAPH.LOAD_STAGE"
    settings.snowflake.password = "must-not-be-persisted"
    settings.snowflake.password_environment_variable = "KG_SNOWFLAKE_PASSWORD"
    store = MemoryDistributedStore()

    DistributedRunPlanner(settings, LocalFileSource(tmp_path), store, store).submit("run-snowflake")

    final = next(task for task in store.tasks if task.stage == TaskStage.FINALIZE_GRAPH)
    serialized = str(final.payload)
    output = final.payload["output"]
    assert isinstance(output, dict)
    target = output["snowflake"]
    assert isinstance(target, dict)
    assert target["database"] == "DB"
    assert target["password_environment_variable"] == "KG_SNOWFLAKE_PASSWORD"
    assert "must-not-be-persisted" not in serialized


def test_planner_stages_sources_concurrently_in_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use object-store concurrency without making task order timing-dependent."""

    settings = _settings(tmp_path)
    store = MemoryDistributedStore()
    planner = DistributedRunPlanner(settings, LocalFileSource(tmp_path), store, store)
    rendezvous = Barrier(2, timeout=2)
    files: list[InputFile] = []
    for name in ("first.txt", "second.txt"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files.extend(LocalFileSource(path).list_files())

    def store_source(run_id: str, file: InputFile) -> ArtifactRef:
        """Require both staging threads to overlap, then return a stable sentinel."""

        rendezvous.wait()
        payload = file.path.read_bytes()
        return ArtifactRef(
            id=f"artifact-{file.path.stem}",
            run_id=run_id,
            kind=ArtifactKind.SOURCE_DOCUMENT,
            media_type=file.mime_type,
            checksum=sha256_hex(payload),
            size_bytes=len(payload),
        )

    monkeypatch.setattr(planner, "_store_source", store_source)

    refs = planner._store_sources("run-test", files)

    assert [ref.id for ref in refs] == ["artifact-first", "artifact-second"]


def test_planner_refills_source_uploads_behind_a_slow_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep upload slots occupied without retaining unbounded source payloads."""

    settings = _settings(tmp_path)
    store = MemoryDistributedStore()
    planner = DistributedRunPlanner(settings, LocalFileSource(tmp_path), store, store)
    third_started = Event()
    files: list[InputFile] = []
    for index in range(3):
        path = tmp_path / f"{index}.txt"
        path.write_text(str(index), encoding="utf-8")
        files.extend(LocalFileSource(path).list_files())

    def store_source(run_id: str, file: InputFile) -> ArtifactRef:
        """Block the first upload until completed work replenishes the pool."""

        if file.path.stem == "0":
            assert third_started.wait(timeout=2)
        elif file.path.stem == "2":
            third_started.set()
        payload = file.path.read_bytes()
        return ArtifactRef(
            id=f"artifact-{file.path.stem}",
            run_id=run_id,
            kind=ArtifactKind.SOURCE_DOCUMENT,
            media_type=file.mime_type,
            checksum=sha256_hex(payload),
            size_bytes=len(payload),
        )

    monkeypatch.setattr(planner, "_store_source", store_source)
    monkeypatch.setattr(
        "kg_processor.application.distributed_planner._source_staging_workers",
        lambda _files: 2,
    )

    staged = list(planner._iter_staged_sources("run-test", files))

    assert {file.id for file, _ref in staged} == {file.id for file in files}
    assert {ref.id for _file, ref in staged} == {
        "artifact-0",
        "artifact-1",
        "artifact-2",
    }
    assert third_started.is_set()


def test_source_staging_limits_memory_for_large_documents(tmp_path: Path) -> None:
    """Keep bounded source concurrency from multiplying complete-file buffers."""

    files = [
        InputFile(
            id=f"file-{index}",
            path=tmp_path / f"file-{index}.pdf",
            source_uri=f"file-{index}.pdf",
            checksum="checksum",
            mime_type="application/pdf",
            size_bytes=128 * 1024 * 1024,
        )
        for index in range(20)
    ]

    assert _source_staging_workers(files) == 4
    assert (
        _source_staging_workers([file.model_copy(update={"size_bytes": 1024}) for file in files])
        == 16
    )


def test_planner_preserves_provenance_for_byte_identical_sources(tmp_path: Path) -> None:
    """Equal bytes at different paths must produce distinct tasks and artifacts."""

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("identical public text", encoding="utf-8")
    second.write_text("identical public text", encoding="utf-8")
    settings = _settings(tmp_path)
    store = MemoryDistributedStore()

    DistributedRunPlanner(settings, LocalFileSource(tmp_path), store, store).submit("identical-run")

    source_artifacts = [
        item for item in store.artifacts.values() if item.ref.kind == ArtifactKind.SOURCE_DOCUMENT
    ]
    preparation_tasks = [task for task in store.tasks if task.stage == TaskStage.PREPARE_DOCUMENT]
    assert len(source_artifacts) == 2
    assert len(preparation_tasks) == 2
    assert len({task.scope_id for task in preparation_tasks}) == 2
    assert {item.ref.metadata["input_file"]["filename"] for item in source_artifacts} == {
        "first.txt",
        "second.txt",
    }


def test_explicit_run_id_submit_is_idempotent_without_cancelling_live_run(
    tmp_path: Path,
) -> None:
    """Return an existing compatible run when a submit request is retried."""

    source = tmp_path / "source.txt"
    source.write_text("public text", encoding="utf-8")
    settings = _settings(tmp_path)
    store = MemoryDistributedStore()
    planner = DistributedRunPlanner(settings, LocalFileSource(tmp_path), store, store)

    first = planner.submit("stable-run")
    second = planner.submit("stable-run")

    assert second.run.id == first.run.id
    assert store.cancelled_run_ids == []


def test_worker_prepares_source_from_artifact_without_shared_input_mount(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = _worker_store(settings)
    source = b"public source"
    source_ref = store.put(
        "run-test",
        ArtifactKind.SOURCE_DOCUMENT,
        source,
        "text/plain",
        metadata={
            "input_file": {
                "id": "file-1",
                "source_uri": "memory://file-1",
                "checksum": sha256_hex(source),
                "mime_type": "text/plain",
                "size_bytes": len(source),
                "filename": "source.txt",
            }
        },
    )
    task = _task(TaskStage.PREPARE_DOCUMENT, payload={"source_artifact_id": source_ref.id})
    store.lease = _lease(task)
    pipeline = RecordingPipeline()

    result = _worker(settings, store, pipeline, TaskStage.PREPARE_DOCUMENT).process_one()

    assert result.succeeded is True
    assert len(store.completed) == 1
    output = store.get(store.completed[0][2][0])
    assert output.ref.kind == ArtifactKind.PREPARED_DOCUMENT
    prepared = PreparedDocumentShard.model_validate_json(output.payload)
    assert prepared.file_ids == ["file-1"]
    follow_ups = store.completed[0][3]
    assert len(follow_ups) == 1
    assert follow_ups[0].stage == TaskStage.EXTRACT_DOCUMENT_CONTEXT
    assert follow_ups[0].dependency_ids == [task.id]
    assert follow_ups[0].payload == {}
    assert store.completed[0][4] == stable_id("task", "run-test", TaskStage.FINALIZE_GRAPH.value)
    assert pipeline.prepared_files[0].endswith("source.txt")
    assert not Path(pipeline.prepared_files[0]).exists()


def test_context_stage_reuses_prepared_shard_and_fans_out_windows(tmp_path: Path) -> None:
    """One context task must precede all dynamically queueable body windows."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=[_chunk("chunk-1", "file-1")],
    )
    prepared_ref = store.put(
        "run-test",
        ArtifactKind.PREPARED_DOCUMENT,
        prepared.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
        dependency_ids=["prepare-1"],
    )
    store.lease = _lease(task, {"prepare-1": [prepared_ref.id]})
    pipeline = RecordingPipeline()

    result = _worker(
        settings,
        store,
        pipeline,
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
    ).process_one()

    assert result.succeeded is True
    assert pipeline.context_file_ids == [["file-1"]]
    output = store.get(store.completed[0][2][0])
    assert output.ref.kind == ArtifactKind.DOCUMENT_CONTEXT
    contextualized = DocumentContextShard.model_validate_json(output.payload)
    assert contextualized.file_ids == ["file-1"]
    follow_ups = store.completed[0][3]
    assert len(follow_ups) == 2
    window_task = next(item for item in follow_ups if item.stage == TaskStage.EXTRACT_ENTITY_WINDOW)
    compact_task = next(
        item for item in follow_ups if item.stage == TaskStage.COMPACT_ENTITY_INVENTORY
    )
    assert window_task.dependency_ids == [task.id]
    assert set(compact_task.dependency_ids) == {task.id, window_task.id}
    assert compact_task.payload["prepared_artifact_id"] == prepared_ref.id
    window_ref = store.get(window_task.payload["window_artifact_id"])
    assert window_ref.ref.kind == ArtifactKind.EXTRACTION_WINDOW
    window = ExtractionWindowShard.model_validate_json(window_ref.payload)
    assert [chunk.id for chunk in window.chunks] == ["chunk-1"]
    assert store.completed[0][4] == stable_id("task", "run-test", TaskStage.FINALIZE_GRAPH.value)


def test_context_stage_limits_task_packs_to_provider_parallelism(tmp_path: Path) -> None:
    """Expose every provider wave to queue work stealing near the corpus tail."""

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "graph": settings.graph.model_copy(
                update={
                    "extraction_parallelism": 2,
                    "extraction_window_tokens": 100,
                    "max_chunks_per_llm_call": 1,
                }
            )
        }
    )
    chunks = [
        _chunk(f"chunk-{index}", "file-1").model_copy(
            update={"chunk_index": index, "content_hash": sha256_hex(f"chunk-{index}")}
        )
        for index in range(5)
    ]
    store = _worker_store(settings)
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=chunks,
    )
    prepared_ref = store.put(
        "run-test",
        ArtifactKind.PREPARED_DOCUMENT,
        prepared.model_dump_json().encode(),
        "application/json",
    )
    task = _task(TaskStage.EXTRACT_DOCUMENT_CONTEXT, dependency_ids=["prepare-1"])
    store.lease = _lease(task, {"prepare-1": [prepared_ref.id]})

    result = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
    ).process_one()

    assert result.succeeded is True
    window_tasks = [
        item for item in store.completed[0][3] if item.stage == TaskStage.EXTRACT_ENTITY_WINDOW
    ]
    shards = [
        ExtractionWindowShard.model_validate_json(
            store.get(item.payload["window_artifact_id"]).payload
        )
        for item in window_tasks
    ]
    assert [shard.logical_window_count for shard in shards] == [2, 2, 1]
    assert [len(shard.chunks) for shard in shards] == [2, 2, 1]


def test_context_stage_skips_reference_window_without_dropping_later_content(
    tmp_path: Path,
) -> None:
    """Keep an appendix after references while avoiding a bibliography LLM call."""

    settings = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "graph": settings.graph.model_copy(
                update={"extraction_window_tokens": 100, "max_chunks_per_llm_call": 1}
            )
        }
    )
    body = _chunk("body", "file-1")
    references = body.model_copy(
        update={
            "id": "references",
            "page_number": 2,
            "chunk_index": 1,
            "content": (
                "References\n[1] A. Author. First paper. 2020.\n[2] B. Author. Second paper. 2021."
            ),
            "content_hash": sha256_hex("references"),
        }
    )
    appendix = body.model_copy(
        update={
            "id": "appendix",
            "page_number": 3,
            "chunk_index": 2,
            "content": "Appendix A reports additional ablation experiments and results.",
            "content_hash": sha256_hex("appendix"),
        }
    )
    store = _worker_store(settings)
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=[body, references, appendix],
    )
    prepared_ref = store.put(
        "run-test",
        ArtifactKind.PREPARED_DOCUMENT,
        prepared.model_dump_json().encode(),
        "application/json",
    )
    task = _task(TaskStage.EXTRACT_DOCUMENT_CONTEXT, dependency_ids=["prepare-1"])
    store.lease = _lease(task, {"prepare-1": [prepared_ref.id]})

    result = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.EXTRACT_DOCUMENT_CONTEXT,
    ).process_one()

    assert result.succeeded is True
    follow_ups = [
        item for item in store.completed[0][3] if item.stage == TaskStage.EXTRACT_ENTITY_WINDOW
    ]
    assert [
        [
            chunk.id
            for chunk in ExtractionWindowShard.model_validate_json(
                store.get(follow_up.payload["window_artifact_id"]).payload
            ).chunks
        ]
        for follow_up in follow_ups
    ] == [
        ["body", "appendix"],
    ]


def test_planner_cancels_partial_run_when_discovered_source_changes(tmp_path: Path) -> None:
    """A stale source checksum must never become a worker artifact."""

    path = tmp_path / "changed.txt"
    path.write_text("current bytes", encoding="utf-8")
    source = _StaticFileSource(
        InputFile(
            id="changed-file",
            path=path,
            source_uri=path.as_uri(),
            checksum="0" * 64,
            mime_type="text/plain",
            size_bytes=path.stat().st_size,
        )
    )
    settings = _settings(tmp_path)
    store = MemoryDistributedStore()

    with pytest.raises(ValueError, match="changed after discovery"):
        DistributedRunPlanner(settings, source, store, store).submit("changed-run")

    assert store.cancelled_run_ids == ["changed-run"]
    assert store.tasks == []


def test_worker_extracts_prepared_dependency_into_portable_artifact(tmp_path: Path) -> None:
    """Extraction workers must consume and emit only serialized stage contracts."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    context = DocumentContextShard(
        file_ids=["file-1"],
    )
    context_ref = store.put(
        "run-test",
        ArtifactKind.DOCUMENT_CONTEXT,
        context.model_dump_json().encode(),
        "application/json",
    )
    window = ExtractionWindowShard(
        file_ids=["file-1"],
        chunks=[_chunk("chunk-1", "file-1")],
    )
    window_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTION_WINDOW,
        window.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.EXTRACT_ENTITY_WINDOW,
        payload={"window_artifact_id": window_ref.id},
        dependency_ids=["prepare-1"],
    )
    store.lease = _lease(task, {"prepare-1": [context_ref.id]})
    pipeline = RecordingPipeline()

    result = _worker(settings, store, pipeline, TaskStage.EXTRACT_ENTITY_WINDOW).process_one()

    assert result.succeeded is True
    assert pipeline.extracted_file_ids == [["file-1"]]
    output = store.get(store.completed[0][2][0])
    assert output.ref.kind == ArtifactKind.EXTRACTED_ENTITY_WINDOW
    extracted = EntityWindowShard.model_validate_json(output.payload)
    assert extracted.chunk_ids == ["chunk-1"]


def test_entity_inventory_barrier_fans_out_relation_windows(tmp_path: Path) -> None:
    """Share all document entities with a second independently queueable wave."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    context_entity = EntityMention(
        id="paper",
        name="A Source Paper",
        type="PAPER",
        description="The focal paper.",
        source_chunk_id="front-matter",
        quote="A Source Paper",
    )
    body_entity = context_entity.model_copy(
        update={
            "id": "method",
            "name": "A Method",
            "type": "METHOD",
            "description": "The method described by the paper.",
            "source_chunk_id": "chunk-1",
            "quote": "A Method",
        }
    )
    context = DocumentContextShard(
        file_ids=["file-1"],
        document_context_entities=[context_entity],
    )
    context_ref = store.put(
        "run-test",
        ArtifactKind.DOCUMENT_CONTEXT,
        context.model_dump_json().encode(),
        "application/json",
    )
    entity_window = EntityWindowShard(
        file_ids=["file-1"],
        chunk_ids=["chunk-1"],
        entities=[body_entity],
    )
    entity_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTED_ENTITY_WINDOW,
        entity_window.model_dump_json().encode(),
        "application/json",
    )
    window = ExtractionWindowShard(
        file_ids=["file-1"],
        chunks=[_chunk("chunk-1", "file-1")],
    )
    window_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTION_WINDOW,
        window.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.COMPACT_ENTITY_INVENTORY,
        payload={
            "prepared_artifact_id": "prepared-ref",
            "windows": [{"window_id": "window-1", "artifact_id": window_ref.id}],
        },
        dependency_ids=["context", "entity"],
    )
    store.lease = _lease(
        task,
        {"context": [context_ref.id], "entity": [entity_ref.id]},
    )

    result = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.COMPACT_ENTITY_INVENTORY,
    ).process_one()

    assert result.succeeded is True
    inventory_output = store.get(store.completed[0][2][0])
    assert inventory_output.ref.kind == ArtifactKind.DOCUMENT_ENTITY_INVENTORY
    inventory = DocumentEntityInventoryShard.model_validate_json(inventory_output.payload)
    assert {entity.id for entity in inventory.entities} == {"paper", "method"}
    follow_ups = store.completed[0][3]
    relation = next(item for item in follow_ups if item.stage == TaskStage.EXTRACT_RELATION_WINDOW)
    compact = next(item for item in follow_ups if item.stage == TaskStage.COMPACT_DOCUMENT)
    assert relation.payload["window_artifact_id"] == window_ref.id
    assert relation.dependency_ids == [task.id]
    assert set(compact.dependency_ids) == {task.id, relation.id}


def test_worker_compacts_window_results_into_one_complete_document(tmp_path: Path) -> None:
    """Keep every prepared chunk while storing context and observations only once."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    body = _chunk("body", "file-1")
    references = body.model_copy(
        update={"id": "references", "chunk_index": 1, "content": "References"}
    )
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        chunks=[body, references],
        trace=[{"stage": "ocr"}],
    )
    prepared_ref = store.put(
        "run-test",
        ArtifactKind.PREPARED_DOCUMENT,
        prepared.model_dump_json().encode(),
        "application/json",
    )
    context = DocumentContextShard(
        file_ids=["file-1"],
        trace=[{"stage": "ocr"}, {"stage": "document_context"}],
    )
    inventory = DocumentEntityInventoryShard(
        file_ids=["file-1"],
        trace=context.trace,
        chunk_count=1,
        window_count=1,
    )
    inventory_ref = store.put(
        "run-test",
        ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
        inventory.model_dump_json().encode(),
        "application/json",
    )
    relation_window = RelationWindowShard(
        file_ids=["file-1"],
        chunk_ids=[body.id],
        trace=[{"stage": "graph_extraction", "window": "body"}],
    )
    window_ref = store.put(
        "run-test",
        ArtifactKind.EXTRACTED_RELATION_WINDOW,
        relation_window.model_dump_json().encode(),
        "application/json",
    )
    task = _task(
        TaskStage.COMPACT_DOCUMENT,
        payload={"prepared_artifact_id": prepared_ref.id},
        dependency_ids=["context", "window"],
    )
    store.lease = _lease(
        task,
        {"context": [inventory_ref.id], "window": [window_ref.id]},
    )

    result = _worker(settings, store, RecordingPipeline(), TaskStage.COMPACT_DOCUMENT).process_one()

    assert result.succeeded is True
    output = store.get(store.completed[0][2][0])
    assert output.ref.kind == ArtifactKind.EXTRACTED_DOCUMENT
    compacted = ExtractedDocumentShard.model_validate_json(output.payload)
    assert [chunk.id for chunk in compacted.prepared.chunks] == ["body", "references"]
    assert compacted.prepared.trace == prepared.trace
    assert compacted.observations.chunk_count == 1
    assert compacted.observations.window_count == 1
    assert compacted.trace.count({"stage": "document_context"}) == 1


def test_combined_observations_include_compacted_document_context() -> None:
    """Treat a focal context mention as finalization input without a body duplicate."""

    context = EntityMention(
        id="paper-context",
        name="A Scalable Graph Method",
        type="PAPER",
        description="The focal source paper.",
        source_chunk_id="front-matter",
        quote="A Scalable Graph Method",
        is_document_context=True,
        contextual_surfaces=["we", "this paper"],
    )
    shard = ExtractedDocumentShard(
        prepared=PreparedDocumentShard(
            file_ids=["file-1"],
            files_seen=1,
            documents_processed=1,
            document_context_entities=[context],
        ),
        observations=ExtractionObservations(chunk_count=0, window_count=0),
    )

    combined = combine_extracted_shards([shard])

    assert combined.entities == [context]


def test_spark_extracted_artifact_does_not_duplicate_prepared_corpus_rows() -> None:
    """Keep Spark observations compact while preserving local self-contained shards."""

    chunk = Chunk(
        id="chunk-1",
        file_id="file-1",
        document_id="document-1",
        page_number=1,
        chunk_index=0,
        content="A substantial prepared source passage.",
        start_offset=0,
        end_offset=38,
        token_count=5,
        content_hash="hash",
    )
    prepared = PreparedDocumentShard(
        file_ids=["file-1"],
        files_seen=1,
        documents_processed=1,
        document_rows=[{"id": "document-1"}],
        page_rows=[{"id": "page-1", "text": chunk.content}],
        block_rows=[{"id": "block-1", "text": chunk.content}],
        asset_rows=[{"id": "asset-1", "uri": "memory://asset"}],
        chunks=[chunk],
        trace=[{"stage": "ocr"}],
    )

    projected = _prepared_projection_for_extracted_artifact(
        prepared,
        spark_finalization=True,
    )

    assert projected.file_ids == prepared.file_ids
    assert projected.files_seen == prepared.files_seen
    assert projected.documents_processed == prepared.documents_processed
    assert projected.document_rows == []
    assert projected.page_rows == []
    assert projected.block_rows == []
    assert projected.asset_rows == []
    assert projected.chunks == []
    assert projected.trace == []
    assert (
        _prepared_projection_for_extracted_artifact(
            prepared,
            spark_finalization=False,
        )
        is prepared
    )


def test_worker_retries_transient_coordination_outage_without_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an idle worker alive when its durable queue is briefly unavailable."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    claim_calls = 0

    def claim_task(
        worker_id: str,
        stages: set[TaskStage],
        lease_duration: timedelta,
        expected_config_digest: str | None = None,
        skip_dependency_outputs_for_stages: set[TaskStage] | None = None,
    ) -> TaskLease | None:
        """Fail once like a DNS outage, then return an ordinary empty poll."""

        nonlocal claim_calls
        _ = (
            worker_id,
            stages,
            lease_duration,
            expected_config_digest,
            skip_dependency_outputs_for_stages,
        )
        claim_calls += 1
        if claim_calls == 1:
            raise TaskStoreUnavailableError("temporary coordination outage")
        return None

    monkeypatch.setattr(store, "claim_task", claim_task)
    monkeypatch.setattr("kg_processor.application.distributed_worker.sleep", lambda _: None)
    iterations: list[Any] = []

    completed = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.EXTRACT_ENTITY_WINDOW,
    ).run_until_stopped(
        lambda: claim_calls >= 2,
        iterations.append,
    )

    assert completed == 0
    assert claim_calls == 2
    assert len(iterations) == 2
    assert iterations[0].claimed is False
    assert iterations[0].succeeded is False
    assert iterations[0].error == {
        "error_type": "TaskStoreUnavailableError",
        "error_message": "temporary coordination outage",
    }
    assert iterations[1].claimed is False
    assert iterations[1].error is None


def test_worker_finalizes_all_dependency_shards_in_stable_order(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = _worker_store(settings)
    dependency_outputs: dict[str, list[str]] = {}
    for dependency_id, file_id in [("extract-b", "b"), ("extract-a", "a")]:
        prepared = PreparedDocumentShard(
            file_ids=[file_id],
            files_seen=1,
            documents_processed=1,
        )
        shard = ExtractedDocumentShard(
            prepared=prepared,
            observations=ExtractionObservations(chunk_count=0, window_count=0),
        )
        ref = store.put(
            "run-test",
            ArtifactKind.EXTRACTED_DOCUMENT,
            shard.model_dump_json().encode(),
            "application/json",
        )
        dependency_outputs[dependency_id] = [ref.id]
    task = _task(
        TaskStage.FINALIZE_GRAPH,
        dependency_ids=["extract-a", "extract-b"],
    )
    store.lease = _lease(task, dependency_outputs)
    pipeline = RecordingPipeline()

    result = _worker(settings, store, pipeline, TaskStage.FINALIZE_GRAPH).process_one()

    assert result.succeeded is True
    assert pipeline.finalized_file_ids == [["a", "b"]]
    graph = store.get(store.completed[0][2][0])
    assert graph.ref.kind == ArtifactKind.GRAPH_RESULT
    assert GraphWriteBatch.model_validate_json(graph.payload).graph_id == "test-graph"
    assert [update[2].phase for update in store.progress_updates] == [
        "read_artifacts",
        "read_artifacts",
        "materialize_inputs",
        "resolve_entities",
        "resolve_entities",
        "assemble_graph",
        "enrich_graph",
        "build_communities",
        "quality_checks",
        "write_tables",
        "publish_destination",
        "persist_manifest",
        "persist_manifest",
    ]
    assert store.progress_updates[-1][2].phase_index == 12
    assert store.progress_updates[-1][2].completed == 1


def test_task_progress_rejects_inconsistent_phase_and_inner_counters() -> None:
    """Keep malformed worker progress out of durable operator status payloads."""

    with pytest.raises(ValueError, match="phase_index"):
        TaskProgress(phase="write", phase_index=3, phase_total=2)
    with pytest.raises(ValueError, match="completed"):
        TaskProgress(
            phase="write",
            phase_index=1,
            phase_total=2,
            completed=3,
            total=2,
        )


def test_worker_does_not_claim_configuration_mismatch(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    store = _worker_store(settings)
    assert store.run is not None
    store.run = store.run.model_copy(update={"config_digest": "different"})
    store.lease = _lease(_task(TaskStage.EXTRACT_ENTITY_WINDOW))

    result = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.EXTRACT_ENTITY_WINDOW,
    ).process_one()

    assert result.claimed is False
    assert result.succeeded is None
    assert store.completed == []
    assert store.failed == []
    assert store.lease is not None


def test_worker_stays_alive_when_lost_lease_blocks_failure_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report a lost completion lease without crashing the long-running worker."""

    settings = _settings(tmp_path)
    store = _worker_store(settings)
    store.lease = _lease(_task(TaskStage.EXTRACT_ENTITY_WINDOW))
    monkeypatch.setattr(
        store,
        "complete_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("lease lost")),
    )
    monkeypatch.setattr(
        store,
        "fail_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("not owner")),
    )

    result = _worker(
        settings,
        store,
        RecordingPipeline(),
        TaskStage.EXTRACT_ENTITY_WINDOW,
    ).process_one()

    assert result.succeeded is False
    assert result.error is not None
    assert result.error["transition_error_type"] == "RuntimeError"


def test_the_same_ontology_matches_whether_it_is_inlined_or_mounted(tmp_path: Path) -> None:
    """Let a run carrying its ontology work with a fleet that mounts the same one.

    The application inlines the profile so a run describes itself wherever it
    executes, while a fleet worker mounts that ontology as a file. Judged by how
    the profile is carried rather than what it says, the two never match, and a
    run whose ontology no worker matches is never claimed: it sits queued with
    nothing on the page to explain it.
    """

    profile = {
        "name": "general",
        "mode": "closed",
        "entity_types": [{"name": "PERSON", "description": "A named human being."}],
        "relation_types": [
            {
                "name": "related_to",
                "description": "A stated association.",
                "source_types": ["PERSON"],
                "target_types": ["PERSON"],
            }
        ],
    }
    mounted_path = tmp_path / "ontology.yaml"
    mounted_path.write_text(yaml.safe_dump(profile, sort_keys=True), encoding="utf-8")

    settings = _settings(tmp_path)
    inlined = settings.model_copy(
        update={"ontology": settings.ontology.model_copy(update={"profile": profile})}
    )
    mounted = settings.model_copy(
        update={
            "ontology": settings.ontology.model_copy(
                update={"profile": None, "profile_path": mounted_path}
            )
        }
    )

    assert distributed_processing_config_digest(inlined) == (
        distributed_processing_config_digest(mounted)
    )

    # A different ontology is still a different run, whichever way it is carried.
    changed = dict(profile)
    changed["entity_types"] = [{"name": "PERSON", "description": "Someone else entirely."}]
    reworded = settings.model_copy(
        update={"ontology": settings.ontology.model_copy(update={"profile": changed})}
    )
    assert distributed_processing_config_digest(reworded) != (
        distributed_processing_config_digest(inlined)
    )


def test_an_ontology_that_says_nothing_matches_in_either_form(tmp_path: Path) -> None:
    """Agree about an empty ontology too, not only a populated one.

    An empty or comment-only file reads as nothing at all, while the application
    records the same absence as an empty mapping. Left to disagree, a fleet
    configured without an ontology is exactly as unclaimable as one whose
    ontology differs.
    """

    settings = _settings(tmp_path)
    for content in ("", "# an ontology that has not been written yet\n"):
        empty = tmp_path / "empty.yaml"
        empty.write_text(content, encoding="utf-8")
        mounted = settings.model_copy(
            update={
                "ontology": settings.ontology.model_copy(
                    update={"profile": None, "profile_path": empty}
                )
            }
        )
        inlined = settings.model_copy(
            update={"ontology": settings.ontology.model_copy(update={"profile": {}})}
        )

        assert distributed_processing_config_digest(mounted) == (
            distributed_processing_config_digest(inlined)
        ), content


def test_an_ontology_that_cannot_be_identified_stably_is_refused(tmp_path: Path) -> None:
    """Refuse a profile whose identity would differ between two readers.

    A YAML set becomes a Python set, whose text order follows hash
    randomization. Rendered rather than refused, two workers reading one file
    would disagree about it and neither would claim the run — silently, which is
    the failure this digest exists to prevent.
    """

    settings = _settings(tmp_path)
    unstable = settings.model_copy(
        update={"ontology": settings.ontology.model_copy(update={"profile": {"tags": {"a", "b"}}})}
    )

    with pytest.raises(TypeError, match="cannot be identified stably"):
        distributed_processing_config_digest(unstable)


def test_an_ontology_carrying_a_plain_yaml_date_can_still_be_identified(
    tmp_path: Path,
) -> None:
    """Hash the ontologies YAML permits, not only the ones JSON happens to share.

    YAML reads an unquoted date as a date rather than a string. Identifying the
    profile has to survive that: raised here, it stops every worker in the fleet
    before any of them can claim a task.
    """

    dated = tmp_path / "dated.yaml"
    dated.write_text(
        "name: general\nrevised: 2026-08-07\nentity_types: []\nrelation_types: []\n",
        encoding="utf-8",
    )
    settings = _settings(tmp_path)
    mounted = settings.model_copy(
        update={
            "ontology": settings.ontology.model_copy(
                update={"profile": None, "profile_path": dated}
            )
        }
    )

    assert distributed_processing_config_digest(mounted)


def test_processing_digest_ignores_worker_local_transport_settings(tmp_path: Path) -> None:
    """Allow equivalent workers to route to different replicas and local paths."""

    settings = _settings(tmp_path)
    relocated = settings.model_copy(
        update={
            "llm": settings.llm.model_copy(
                update={"endpoint": "http://node-local-model/v1", "api_key": "local-key"}
            ),
            "embedding": settings.embedding.model_copy(
                update={"endpoint": "http://node-local-embedding/v1", "batch_size": 64}
            ),
            "writer": settings.writer.model_copy(
                update={"output_path": tmp_path / "different-output"}
            ),
            "cache": settings.cache.model_copy(update={"path": tmp_path / "different-cache"}),
            "graph": settings.graph.model_copy(
                update={
                    "extraction_parallelism": 8,
                    "resolution_parallelism": 4,
                    "community_report_parallelism": 6,
                    "description_merge_parallelism": 3,
                }
            ),
        }
    )

    assert distributed_processing_config_digest(relocated) == (
        distributed_processing_config_digest(settings)
    )

    different_model = relocated.model_copy(
        update={"llm": relocated.llm.model_copy(update={"model": "different-model"})}
    )
    assert distributed_processing_config_digest(different_model) != (
        distributed_processing_config_digest(settings)
    )


def test_workers_reading_different_prompts_do_not_share_one_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate images by what they will actually ask the model, not by a label.

    Prompt text and the response contract are executable extraction policy. Held
    apart only by a revision constant someone has to remember to bump, an old and
    a new worker image compute the same digest during a rolling update, both
    claim tasks from the same run, and the graph is assembled from observations
    produced under two different sets of instructions.
    """

    settings = _settings(tmp_path)
    baseline = distributed_processing_config_digest(settings)
    compatibility = distributed_processing_compatibility_config(settings)
    assert compatibility["extraction_prompts"]
    assert compatibility["extraction_contracts"]

    monkeypatch.setattr(
        "kg_processor.application.distributed_planner.two_pass_prompt_fingerprints",
        lambda: {"entity_extraction": {"revision": "changed", "checksum": "0" * 64}},
    )
    assert distributed_processing_config_digest(settings) != baseline
    monkeypatch.undo()

    monkeypatch.setattr(
        "kg_processor.application.distributed_planner.extraction_contract_fingerprint",
        lambda: "0" * 64,
    )
    assert distributed_processing_config_digest(settings) != baseline


def test_processing_digest_ignores_graph_identity(tmp_path: Path) -> None:
    """Allow one immutable worker profile to process independently named graphs."""

    settings = _settings(tmp_path)
    renamed = settings.model_copy(
        update={"job": settings.job.model_copy(update={"graph_id": "another-graph"})}
    )

    assert distributed_processing_config_digest(renamed) == (
        distributed_processing_config_digest(settings)
    )


def test_processing_digest_identifies_ontology_content_not_mount_path(tmp_path: Path) -> None:
    """Accept equivalent ontology mounts and reject changed semantic content."""

    first_path = tmp_path / "first-ontology.yaml"
    second_path = tmp_path / "second-ontology.yaml"
    changed_path = tmp_path / "changed-ontology.yaml"
    first_path.write_text("entity_types: [PERSON]\n", encoding="utf-8")
    second_path.write_text("entity_types: [PERSON]\n", encoding="utf-8")
    changed_path.write_text("entity_types: [LOCATION]\n", encoding="utf-8")
    settings = _settings(tmp_path)

    def with_ontology(path: Path) -> Settings:
        """Return the common profile with one deployment-specific ontology mount."""

        return settings.model_copy(
            update={"ontology": settings.ontology.model_copy(update={"profile_path": path})}
        )

    assert distributed_processing_config_digest(with_ontology(first_path)) == (
        distributed_processing_config_digest(with_ontology(second_path))
    )
    assert distributed_processing_config_digest(with_ontology(first_path)) != (
        distributed_processing_config_digest(with_ontology(changed_path))
    )


def _settings(tmp_path: Path) -> Settings:
    """Return a complete lightweight processing profile for distributed tests."""

    return Settings.load(
        env={},
        overrides={
            "job": {"graph_id": "test-graph"},
            "files": {"input_path": str(tmp_path)},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake", "api_key": "test-secret"},
            "embedding": {"provider": "hash", "dimension": 8},
            "writer": {"provider": "local_artifacts", "output_path": str(tmp_path / "out")},
            "cache": {"provider": "none"},
            "distributed": {"database_url": "postgresql://user:secret@db/test"},
        },
    )


class _StaticFileSource:
    """Return one preconstructed file record for source-integrity tests."""

    def __init__(self, input_file: InputFile) -> None:
        self.input_file = input_file

    def list_files(self) -> list[InputFile]:
        """Return the record without recomputing its intentionally stale checksum."""

        return [self.input_file]


def _worker_store(settings: Settings) -> MemoryDistributedStore:
    """Create a store containing the run identity expected by a worker."""

    store = MemoryDistributedStore()
    store.run = RunDefinition(
        id="run-test",
        graph_id=settings.job.graph_id,
        config={},
        config_digest=distributed_processing_config_digest(settings),
        status=RunStatus.RUNNING,
    )
    return store


def _task(
    stage: TaskStage,
    *,
    payload: dict[str, Any] | None = None,
    dependency_ids: list[str] | None = None,
) -> TaskDefinition:
    """Build one stable task definition for a worker test."""

    return TaskDefinition(
        id=f"task-{stage.value}",
        run_id="run-test",
        stage=stage,
        scope_id="file-1",
        payload=payload or {},
        dependency_ids=dependency_ids or [],
    )


def _lease(
    task: TaskDefinition,
    dependency_outputs: dict[str, list[str]] | None = None,
) -> TaskLease:
    """Wrap a task in an owned lease that remains valid during a unit test."""

    return TaskLease(
        task=task,
        worker_id="worker-1",
        attempt=1,
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        dependency_outputs=dependency_outputs or {},
    )


def _worker(
    settings: Settings,
    store: MemoryDistributedStore,
    pipeline: RecordingPipeline,
    stage: TaskStage,
) -> DistributedWorker:
    """Construct a worker with one eligible stage and in-memory application ports."""

    return DistributedWorker(
        settings,
        "worker-1",
        {stage},
        pipeline,
        store,
        store,
    )


def _empty_batch(graph_id: str) -> GraphWriteBatch:
    """Return the smallest valid final graph artifact."""

    return GraphWriteBatch(
        graph_id=graph_id,
        documents=[],
        pages=[],
        chunks=[],
        nodes=[],
        edges=[],
        evidence=[],
        entity_sources=[],
        communities=[],
        community_findings=[],
        run_report={},
    )


def _chunk(chunk_id: str, file_id: str) -> Chunk:
    """Build one valid extraction chunk for window-worker tests."""

    return Chunk(
        id=chunk_id,
        file_id=file_id,
        document_id=file_id,
        page_number=1,
        chunk_index=0,
        content="public source",
        start_offset=0,
        end_offset=13,
        token_count=2,
        content_hash=sha256_hex("public source"),
    )


def test_a_window_with_nothing_to_extract_is_not_an_error() -> None:
    """A title slide or page of furniture genuinely contains no entities."""

    _reject_fully_discarded_window(
        [{"stage": "entity_extraction", "input_records": 0, "accepted_records": 0}],
        ["file_1"],
    )


def test_a_window_whose_every_record_was_discarded_fails_loudly() -> None:
    """Silence here is how a corpus loses a whole document.

    The model returned records and validation rejected all of them. Downstream
    that is indistinguishable from an empty window, so the run reports success
    while the document contributes nothing to the graph.
    """

    with pytest.raises(ValueError, match="discarded every record"):
        _reject_fully_discarded_window(
            [
                {
                    "stage": "entity_extraction",
                    "input_records": 3,
                    "accepted_records": 0,
                    "record_actions": {"ungrounded_quote": 3, "untrusted_alias": 11},
                }
            ],
            ["file_601352166d0d26e0ee2ac50bc2a82bc2"],
        )


def test_a_window_that_kept_some_records_is_accepted() -> None:
    """Partial rejection is normal validation, not a failed window."""

    _reject_fully_discarded_window(
        [
            {
                "stage": "entity_extraction",
                "input_records": 9,
                "accepted_records": 7,
                "record_actions": {"ungrounded_quote": 2},
            }
        ],
        ["file_1"],
    )
