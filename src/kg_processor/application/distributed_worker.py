"""Lease-driven worker service for provider-neutral distributed graph stages."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from itertools import batched
from pathlib import Path
from time import sleep
from typing import Any

from kg_processor.application.distributed_planner import distributed_processing_config_digest
from kg_processor.application.extraction_windows import build_extraction_windows, window_skip_reason
from kg_processor.application.lease_heartbeat import LeaseHeartbeat, heartbeat_interval_seconds
from kg_processor.application.progress import error_metadata
from kg_processor.application.spark_finalization import (
    FINALIZATION_PHASES,
    SparkFinalizationRequest,
    SparkGraphFinalizer,
    finalization_progress,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    ArtifactKind,
    ArtifactRef,
    InheritedDocuments,
    PublicationLease,
    StoredArtifact,
    TaskDefinition,
    TaskLease,
    TaskProgress,
    TaskStage,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    RelationObservation,
)
from kg_processor.domain.finalization import GraphDatasetManifest
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.ids import stable_id
from kg_processor.domain.stages import (
    DocumentContextShard,
    DocumentEntityInventoryShard,
    EntityWindowShard,
    ExtractedDocumentShard,
    ExtractionWindowShard,
    PreparedDocumentShard,
    RelationWindowShard,
)
from kg_processor.ports.artifact_store import ArtifactStore
from kg_processor.ports.distributed_pipeline import DistributedPipeline
from kg_processor.ports.graph_manifest_publisher import GraphManifestPublisher
from kg_processor.ports.task_store import TaskStore, TaskStoreUnavailableError

_JSON_MEDIA_TYPE = "application/json"
_GRAPH_MANIFEST_MEDIA_TYPE = "application/vnd.flakegraph.graph-manifest+json"
_MAX_LOGICAL_WINDOWS_PER_TASK = 4


@dataclass(frozen=True)
class WorkerIteration:
    """Summarize one polling iteration for CLI logging and focused tests."""

    claimed: bool
    task_id: str | None = None
    stage: TaskStage | None = None
    succeeded: bool | None = None
    output_artifact_ids: tuple[str, ...] = ()
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class TaskExecution:
    """Carry stage outputs and transactional queue fan-out to task completion.

    Preparation, context extraction, and entity compaction may create follow-up
    work. Keeping the completion command explicit lets the task store publish each
    complete dynamic subgraph atomically without stage implementations mutating
    coordination tables.
    """

    output_artifact_ids: tuple[str, ...]
    follow_up_tasks: tuple[TaskDefinition, ...] = ()
    barrier_task_id: str | None = None


logger = logging.getLogger(__name__)


_RUN_CONTEXT_CACHE_SIZE = 64


@dataclass(frozen=True)
class _RunContext:
    """What one run executes with: its settings and the pipeline running them."""

    settings: Settings
    pipeline: DistributedPipeline


def settings_for_run(base: Settings, run_config: dict[str, Any]) -> Settings:
    """Apply a run's stored ontology to a worker's own settings.

    The run's configuration is the snapshot the planner stored. Its ontology
    profile, when carried inline, is what the run's graph is to be built with;
    a profile the submitter only named by path is not readable here, so such a
    run executes with the worker's mounted profile. ``base`` is returned as is
    when the run changes nothing, so callers can tell the two apart.
    """

    ontology = run_config.get("ontology")
    profile = ontology.get("profile") if isinstance(ontology, dict) else None
    if not isinstance(profile, dict):
        return base
    if base.ontology.profile == profile and base.ontology.profile_path is None:
        return base
    return base.model_copy(
        update={
            "ontology": base.ontology.model_copy(update={"profile": profile, "profile_path": None})
        }
    )


class DistributedWorker:
    """Execute eligible durable tasks sequentially in one worker process.

    Horizontal concurrency comes from multiple worker processes or Kubernetes pods.
    A process handles one task at a time so provider clients, local model memory, and
    OCR subprocesses have explicit resource bounds. PostgreSQL row leases provide
    at-least-once recovery after process or node failure.
    """

    def __init__(
        self,
        settings: Settings,
        worker_id: str,
        stages: set[TaskStage],
        pipeline: DistributedPipeline,
        task_store: TaskStore,
        artifact_store: ArtifactStore,
        manifest_publisher: GraphManifestPublisher | None = None,
        pipeline_for: Callable[[Settings], DistributedPipeline] | None = None,
    ) -> None:
        """Validate worker identity and retain its injected application dependencies.

        ``pipeline_for`` builds the pipeline that executes one run's settings -
        the worker's own with that run's ontology applied - over the worker's
        adapters. Without it every run executes on ``pipeline`` as configured,
        which is what a test double wants.
        """

        if not worker_id.strip():
            raise ValueError("distributed worker id must not be blank")
        if not stages:
            raise ValueError("distributed worker requires at least one eligible stage")
        self.settings = settings
        self.worker_id = worker_id
        self.stages = stages
        self.pipeline = pipeline
        self.task_store = task_store
        self.artifact_store = artifact_store
        self.manifest_publisher = manifest_publisher
        self.lease_duration = timedelta(seconds=settings.distributed.lease_seconds)
        self.retry_delay = timedelta(seconds=settings.distributed.retry_delay_seconds)
        self._declared_configuration = False
        self._pipeline_for = pipeline_for
        self._run_contexts: dict[str, _RunContext] = {}

    def _declare_served_configuration(self) -> None:
        """Record what this fleet serves, once per process.

        Only a run carrying this digest can be claimed here, so the autoscaling
        signal needs to know the digest in order to stop asking for workers to do
        work no worker is able to take. Declared once rather than per poll: it
        cannot change without restarting the process.
        """

        if self._declared_configuration:
            return
        try:
            self.task_store.record_served_configuration(
                self.stages, distributed_processing_config_digest(self.settings)
            )
        except Exception:
            # Losing this costs an accurate demand signal, not the ability to
            # work, so a worker that cannot record it still claims tasks.
            logger.warning("Could not record the served configuration", exc_info=True)
        self._declared_configuration = True

    def process_one(self) -> WorkerIteration:
        """Claim and execute at most one task, returning immediately when none is ready."""

        publication = self._process_one_publication()
        if publication is not None:
            return publication
        self._declare_served_configuration()
        lease = self.task_store.claim_task(
            self.worker_id,
            self.stages,
            self.lease_duration,
            distributed_processing_config_digest(self.settings),
            ({TaskStage.FINALIZE_GRAPH} if _use_spark_finalization(self.settings) else None),
        )
        if lease is None:
            return WorkerIteration(claimed=False)
        try:
            with LeaseHeartbeat(
                lambda: self.task_store.heartbeat(
                    lease.task.id,
                    self.worker_id,
                    self.lease_duration,
                ),
                heartbeat_interval_seconds(self.settings.distributed.lease_seconds),
            ) as heartbeat:
                execution = self._execute(lease)
            heartbeat.raise_if_unhealthy()
            output_ids = list(execution.output_artifact_ids)
            self.task_store.complete_task(
                lease.task.id,
                self.worker_id,
                output_ids,
                list(execution.follow_up_tasks),
                execution.barrier_task_id,
            )
            return WorkerIteration(
                claimed=True,
                task_id=lease.task.id,
                stage=lease.task.stage,
                succeeded=True,
                output_artifact_ids=tuple(output_ids),
            )
        except Exception as exc:
            error = error_metadata(exc)
            try:
                self.task_store.fail_task(
                    lease.task.id,
                    self.worker_id,
                    error,
                    self.retry_delay,
                )
            except TaskStoreUnavailableError:
                # The outer polling loop owns coordination outages and retries
                # after backoff without pretending a durable failure was recorded.
                raise
            except Exception as transition_error:
                # Lost ownership means another worker or lease expiry already
                # controls the durable task state. Surface it in this iteration,
                # but keep the process alive rather than crashing the worker pod.
                error = {
                    **error,
                    "transition_error_type": type(transition_error).__name__,
                }
            return WorkerIteration(
                claimed=True,
                task_id=lease.task.id,
                stage=lease.task.stage,
                succeeded=False,
                error=error,
            )

    def _process_one_publication(self) -> WorkerIteration | None:
        """Drain one durable destination command before claiming more finalizer work."""

        store = self.task_store
        if self.manifest_publisher is None:
            return None
        lease = store.claim_publication(self.worker_id, self.lease_duration)
        if lease is None:
            return None
        try:
            with LeaseHeartbeat(
                lambda: store.heartbeat_publication(
                    lease.id,
                    self.worker_id,
                    self.lease_duration,
                ),
                heartbeat_interval_seconds(self.settings.distributed.lease_seconds),
            ) as heartbeat:

                def publish_with_health_check(publication: PublicationLease) -> None:
                    self._publish_manifest_lease(publication)
                    heartbeat.raise_if_unhealthy()

                store.publish_claimed(lease.id, self.worker_id, publish_with_health_check)
            return WorkerIteration(
                claimed=True,
                task_id=lease.id,
                succeeded=True,
                output_artifact_ids=(lease.artifact_id,),
            )
        except Exception as exc:
            error = error_metadata(exc)
            try:
                store.fail_publication(
                    lease.id,
                    self.worker_id,
                    error,
                    self.retry_delay,
                )
            except TaskStoreUnavailableError:
                raise
            except Exception as transition_error:
                error = {
                    **error,
                    "transition_error_type": type(transition_error).__name__,
                }
            return WorkerIteration(
                claimed=True,
                task_id=lease.id,
                succeeded=False,
                error=error,
            )

    def _publish_manifest_lease(self, lease: PublicationLease) -> None:
        """Load and publish the immutable manifest selected by the fenced outbox row."""

        if self.manifest_publisher is None:
            raise RuntimeError("publication lease requires a graph manifest publisher")
        stored = self.artifact_store.get(lease.artifact_id)
        if stored.ref.kind != ArtifactKind.GRAPH_RESULT:
            raise ValueError("publication artifact is not a finalized graph result")
        manifest = GraphDatasetManifest.model_validate_json(stored.payload)
        payload = {
            **lease.task_payload,
            "publication": {
                "id": lease.id,
                "generation": lease.generation,
                "attempt": lease.attempt,
            },
        }
        self.manifest_publisher.publish(manifest, payload)

    def run_until_stopped(
        self,
        stop_requested: Callable[[], bool],
        on_iteration: Callable[[WorkerIteration], None] | None = None,
    ) -> int:
        """Poll indefinitely until the caller requests a graceful process shutdown.

        Returning a count makes shutdown observable without coupling this service to
        signals, Typer, Kubernetes, or a particular logging implementation.
        """

        completed = 0
        while not stop_requested():
            try:
                iteration = self.process_one()
            except TaskStoreUnavailableError as exc:
                # No durable transition can be trusted while coordination is
                # unavailable. Keep the pod alive, surface the outage through
                # normal progress output, and reconnect after bounded backoff.
                iteration = WorkerIteration(
                    claimed=False,
                    succeeded=False,
                    error=error_metadata(exc),
                )
                if on_iteration is not None:
                    on_iteration(iteration)
                sleep(self.settings.distributed.retry_delay_seconds)
                continue
            if on_iteration is not None:
                on_iteration(iteration)
            if iteration.claimed:
                completed += 1
                continue
            sleep(self.settings.distributed.poll_interval_seconds)
        return completed

    def _execute(self, lease: TaskLease) -> TaskExecution:  # noqa: PLR0911
        """Dispatch one validated lease to its stage-specific implementation."""

        run = self._run_context(lease.task.run_id)
        if lease.task.stage == TaskStage.PREPARE_DOCUMENT:
            return self._prepare_document(lease, run)
        if lease.task.stage == TaskStage.EXTRACT_DOCUMENT_CONTEXT:
            return self._extract_document_context(lease, run)
        if lease.task.stage == TaskStage.EXTRACT_ENTITY_WINDOW:
            return TaskExecution((self._extract_entity_window(lease, run),))
        if lease.task.stage == TaskStage.COMPACT_ENTITY_INVENTORY:
            return self._compact_entity_inventory(lease)
        if lease.task.stage == TaskStage.EXTRACT_RELATION_WINDOW:
            return TaskExecution((self._extract_relation_window(lease, run),))
        if lease.task.stage == TaskStage.COMPACT_DOCUMENT:
            return TaskExecution((self._compact_document(lease),))
        if lease.task.stage == TaskStage.FINALIZE_GRAPH:
            return TaskExecution((self._finalize_graph(lease, run),))
        raise ValueError(f"unsupported distributed task stage: {lease.task.stage}")

    def _run_context(self, run_id: str) -> _RunContext:
        """The settings and pipeline one run executes with, resolved once per run.

        A run carries its ontology in its stored configuration. The worker's
        own profile is the default for a run that names none; a run that does
        is executed with its own, so two graphs built by one fleet can extract
        different vocabularies. Bounded: a worker sees a handful of runs.
        """

        cached = self._run_contexts.get(run_id)
        if cached is not None:
            return cached
        run_settings = settings_for_run(
            self.settings, self.task_store.get_run_summary(run_id).run.config
        )
        pipeline = (
            self.pipeline
            if run_settings is self.settings or self._pipeline_for is None
            else self._pipeline_for(run_settings)
        )
        context = _RunContext(settings=run_settings, pipeline=pipeline)
        if len(self._run_contexts) >= _RUN_CONTEXT_CACHE_SIZE:
            self._run_contexts.pop(next(iter(self._run_contexts)))
        self._run_contexts[run_id] = context
        return context

    def _prepare_document(self, lease: TaskLease, run: _RunContext) -> TaskExecution:
        """Run OCR/chunking and fan its discovered windows into the shared queue.

        The returned follow-up definitions are committed with this task's success.
        Consequently, finalization can never observe a completed preparation task
        without also waiting for every extraction window it produced.
        """

        artifact_id = lease.task.payload.get("source_artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("prepare_document task is missing source_artifact_id")
        source = self.artifact_store.get(artifact_id)
        if source.ref.kind != ArtifactKind.SOURCE_DOCUMENT:
            raise ValueError("prepare_document input is not a source document artifact")
        input_metadata = source.ref.metadata.get("input_file")
        if not isinstance(input_metadata, dict):
            raise ValueError("source document artifact is missing input_file metadata")
        filename = _safe_filename(input_metadata.get("filename"))
        with tempfile.TemporaryDirectory(prefix="flakegraph-source-") as directory:
            path = Path(directory) / filename
            path.write_bytes(source.payload)
            input_file = InputFile.model_validate({**input_metadata, "path": path})
            prepared = run.pipeline.prepare_documents([input_file])
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.PREPARED_DOCUMENT,
            prepared.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={"file_ids": prepared.file_ids, "schema": "PreparedDocumentShard"},
        )
        windows = build_extraction_windows(
            prepared.chunks,
            self.settings.graph.extraction_window_tokens,
            self.settings.graph.max_chunks_per_llm_call,
        )
        if not windows:
            # A blank or image-only document can legitimately produce no chunks.
            # Persist an empty extracted-document boundary so Spark still has a
            # readable stage prefix and publishes the document provenance tables.
            empty = ExtractedDocumentShard(
                prepared=_prepared_projection_for_extracted_artifact(
                    prepared,
                    spark_finalization=_use_spark_finalization(self.settings),
                ),
                observations=ExtractionObservations(chunk_count=0, window_count=0),
                trace=prepared.trace,
            )
            empty_ref = self.artifact_store.put(
                lease.task.run_id,
                ArtifactKind.EXTRACTED_DOCUMENT,
                empty.model_dump_json().encode("utf-8"),
                _JSON_MEDIA_TYPE,
                metadata={
                    "file_ids": empty.prepared.file_ids,
                    "windows": 0,
                    "schema": ExtractedDocumentShard.__name__,
                },
                identity_key=lease.task.scope_id,
            )
            return TaskExecution(output_artifact_ids=(ref.id, empty_ref.id))
        final_task_id = stable_id("task", lease.task.run_id, TaskStage.FINALIZE_GRAPH.value)
        context_task = TaskDefinition(
            id=stable_id(
                "task",
                lease.task.run_id,
                TaskStage.EXTRACT_DOCUMENT_CONTEXT.value,
                lease.task.scope_id,
            ),
            run_id=lease.task.run_id,
            stage=TaskStage.EXTRACT_DOCUMENT_CONTEXT,
            scope_id=lease.task.scope_id,
            dependency_ids=[lease.task.id],
            priority=self.settings.distributed.task_priority(15),
            max_attempts=self.settings.distributed.max_attempts,
        )
        return TaskExecution(
            output_artifact_ids=(ref.id,),
            follow_up_tasks=(context_task,),
            barrier_task_id=final_task_id,
        )

    def _extract_document_context(self, lease: TaskLease, run: _RunContext) -> TaskExecution:
        """Extract reusable focal entities, then fan out independent body windows.

        Keeping this queue stage between OCR and window extraction makes one LLM
        context call reusable across the whole document while retaining dynamic
        work stealing for every ordinary extraction window.
        """

        inputs = self._dependency_artifacts(lease, ArtifactKind.PREPARED_DOCUMENT)
        if len(inputs) != 1:
            raise ValueError("extract_document_context requires one prepared artifact")
        prepared = PreparedDocumentShard.model_validate_json(inputs[0].payload)
        contextualized = run.pipeline.extract_document_context(prepared)
        context = DocumentContextShard(
            file_ids=contextualized.file_ids,
            document_context_entities=contextualized.document_context_entities,
            trace=contextualized.trace,
        )
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.DOCUMENT_CONTEXT,
            context.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": context.file_ids,
                "schema": DocumentContextShard.__name__,
                "document_context_entities": len(context.document_context_entities),
            },
        )
        windows = [
            window
            for window in build_extraction_windows(
                contextualized.chunks,
                self.settings.graph.extraction_window_tokens,
                self.settings.graph.max_chunks_per_llm_call,
            )
            if window_skip_reason(window) is None
        ]
        follow_up_tasks: list[TaskDefinition] = []
        entity_task_ids: list[str] = []
        window_payloads: list[dict[str, str]] = []
        # A task should expose no more windows than its bounded provider pool can
        # execute concurrently. Larger packs create a hidden serial wave inside
        # one lease, leaving the rest of the fleet idle at the end of a corpus and
        # repeating already-completed calls if the final wave needs a retry.
        windows_per_task = min(
            self.settings.graph.extraction_parallelism,
            _MAX_LOGICAL_WINDOWS_PER_TASK,
        )
        for window_batch in batched(windows, windows_per_task, strict=False):
            batch_id = stable_id("window_batch", *(window.id for window in window_batch))
            window_shard = ExtractionWindowShard(
                file_ids=contextualized.file_ids,
                chunks=[chunk for window in window_batch for chunk in window.chunks],
                logical_window_count=len(window_batch),
            )
            window_ref = self.artifact_store.put(
                lease.task.run_id,
                ArtifactKind.EXTRACTION_WINDOW,
                window_shard.model_dump_json().encode("utf-8"),
                _JSON_MEDIA_TYPE,
                metadata={
                    "file_ids": window_shard.file_ids,
                    "window_id": batch_id,
                    "chunks": len(window_shard.chunks),
                    "logical_windows": window_shard.logical_window_count,
                    "schema": ExtractionWindowShard.__name__,
                },
                identity_key=batch_id,
            )
            window_payloads.append({"window_id": batch_id, "artifact_id": window_ref.id})
            window_task_id = stable_id(
                "task",
                lease.task.run_id,
                TaskStage.EXTRACT_ENTITY_WINDOW.value,
                batch_id,
            )
            entity_task_ids.append(window_task_id)
            follow_up_tasks.append(
                TaskDefinition(
                    id=window_task_id,
                    run_id=lease.task.run_id,
                    stage=TaskStage.EXTRACT_ENTITY_WINDOW,
                    scope_id=batch_id,
                    payload={"window_artifact_id": window_ref.id},
                    dependency_ids=[lease.task.id],
                    priority=self.settings.distributed.task_priority(10),
                    max_attempts=self.settings.distributed.max_attempts,
                )
            )
        # This lightweight barrier materializes the complete document vocabulary.
        # It then emits a second work-stealing wave for relation extraction, so
        # windows can connect entities found elsewhere without serial model work.
        follow_up_tasks.append(
            TaskDefinition(
                id=stable_id(
                    "task",
                    lease.task.run_id,
                    TaskStage.COMPACT_ENTITY_INVENTORY.value,
                    lease.task.scope_id,
                ),
                run_id=lease.task.run_id,
                stage=TaskStage.COMPACT_ENTITY_INVENTORY,
                scope_id=lease.task.scope_id,
                payload={
                    "prepared_artifact_id": inputs[0].ref.id,
                    "windows": window_payloads,
                },
                dependency_ids=[lease.task.id, *entity_task_ids],
                priority=self.settings.distributed.task_priority(5),
                max_attempts=self.settings.distributed.max_attempts,
            )
        )
        return TaskExecution(
            output_artifact_ids=(ref.id,),
            follow_up_tasks=tuple(follow_up_tasks),
            barrier_task_id=stable_id("task", lease.task.run_id, TaskStage.FINALIZE_GRAPH.value),
        )

    def _extract_entity_window(self, lease: TaskLease, run: _RunContext) -> str:
        """Extract entities from one bounded window claimed from the shared queue.

        A compact dependency supplies reusable context and a content-addressed
        window artifact supplies only selected chunks. Prepared page and block rows
        remain in their original stage artifact for graph finalization and are not
        transferred repeatedly to extraction workers.
        """

        context_inputs = self._dependency_artifacts(lease, ArtifactKind.DOCUMENT_CONTEXT)
        if len(context_inputs) != 1:
            raise ValueError("extract_entity_window requires exactly one document-context artifact")
        context = DocumentContextShard.model_validate_json(context_inputs[0].payload)
        window_artifact_id = lease.task.payload.get("window_artifact_id")
        if not isinstance(window_artifact_id, str) or not window_artifact_id:
            raise ValueError("extract_entity_window task requires window_artifact_id")
        window_artifact = self.artifact_store.get(window_artifact_id)
        if window_artifact.ref.kind != ArtifactKind.EXTRACTION_WINDOW:
            raise ValueError("extract_entity_window input is not an extraction-window artifact")
        window = ExtractionWindowShard.model_validate_json(window_artifact.payload)
        if window.file_ids != context.file_ids:
            raise ValueError("extraction window and document context file identities differ")
        window_prepared = PreparedDocumentShard(
            file_ids=window.file_ids,
            files_seen=0,
            documents_processed=0,
            chunks=window.chunks,
            document_context_entities=context.document_context_entities,
            # Context trace is persisted once by document compaction rather than
            # copied into every window result.
            trace=[],
        )
        extracted = run.pipeline.extract_window_entities(window_prepared)
        if extracted.logical_window_count != window.logical_window_count:
            raise ValueError("entity task reconstructed a different logical window count")
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.EXTRACTED_ENTITY_WINDOW,
            extracted.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": extracted.file_ids,
                "window_id": lease.task.scope_id,
                "schema": EntityWindowShard.__name__,
            },
        )
        return ref.id

    def _compact_entity_inventory(self, lease: TaskLease) -> TaskExecution:
        """Compact first-phase mentions and fan out relation-window tasks.

        This stage performs no provider work. It stores one document-wide entity
        inventory, then atomically extends the durable DAG with relation tasks and
        their final document compaction barrier. At-least-once retries are safe
        because task IDs and artifact identities are deterministic.
        """

        dependencies = self._all_dependency_artifacts(lease)
        contexts = [item for item in dependencies if item.ref.kind == ArtifactKind.DOCUMENT_CONTEXT]
        entity_windows = [
            item for item in dependencies if item.ref.kind == ArtifactKind.EXTRACTED_ENTITY_WINDOW
        ]
        unexpected = [
            item.ref.id
            for item in dependencies
            if item.ref.kind
            not in {ArtifactKind.DOCUMENT_CONTEXT, ArtifactKind.EXTRACTED_ENTITY_WINDOW}
        ]
        if unexpected:
            raise ValueError(
                f"compact_entity_inventory received unsupported artifacts: {unexpected}"
            )
        if len(contexts) != 1:
            raise ValueError(
                "compact_entity_inventory requires exactly one document-context artifact"
            )
        context = DocumentContextShard.model_validate_json(contexts[0].payload)
        shards = [EntityWindowShard.model_validate_json(item.payload) for item in entity_windows]
        if any(shard.file_ids != context.file_ids for shard in shards):
            raise ValueError("entity windows and document context file identities differ")
        entities = _unique_entities(
            [
                *context.document_context_entities,
                *(entity for shard in shards for entity in shard.entities),
            ]
        )
        inventory = DocumentEntityInventoryShard(
            file_ids=context.file_ids,
            entities=entities,
            trace=[*context.trace, *(event for shard in shards for event in shard.trace)],
            chunk_count=sum(len(shard.chunk_ids) for shard in shards),
            window_count=sum(shard.logical_window_count for shard in shards),
        )
        inventory_ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
            inventory.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": inventory.file_ids,
                "entities": len(inventory.entities),
                "windows": inventory.window_count,
                "schema": DocumentEntityInventoryShard.__name__,
            },
            identity_key=lease.task.scope_id,
        )

        window_payloads = lease.task.payload.get("windows")
        if not isinstance(window_payloads, list):
            raise ValueError("compact_entity_inventory requires a windows payload")
        relation_tasks: list[TaskDefinition] = []
        for item in window_payloads:
            if not isinstance(item, dict):
                raise ValueError("compact_entity_inventory windows must be objects")
            window_id = item.get("window_id")
            artifact_id = item.get("artifact_id")
            if not isinstance(window_id, str) or not isinstance(artifact_id, str):
                raise ValueError("compact_entity_inventory window identities must be strings")
            relation_tasks.append(
                TaskDefinition(
                    id=stable_id(
                        "task",
                        lease.task.run_id,
                        TaskStage.EXTRACT_RELATION_WINDOW.value,
                        window_id,
                    ),
                    run_id=lease.task.run_id,
                    stage=TaskStage.EXTRACT_RELATION_WINDOW,
                    scope_id=window_id,
                    payload={"window_artifact_id": artifact_id},
                    dependency_ids=[lease.task.id],
                    priority=self.settings.distributed.task_priority(4),
                    max_attempts=self.settings.distributed.max_attempts,
                )
            )
        compact_task = TaskDefinition(
            id=stable_id(
                "task",
                lease.task.run_id,
                TaskStage.COMPACT_DOCUMENT.value,
                lease.task.scope_id,
            ),
            run_id=lease.task.run_id,
            stage=TaskStage.COMPACT_DOCUMENT,
            scope_id=lease.task.scope_id,
            payload={"prepared_artifact_id": lease.task.payload.get("prepared_artifact_id")},
            dependency_ids=[lease.task.id, *(task.id for task in relation_tasks)],
            priority=self.settings.distributed.task_priority(3),
            max_attempts=self.settings.distributed.max_attempts,
        )
        return TaskExecution(
            output_artifact_ids=(inventory_ref.id,),
            follow_up_tasks=(*relation_tasks, compact_task),
            barrier_task_id=stable_id(
                "task",
                lease.task.run_id,
                TaskStage.FINALIZE_GRAPH.value,
            ),
        )

    def _extract_relation_window(self, lease: TaskLease, run: _RunContext) -> str:
        """Extract relations in one window against its complete document inventory."""

        inventory_inputs = self._dependency_artifacts(lease, ArtifactKind.DOCUMENT_ENTITY_INVENTORY)
        if len(inventory_inputs) != 1:
            raise ValueError(
                "extract_relation_window requires exactly one document entity inventory"
            )
        inventory = DocumentEntityInventoryShard.model_validate_json(inventory_inputs[0].payload)
        window_artifact_id = lease.task.payload.get("window_artifact_id")
        if not isinstance(window_artifact_id, str) or not window_artifact_id:
            raise ValueError("extract_relation_window task requires window_artifact_id")
        window_artifact = self.artifact_store.get(window_artifact_id)
        if window_artifact.ref.kind != ArtifactKind.EXTRACTION_WINDOW:
            raise ValueError("extract_relation_window input is not an extraction-window artifact")
        window = ExtractionWindowShard.model_validate_json(window_artifact.payload)
        if window.file_ids != inventory.file_ids:
            raise ValueError("relation window and entity inventory file identities differ")
        window_prepared = PreparedDocumentShard(
            file_ids=window.file_ids,
            files_seen=0,
            documents_processed=0,
            chunks=window.chunks,
        )
        extracted = run.pipeline.extract_window_relations(
            window_prepared,
            inventory.entities,
        )
        if extracted.logical_window_count != window.logical_window_count:
            raise ValueError("relation task reconstructed a different logical window count")
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.EXTRACTED_RELATION_WINDOW,
            extracted.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": extracted.file_ids,
                "window_id": lease.task.scope_id,
                "schema": RelationWindowShard.__name__,
            },
        )
        return ref.id

    def _compact_document(self, lease: TaskLease) -> str:
        """Assemble independently extracted windows into one document artifact.

        The compaction boundary keeps the object count consumed by Spark
        proportional to documents rather than LLM windows. It performs no provider
        calls and is safe to retry because both its source artifacts and resulting
        content-addressed document shard are immutable.
        """

        prepared_artifact_id = lease.task.payload.get("prepared_artifact_id")
        if not isinstance(prepared_artifact_id, str) or not prepared_artifact_id:
            raise ValueError("compact_document requires prepared_artifact_id")
        prepared_artifact = self.artifact_store.get(prepared_artifact_id)
        if prepared_artifact.ref.kind != ArtifactKind.PREPARED_DOCUMENT:
            raise ValueError("compact_document input is not a prepared document artifact")
        prepared = PreparedDocumentShard.model_validate_json(prepared_artifact.payload)

        dependencies = self._all_dependency_artifacts(lease)
        inventories = [
            item for item in dependencies if item.ref.kind == ArtifactKind.DOCUMENT_ENTITY_INVENTORY
        ]
        windows = [
            item for item in dependencies if item.ref.kind == ArtifactKind.EXTRACTED_RELATION_WINDOW
        ]
        unexpected = [
            item.ref.id
            for item in dependencies
            if item.ref.kind
            not in {
                ArtifactKind.DOCUMENT_ENTITY_INVENTORY,
                ArtifactKind.EXTRACTED_RELATION_WINDOW,
            }
        ]
        if unexpected:
            raise ValueError(f"compact_document received unsupported artifacts: {unexpected}")
        if len(inventories) != 1:
            raise ValueError("compact_document requires exactly one document entity inventory")
        inventory = DocumentEntityInventoryShard.model_validate_json(inventories[0].payload)
        if inventory.file_ids != prepared.file_ids:
            raise ValueError("prepared document and entity inventory file identities differ")
        relation_shards = [
            RelationWindowShard.model_validate_json(item.payload) for item in windows
        ]
        if any(shard.file_ids != prepared.file_ids for shard in relation_shards):
            raise ValueError("relation windows and prepared document file identities differ")
        if sum(shard.logical_window_count for shard in relation_shards) != inventory.window_count:
            raise ValueError("relation windows do not cover the complete entity inventory")
        relation_trace = [event for shard in relation_shards for event in shard.trace]
        compacted = ExtractedDocumentShard(
            prepared=_prepared_projection_for_extracted_artifact(
                prepared,
                spark_finalization=_use_spark_finalization(self.settings),
            ),
            observations=ExtractionObservations(
                entities=inventory.entities,
                relations=_unique_relations(
                    [relation for shard in relation_shards for relation in shard.relations]
                ),
                trace=[*inventory.trace, *relation_trace],
                chunk_count=inventory.chunk_count,
                window_count=inventory.window_count,
            ),
            trace=[*inventory.trace, *relation_trace],
        )
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.EXTRACTED_DOCUMENT,
            compacted.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "file_ids": compacted.prepared.file_ids,
                "windows": len(relation_shards),
                "schema": ExtractedDocumentShard.__name__,
            },
            identity_key=lease.task.scope_id,
        )
        return ref.id

    def _finalize_graph(self, lease: TaskLease, run: _RunContext) -> str:
        """Select the local or partitioned engine without a corpus-size cutoff."""

        if _use_spark_finalization(self.settings):
            return self._finalize_graph_with_spark(lease, run)
        return self._finalize_graph_locally(lease, run)

    def _finalize_graph_with_spark(self, lease: TaskLease, run: _RunContext) -> str:
        """Coordinate distributed finalization and persist only its small manifest.

        Spark executors read immutable stage objects directly. The leased worker is
        a driver and heartbeat owner; it never deserializes the complete corpus.
        """

        report = self._task_progress_reporter(lease)
        stage_kinds = {ArtifactKind.PREPARED_DOCUMENT, ArtifactKind.EXTRACTED_DOCUMENT}
        inputs = self.artifact_store.list_run_artifacts(lease.task.run_id, stage_kinds)
        inputs.extend(self._inherited_artifact_refs(lease, stage_kinds))
        manifest = SparkGraphFinalizer(run.settings, report).finalize(
            SparkFinalizationRequest(
                run_id=lease.task.run_id,
                graph_id=lease.task.scope_id,
                attempt=lease.attempt,
                artifact_ids=frozenset(ref.id for ref in inputs),
                source_run_ids=frozenset(ref.run_id for ref in inputs),
            )
        )
        # Destination publication is queued atomically by ``complete_task`` after
        # this immutable manifest exists. A separate fenced outbox delivery owns
        # the externally visible Snowflake effect.
        report(finalization_progress("publish_destination", completed=0, total=1))
        report(finalization_progress("persist_manifest", completed=0, total=1))
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.GRAPH_RESULT,
            manifest.model_dump_json().encode("utf-8"),
            _GRAPH_MANIFEST_MEDIA_TYPE,
            metadata={
                "graph_id": manifest.graph_id,
                "engine": manifest.engine,
                "schema": manifest.format,
                "tables": {name: table.row_count for name, table in manifest.tables.items()},
            },
        )
        report(finalization_progress("persist_manifest", completed=1, total=1))
        return ref.id

    def _finalize_graph_locally(self, lease: TaskLease, run: _RunContext) -> str:
        """Recombine stage objects for the lightweight in-process execution path."""

        report = self._task_progress_reporter(lease)
        report(finalization_progress("read_artifacts", completed=0, total=1))
        inputs = self.artifact_store.get_run_artifacts(
            lease.task.run_id,
            {ArtifactKind.EXTRACTED_DOCUMENT},
        )
        inherited = self._inherited_artifact_refs(lease, {ArtifactKind.EXTRACTED_DOCUMENT})
        inputs.extend(self.artifact_store.get_many([ref.id for ref in inherited]))
        document_shards = [
            ExtractedDocumentShard.model_validate_json(item.payload) for item in inputs
        ]
        document_shards.sort(key=lambda shard: tuple(shard.prepared.file_ids))
        if not document_shards:
            raise ValueError("finalize_graph requires at least one extracted document")
        report(finalization_progress("read_artifacts", completed=1, total=1))
        report(finalization_progress("materialize_inputs", completed=1, total=1))
        report(finalization_progress("resolve_entities", completed=0, total=1))
        batch = run.pipeline.finalize_document_shards(document_shards, write=True)
        for phase, _label in FINALIZATION_PHASES[4:11]:
            report(finalization_progress(phase, completed=1, total=1))
        report(finalization_progress("persist_manifest", completed=0, total=1))
        ref = self.artifact_store.put(
            lease.task.run_id,
            ArtifactKind.GRAPH_RESULT,
            batch.model_dump_json().encode("utf-8"),
            _JSON_MEDIA_TYPE,
            metadata={
                "graph_id": batch.graph_id,
                "nodes": len(batch.nodes),
                "edges": len(batch.edges),
                "schema": GraphWriteBatch.__name__,
            },
        )
        report(finalization_progress("persist_manifest", completed=1, total=1))
        return ref.id

    def _inherited_artifact_refs(
        self,
        lease: TaskLease,
        kinds: set[ArtifactKind],
    ) -> list[ArtifactRef]:
        """The stage outputs a revision keeps from earlier runs of its graph.

        The finalizer's payload names each run and the documents kept from it;
        an artifact is kept when every document it covers is. Nothing is copied:
        the earlier run's objects are read where they are.
        """

        entries = lease.task.payload.get("inherit")
        if not isinstance(entries, list) or not entries:
            return []
        refs: list[ArtifactRef] = []
        for raw in entries:
            entry = InheritedDocuments.model_validate(raw)
            kept = set(entry.file_ids)
            for ref in self.artifact_store.list_run_artifacts(entry.run_id, kinds):
                covered = ref.metadata.get("file_ids")
                if isinstance(covered, list) and covered and set(map(str, covered)) <= kept:
                    refs.append(ref)
        return refs

    def _task_progress_reporter(
        self,
        lease: TaskLease,
    ) -> Callable[[TaskProgress], None]:
        """Return a best-effort reporter bound to the task's current lease owner."""

        def report(progress: TaskProgress) -> None:
            """Persist progress without allowing observability failure to abort work."""

            try:
                self.task_store.report_task_progress(lease.task.id, self.worker_id, progress)
            except Exception:
                # Lease heartbeat remains authoritative for ownership and
                # availability. A progress write is useful but non-critical.
                return

        return report

    def _dependency_artifacts(
        self,
        lease: TaskLease,
        expected_kind: ArtifactKind,
    ) -> list[StoredArtifact]:
        """Load direct dependency outputs in deterministic task/output order."""

        artifact_ids = [
            artifact_id
            for dependency_id in sorted(lease.dependency_outputs)
            for artifact_id in lease.dependency_outputs[dependency_id]
        ]
        artifacts = self.artifact_store.get_many(artifact_ids)
        unexpected = [item.ref.id for item in artifacts if item.ref.kind != expected_kind]
        if unexpected:
            raise ValueError(
                f"task dependencies contain artifacts with the wrong kind: {unexpected}"
            )
        return artifacts

    def _all_dependency_artifacts(self, lease: TaskLease) -> list[StoredArtifact]:
        """Load all direct outputs in deterministic task and output order."""

        artifact_ids = [
            artifact_id
            for dependency_id in sorted(lease.dependency_outputs)
            for artifact_id in lease.dependency_outputs[dependency_id]
        ]
        return self.artifact_store.get_many(artifact_ids)


def _unique_entities(entities: list[EntityMention]) -> list[EntityMention]:
    """Deduplicate entity observations deterministically at a document barrier.

    Mention IDs include grounded source identity, so retaining the first record for
    each ID removes context/window overlap without collapsing distinct occurrences
    that graph-wide entity resolution still needs to adjudicate.
    """

    return list({entity.id: entity for entity in entities}.values())


def _unique_relations(relations: list[RelationObservation]) -> list[RelationObservation]:
    """Deduplicate identical grounded relation observations after queue fan-in."""

    return list({relation.id: relation for relation in relations}.values())


def _prepared_projection_for_extracted_artifact(
    prepared: PreparedDocumentShard,
    *,
    spark_finalization: bool,
) -> PreparedDocumentShard:
    """Avoid copying prepared corpus rows into Spark extraction artifacts.

    Spark reads the immutable ``prepared_document`` prefix directly for documents,
    pages, blocks, assets, and chunks. Repeating those rows inside every completed
    document doubles object-store bytes and JSON scanning at corpus scale. The
    extracted artifact therefore retains only stable file-level counters on the
    Spark path; its observation payload already contains every entity and relation
    needed by finalization. Local finalization keeps the complete prepared shard
    because it intentionally operates from self-contained document artifacts.
    """

    if not spark_finalization:
        return prepared
    return PreparedDocumentShard(
        file_ids=prepared.file_ids,
        files_seen=prepared.files_seen,
        documents_processed=prepared.documents_processed,
        ocr_cache_hits=prepared.ocr_cache_hits,
    )


def _safe_filename(value: object) -> str:
    """Return a basename suitable for a temporary worker directory."""

    if not isinstance(value, str) or not value.strip():
        return "source.bin"
    return Path(value).name or "source.bin"


def _use_spark_finalization(settings: Settings) -> bool:
    """Resolve automatic engine selection from runtime capabilities, not corpus size.

    Local installations remain dependency-light. A Kubernetes deployment with a
    shared artifact URI automatically receives the scalable engine for both small
    and large corpora, giving one semantic execution path throughout that fleet.
    """

    selected = settings.distributed.finalization_engine
    if selected == "spark":
        return True
    if selected == "local":
        return False
    return settings.runtime.runtime == "kubernetes" and bool(settings.distributed.artifact_uri)
