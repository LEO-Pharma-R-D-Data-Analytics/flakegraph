"""Build durable, dynamically fanned-out task graphs for FlakeGraph runs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time
from itertools import chain, islice
from pathlib import Path
from typing import cast
from uuid import uuid4

import yaml

from kg_processor.application.extraction_contracts import extraction_contract_fingerprint
from kg_processor.application.prompt_registry import two_pass_prompt_fingerprints
from kg_processor.application.redaction import redact_sensitive_data, redact_sensitive_text
from kg_processor.config.settings import Settings
from kg_processor.domain.distributed import (
    ArtifactKind,
    ArtifactRef,
    RunDefinition,
    RunSummary,
    TaskDefinition,
    TaskStage,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.ports.artifact_store import ArtifactStore
from kg_processor.ports.file_source import FileSource, IterableFileSource
from kg_processor.ports.task_store import InitialTaskStreamWriter, TaskStore

_SOURCE_STAGING_PARALLELISM = 16
_SOURCE_STAGING_MAX_IN_FLIGHT_BYTES = 512 * 1024 * 1024
_SOURCE_STAGING_BATCH_SIZE = 256
_FINALIZER_PASSWORD_SLOT = "KG_SNOWFLAKE_PASSWORD"
_FINALIZER_OAUTH_TOKEN_SLOT = "KG_SNOWFLAKE_OAUTH_TOKEN"


class DistributedRunPlanner:
    """Persist sources and the initial dependency graph for a distributed run.

    Each source starts with one preparation task. Preparation discovers the actual
    extraction windows after OCR and chunking, then publishes those fine-grained
    tasks atomically. Each document's windows converge on one compaction task. The
    final task is a run-wide barrier, so it needs no explicit edge from every
    document and remains constant-size for corpora containing hundreds of thousands
    of files.
    """

    def __init__(
        self,
        settings: Settings,
        file_source: FileSource,
        task_store: TaskStore,
        artifact_store: ArtifactStore,
    ) -> None:
        """Retain explicit ports so planning is testable without PostgreSQL or filesystems."""

        self.settings = settings
        self.file_source = file_source
        self.task_store = task_store
        self.artifact_store = artifact_store

    def submit(self, run_id: str | None = None) -> RunSummary:
        """Create, populate, and atomically activate one distributed graph run.

        Planning failures cancel the partially-created run. Source artifacts remain
        available for diagnosis until an operator deliberately removes terminal-run
        artifacts; workers can never see tasks until the complete DAG is activated.
        """

        files = self._iter_input_files()
        first_file = next(files, None)
        if first_file is None:
            raise ValueError("distributed run source did not contain any supported files")
        validated_files = _iter_unique_file_ids(chain((first_file,), files))
        effective_run_id = run_id or f"run_{uuid4().hex}"
        snapshot = distributed_processing_config(self.settings)
        definition = RunDefinition(
            id=effective_run_id,
            graph_id=self.settings.job.graph_id,
            config=snapshot,
            config_digest=distributed_processing_config_digest(self.settings),
        )
        if run_id is not None:
            try:
                existing = self.task_store.get_run_summary(effective_run_id)
            except KeyError:
                existing = None
            if existing is not None and existing.run.status.value != "planning":
                if existing.run.config_digest != definition.config_digest:
                    raise ValueError(
                        f"run id {effective_run_id} already exists with different inputs"
                    )
                # Explicit run IDs are idempotency keys. Returning the existing
                # summary is safe for active and terminal runs and, critically,
                # never cancels live work merely because submit was retried.
                return existing
        self.task_store.create_run(definition)
        try:
            tasks = self._iter_initial_tasks(effective_run_id, validated_files)
            if isinstance(self.task_store, InitialTaskStreamWriter):
                self.task_store.add_initial_tasks(effective_run_id, tasks)
            else:
                # Lightweight or third-party stores can retain the general DAG
                # API. Production PostgreSQL consumes the iterator directly.
                self.task_store.add_tasks(effective_run_id, list(tasks))
            self.task_store.activate_run(effective_run_id)
        except Exception:
            # Explicit run IDs are durable planning idempotency keys. PostgreSQL
            # commits bounded COPY batches so an interrupted coordinator can
            # safely replay the deterministic stream and activate the same run.
            try:
                partial = self.task_store.get_run_summary(effective_run_id)
            except Exception:
                partial = None
            if run_id is None or partial is None or partial.total_tasks == 0:
                self.task_store.cancel_run(effective_run_id)
            raise
        # Submission is an operator-facing path, so return bounded counts rather
        # than serializing every task immediately after inserting a large plan.
        return self.task_store.get_run_summary(effective_run_id)

    def _build_tasks(self, run_id: str, files: list[InputFile]) -> list[TaskDefinition]:
        """Persist immutable inputs and return the safe initial task graph.

        Extraction windows cannot be known before OCR and chunking. Deferring that
        fan-out also avoids estimating work from source byte size, which correlates
        poorly with pages, chunks, and model requests.
        """

        return list(self._iter_tasks(run_id, files))

    def _iter_tasks(self, run_id: str, files: list[InputFile]) -> Iterator[TaskDefinition]:
        """Stage sources and yield the initial plan without retaining task rows.

        Ordered executor mapping provides one artifact reference at a time. The
        PostgreSQL adapter writes each corresponding task directly to ``COPY``, so
        coordinator memory depends on upload concurrency rather than corpus size.
        """

        ordered_files = sorted(files, key=lambda item: (item.source_uri, item.id))
        yield from self._iter_initial_tasks(run_id, ordered_files)

    def _iter_initial_tasks(
        self,
        run_id: str,
        files: Iterable[InputFile],
    ) -> Iterator[TaskDefinition]:
        """Yield stable tasks from bounded, continuously replenished upload batches.

        Task and artifact identities derive from the run and file rather than
        insertion order. Emitting each upload as it completes therefore lets a
        slow object overlap later files without changing retry or graph semantics.
        """

        for file_batch in _batches(files, _SOURCE_STAGING_BATCH_SIZE):
            for file, source_ref in self._iter_staged_sources(run_id, file_batch):
                yield TaskDefinition(
                    id=stable_id("task", run_id, TaskStage.PREPARE_DOCUMENT.value, file.id),
                    run_id=run_id,
                    stage=TaskStage.PREPARE_DOCUMENT,
                    scope_id=file.id,
                    payload={"source_artifact_id": source_ref.id},
                    priority=20,
                    max_attempts=self.settings.distributed.max_attempts,
                )
        yield TaskDefinition(
            id=stable_id("task", run_id, TaskStage.FINALIZE_GRAPH.value),
            run_id=run_id,
            stage=TaskStage.FINALIZE_GRAPH,
            scope_id=self.settings.job.graph_id,
            payload=_graph_output_payload(self.settings),
            priority=0,
            max_attempts=self.settings.distributed.max_attempts,
        )

    def _iter_input_files(self) -> Iterator[InputFile]:
        """Use streaming discovery when an adapter provides deterministic iteration."""

        if isinstance(self.file_source, IterableFileSource):
            return self.file_source.iter_files()
        files = sorted(
            self.file_source.list_files(),
            key=lambda item: (item.source_uri, item.id),
        )
        return iter(files)

    def _store_sources(self, run_id: str, files: list[InputFile]) -> list[ArtifactRef]:
        """Stage source bytes concurrently while retaining deterministic ordering.

        Source reads and object-store writes are independent, latency-bound work.
        The bounded ``buffersize`` prevents a large corpus from being read into
        memory at once, while ordered ``map`` output keeps task construction and
        its reproducible IDs independent from upload completion timing.
        """

        return list(self._iter_sources(run_id, files))

    def _iter_staged_sources(
        self,
        run_id: str,
        files: list[InputFile],
    ) -> Iterator[tuple[InputFile, ArtifactRef]]:
        """Yield source/reference pairs as bounded concurrent uploads complete.

        A planning batch contains at most 256 lightweight file records. Queuing
        that bounded metadata is cheap, while only ``workers`` calls can read and
        retain complete source bytes. Completion-order iteration immediately
        replaces a finished upload even when an earlier object remains slow.
        """

        if len(files) <= 1:
            for file in files:
                yield file, self._store_source(run_id, file)
            return
        workers = _source_staging_workers(files)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            uploads = {executor.submit(self._store_source, run_id, file): file for file in files}
            for upload in as_completed(uploads):
                yield uploads[upload], upload.result()

    def _iter_sources(self, run_id: str, files: list[InputFile]) -> Iterator[ArtifactRef]:
        """Yield staged source references through a bounded ordered upload pool."""

        if len(files) <= 1:
            for file in files:
                yield self._store_source(run_id, file)
            return
        workers = _source_staging_workers(files)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            yield from executor.map(
                lambda file: self._store_source(run_id, file),
                files,
                buffersize=workers,
            )

    def _store_source(self, run_id: str, file: InputFile) -> ArtifactRef:
        """Read and checksum-verify one file before handing its bytes to workers."""

        payload = file.path.read_bytes()
        checksum = sha256_hex(payload)
        if checksum != file.checksum:
            raise ValueError(
                f"source file changed after discovery: {file.path} "
                f"(expected {file.checksum}, got {checksum})"
            )
        if len(payload) != file.size_bytes:
            raise ValueError(
                f"source file size changed after discovery: {file.path} "
                f"(expected {file.size_bytes}, got {len(payload)})"
            )
        return self.artifact_store.put(
            run_id,
            ArtifactKind.SOURCE_DOCUMENT,
            payload,
            file.mime_type,
            metadata={
                "input_file": {
                    "id": file.id,
                    "source_uri": redact_sensitive_text(file.source_uri),
                    "checksum": file.checksum,
                    "mime_type": file.mime_type,
                    "size_bytes": file.size_bytes,
                    "filename": Path(file.path).name,
                }
            },
            identity_key=file.id,
        )


def _graph_output_payload(settings: Settings) -> dict[str, object]:
    """Return the non-secret destination contract consumed by finalizer workers.

    Worker pools may serve many runs with different Snowflake schemas. Connection
    coordinates therefore travel with the final task, while credentials remain
    named environment references resolved only inside the claiming worker.
    """

    provider = settings.writer.provider
    if provider == "local_artifacts":
        return {"output": {"provider": provider}}
    if provider != "snowflake_bulk":
        raise ValueError(
            "distributed Spark finalization supports local_artifacts or snowflake_bulk output"
        )
    snowflake = settings.snowflake
    _validate_finalizer_credential_slot(
        snowflake.password_environment_variable,
        _FINALIZER_PASSWORD_SLOT,
        "password",
    )
    _validate_finalizer_credential_slot(
        snowflake.oauth_token_environment_variable,
        _FINALIZER_OAUTH_TOKEN_SLOT,
        "OAuth token",
    )
    return {
        "output": {
            "provider": provider,
            "snowflake": {
                "account": snowflake.account,
                "host": snowflake.host,
                "user": snowflake.user,
                "authenticator": snowflake.authenticator,
                "database": snowflake.database,
                "schema": snowflake.schema_name,
                "role": snowflake.role,
                "warehouse": snowflake.warehouse,
                "bulk_stage": snowflake.bulk_stage,
                "password_credential_slot": (
                    "password" if snowflake.password_environment_variable else None
                ),
                "password_environment_variable": (
                    _FINALIZER_PASSWORD_SLOT if snowflake.password_environment_variable else None
                ),
                "oauth_token_credential_slot": (
                    "oauth_token" if snowflake.oauth_token_environment_variable else None
                ),
                "oauth_token_environment_variable": (
                    _FINALIZER_OAUTH_TOKEN_SLOT
                    if snowflake.oauth_token_environment_variable
                    else None
                ),
                "bulk_target_file_size_mb": snowflake.bulk_target_file_size_mb,
            },
        }
    }


def _validate_finalizer_credential_slot(
    configured_name: str | None,
    dedicated_name: str,
    label: str,
) -> None:
    """Prevent task payloads from turning arbitrary worker environment into secrets."""

    if configured_name is not None and configured_name != dedicated_name:
        raise ValueError(
            f"distributed Snowflake finalizers require the dedicated {label} "
            f"credential slot {dedicated_name}"
        )


def _source_staging_workers(files: list[InputFile]) -> int:
    """Bound upload concurrency by both object count and worst-case memory.

    ``ArtifactStore.put`` accepts immutable bytes, so each active staging worker
    must hold one complete source object while checksumming and uploading it.
    Typical papers retain sixteen-way object-store concurrency; unusually large
    files reduce the pool automatically so a submitter cannot transiently retain
    many gigabytes merely because the corpus contains several large documents.
    """

    if not files:
        return 1
    largest_file = max(file.size_bytes for file in files)
    byte_limited_workers = max(
        1,
        _SOURCE_STAGING_MAX_IN_FLIGHT_BYTES // max(largest_file, 1),
    )
    return min(_SOURCE_STAGING_PARALLELISM, len(files), byte_limited_workers)


def _batches[T](items: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield non-empty bounded lists from a potentially corpus-sized iterator."""

    if size <= 0:
        raise ValueError("batch size must be positive")
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch


def _iter_unique_file_ids(files: Iterable[InputFile]) -> Iterator[InputFile]:
    """Reject duplicate durable file identities while preserving streaming order."""

    seen: set[str] = set()
    for file in files:
        if file.id in seen:
            raise ValueError(f"duplicate input file id: {file.id}")
        seen.add(file.id)
        yield file


def distributed_processing_config(settings: Settings) -> dict[str, object]:
    """Return the redacted graph-semantic configuration shared by every worker.

    Worker IDs, leases, polling intervals, source discovery paths, and deployment
    runtime do not affect graph semantics and are intentionally excluded. This lets
    specialized worker pools tune operations without failing compatibility checks.
    """

    full = settings.model_dump(mode="json", by_alias=True)
    snapshot = {
        "graph_id": settings.job.graph_id,
        "ocr": full["ocr"],
        "generic_http_ocr": full["generic_http_ocr"],
        "llm": full["llm"],
        "embedding": full["embedding"],
        "ontology": full["ontology"],
        "extractors": full["extractors"],
        "graph": full["graph"],
        "writer": full["writer"],
        "cache": full["cache"],
        "snowflake": full["snowflake"],
    }
    return cast(dict[str, object], redact_sensitive_data(snapshot))


def distributed_processing_config_digest(settings: Settings) -> str:
    """Hash graph-semantic settings while permitting deployment-local routing.

    The persisted run snapshot retains redacted endpoints and operational settings
    for diagnosis, but those values cannot participate in worker compatibility.
    A production fleet may route extraction workers to node-local replicas while a
    finalizer uses a cluster-wide Service, and heterogeneous workers may use
    different device names or cache paths. Provider and model identities, parsing
    behavior, ontology content, extraction policy, and graph policy remain part of
    the digest because changing any of them can change the resulting graph.
    """

    canonical = json.dumps(
        distributed_processing_compatibility_config(settings),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256_hex(canonical)


def distributed_processing_compatibility_config(settings: Settings) -> dict[str, object]:
    """Return the canonical semantic contract that every worker must share.

    Transport addresses, credentials, timeouts, executable paths, batching, cache
    locations, writer destinations, and Snowflake connection details describe how
    a worker reaches infrastructure rather than what graph it should produce. They
    are deliberately absent. The ontology is represented by its content digest
    instead of its mount path, so equivalent mounts compare equal and changed
    ontology bytes are detected.
    """

    full = settings.model_dump(mode="json", by_alias=True)
    compatibility = {
        # Prompt text and response contracts are executable extraction policy,
        # so old and new worker images with identical YAML must not contribute to
        # one run. Read from the prompts and contracts themselves rather than
        # from a revision label: the label is a second thing to remember on every
        # edit, and a forgotten bump is invisible — both images claim the same
        # run and mix observations produced under different instructions.
        "extraction_prompts": two_pass_prompt_fingerprints(),
        "extraction_contracts": extraction_contract_fingerprint(),
        "ocr": _without_keys(
            full["ocr"],
            {
                "mineru_api_key",
                "mineru_api_url",
                "mineru_command",
                "mineru_server_url",
                "model_cache_dir",
                "tesseract_command",
                "tesseract_pdf_renderer_command",
                "timeout_seconds",
            },
        ),
        "generic_http_ocr": _without_keys(
            full["generic_http_ocr"],
            {"api_key", "endpoint", "max_response_bytes"},
        ),
        "llm": _without_keys(full["llm"], {"api_key", "endpoint", "timeout_seconds"}),
        "embedding": _without_keys(
            full["embedding"],
            {"api_key", "batch_size", "device", "endpoint"},
        ),
        "ontology": {"profile_sha256": _ontology_profile_digest(settings)},
        "extractors": full["extractors"],
        "graph": _without_keys(
            full["graph"],
            {
                "community_report_parallelism",
                "description_merge_parallelism",
                "extraction_parallelism",
                "resolution_parallelism",
            },
        ),
    }
    return cast(dict[str, object], compatibility)


def _without_keys(value: object, excluded: set[str]) -> dict[str, object]:
    """Copy one serialized settings mapping without deployment-specific fields."""

    if not isinstance(value, dict):
        raise TypeError("serialized settings section must be a mapping")
    return {str(key): item for key, item in value.items() if str(key) not in excluded}


def _ontology_profile_digest(settings: Settings) -> str | None:
    """Identify ontology semantics independently of how the profile is carried.

    A run submitted by the application inlines its profile so the run describes
    itself wherever it is executed, while a fleet worker mounts the same ontology
    as a file. Hashing the one as canonical content and the other as raw bytes
    makes two identical ontologies compare unequal, and a run no worker can match
    is simply never claimed: it stays queued with nothing on the page to say why.
    Both forms are therefore read into the same canonical content first.
    """

    profile = settings.ontology.profile
    if profile is None:
        profile_path = settings.ontology.profile_path
        if profile_path is None:
            return None
        # An empty or comment-only file reads as ``None``. The application's own
        # inliners record that as an empty mapping, so this branch must too, or
        # the two forms disagree again over an ontology that says nothing.
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    canonical = json.dumps(
        profile,
        sort_keys=True,
        separators=(",", ":"),
        default=_canonical_ontology_scalar,
    )
    return sha256_hex(canonical.encode("utf-8"))


def _canonical_ontology_scalar(value: object) -> str:
    """Render the scalars YAML admits and JSON does not, or refuse the profile.

    An unquoted date reads as a date, and rendering it by its ISO text keeps an
    unusual but valid ontology identifiable. A set is deliberately not rendered:
    its text order follows hash randomization, so two workers reading the same
    file would disagree about it and neither would claim the run — the very
    failure this digest exists to prevent, reintroduced silently.
    """

    if isinstance(value, date | datetime | time):
        return value.isoformat()
    raise TypeError(
        f"ontology profile contains a {type(value).__name__} that cannot be identified stably"
    )
