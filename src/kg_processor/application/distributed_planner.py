"""Build durable, dynamically fanned-out task graphs for FlakeGraph runs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time
from itertools import batched, chain
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
    InheritedDocuments,
    RevisionRequest,
    RunDefinition,
    RunStatus,
    RunSummary,
    TaskDefinition,
    TaskStage,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.ports.artifact_store import ArtifactStore
from kg_processor.ports.file_source import FileSource, IterableFileSource
from kg_processor.ports.task_store import TaskStore

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
        file_source: FileSource | None,
        task_store: TaskStore,
        artifact_store: ArtifactStore,
    ) -> None:
        """Retain explicit ports so planning is testable without PostgreSQL or filesystems.

        A revision that only removes documents has no source to discover, so the
        file source is optional there; a run that adds nothing and keeps nothing
        is refused at submit.
        """

        self.settings = settings
        self.file_source = file_source
        self.task_store = task_store
        self.artifact_store = artifact_store

    def submit(
        self,
        run_id: str | None = None,
        revision: RevisionRequest | None = None,
    ) -> RunSummary:
        """Create, populate, and atomically activate one distributed graph run.

        Planning failures cancel the partially-created run. Source artifacts remain
        available for diagnosis until an operator deliberately removes terminal-run
        artifacts; workers can never see tasks until the complete DAG is activated.

        With a ``revision`` the run is a new version of the base run's graph: it
        keeps the base's documents minus the dropped ones by reading their stage
        outputs, and processes only what its source adds.
        """

        plan = self._plan_revision(revision) if revision is not None else None
        files = self._iter_input_files()
        if plan is not None:
            files = plan.without_known(files)
        first_file = next(files, None)
        if first_file is None and plan is None:
            raise ValueError("distributed run source did not contain any supported files")
        if first_file is None and plan is not None and not plan.changes_anything():
            held = f"; the {len(plan.skipped)} offered are already in it" if plan.skipped else ""
            raise ValueError(
                "revision adds no new documents and drops none, "
                f"so the graph would not change{held}"
            )
        validated_files = _iter_unique_file_ids(
            chain((first_file,), files) if first_file is not None else files
        )
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
            tasks = self._iter_initial_tasks(effective_run_id, validated_files, plan)
            self.task_store.add_initial_tasks(effective_run_id, tasks)
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

    def _plan_revision(self, revision: RevisionRequest) -> _RevisionPlan:
        """Work out what a revision keeps from its base, and what it must not re-add.

        Only a succeeded run of this graph can be revised: its documents are the
        ones whose stage outputs exist. Their identities and byte checksums let
        the source's files be told apart into new, already held, and replaced.
        """

        try:
            base = self.task_store.get_run_summary(revision.base_run_id)
        except KeyError as exc:
            raise ValueError(f"unknown base run: {revision.base_run_id}") from exc
        if base.run.status != RunStatus.SUCCEEDED:
            raise ValueError(
                f"only a succeeded run can be revised; {revision.base_run_id} is "
                f"{base.run.status.value}"
            )
        if base.run.graph_id != self.settings.job.graph_id:
            raise ValueError(
                f"base run {revision.base_run_id} built graph {base.run.graph_id}, "
                f"not {self.settings.job.graph_id}"
            )
        kept: dict[str, set[str]] = {}
        own = {
            file_id
            for ref in self.artifact_store.list_run_artifacts(
                revision.base_run_id, {ArtifactKind.EXTRACTED_DOCUMENT}
            )
            for file_id in _file_ids_of(ref)
        }
        if own:
            kept[revision.base_run_id] = own
        for finalizer in self.task_store.get_stage_tasks(
            revision.base_run_id, TaskStage.FINALIZE_GRAPH
        ):
            for entry in _inherited_documents(finalizer.task.payload):
                kept.setdefault(entry.run_id, set()).update(entry.file_ids)
        held = {file_id for file_ids in kept.values() for file_id in file_ids}
        unknown = sorted(set(revision.drop_file_ids) - held)
        if unknown:
            raise ValueError(f"documents to drop are not in the graph: {', '.join(unknown)}")
        for file_ids in kept.values():
            file_ids.difference_update(revision.drop_file_ids)
        checksums: dict[str, str] = {}
        for source_run_id, file_ids in kept.items():
            for ref in self.artifact_store.list_run_artifacts(
                source_run_id, {ArtifactKind.SOURCE_DOCUMENT}, linked=False
            ):
                input_file = ref.metadata.get("input_file")
                if not isinstance(input_file, dict):
                    continue
                file_id = str(input_file.get("id", ""))
                checksum = str(input_file.get("checksum", ""))
                if file_id in file_ids and checksum:
                    checksums[checksum] = file_id
        return _RevisionPlan(kept=kept, checksums=checksums, dropped=set(revision.drop_file_ids))

    def _iter_initial_tasks(
        self,
        run_id: str,
        files: Iterable[InputFile],
        plan: _RevisionPlan | None = None,
    ) -> Iterator[TaskDefinition]:
        """Yield stable tasks from bounded, continuously replenished upload batches.

        Task and artifact identities derive from the run and file rather than
        insertion order. Emitting each upload as it completes therefore lets a
        slow object overlap later files without changing retry or graph semantics.
        """

        for file_batch in batched(files, _SOURCE_STAGING_BATCH_SIZE, strict=False):
            for file, source_ref in self._iter_staged_sources(run_id, file_batch):
                yield TaskDefinition(
                    id=stable_id("task", run_id, TaskStage.PREPARE_DOCUMENT.value, file.id),
                    run_id=run_id,
                    stage=TaskStage.PREPARE_DOCUMENT,
                    scope_id=file.id,
                    payload={"source_artifact_id": source_ref.id},
                    priority=self.settings.distributed.task_priority(20),
                    max_attempts=self.settings.distributed.max_attempts,
                )
        payload = _graph_output_payload(self.settings)
        # Read after the files streamed past: a file that replaced a kept
        # document has left the kept set by now.
        inherited = plan.inherited if plan is not None else []
        if inherited:
            payload["inherit"] = [entry.model_dump(mode="json") for entry in inherited]
        yield TaskDefinition(
            id=stable_id("task", run_id, TaskStage.FINALIZE_GRAPH.value),
            run_id=run_id,
            stage=TaskStage.FINALIZE_GRAPH,
            scope_id=self.settings.job.graph_id,
            payload=payload,
            priority=self.settings.distributed.task_priority(0),
            max_attempts=self.settings.distributed.max_attempts,
        )

    def _iter_input_files(self) -> Iterator[InputFile]:
        """Use streaming discovery when an adapter provides deterministic iteration."""

        if self.file_source is None:
            return iter(())
        if isinstance(self.file_source, IterableFileSource):
            return self.file_source.iter_files()
        files = sorted(
            self.file_source.list_files(),
            key=lambda item: (item.source_uri, item.id),
        )
        return iter(files)

    def _iter_staged_sources(
        self,
        run_id: str,
        files: Sequence[InputFile],
    ) -> Iterator[tuple[InputFile, ArtifactRef]]:
        """Yield source/reference pairs as bounded concurrent uploads complete.

        A planning batch contains at most 256 lightweight file records. Queuing
        that bounded metadata is cheap, while only ``workers`` calls can read and
        retain complete source bytes. Completion-order iteration immediately
        replaces a finished upload even when an earlier object remains slow.
        """

        workers = _source_staging_workers(files)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            uploads = {executor.submit(self._store_source, run_id, file): file for file in files}
            for upload in as_completed(uploads):
                yield uploads[upload], upload.result()

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


def _source_staging_workers(files: Sequence[InputFile]) -> int:
    """Bound upload concurrency by both object count and worst-case memory.

    ``ArtifactStore.put`` accepts immutable bytes, so each active staging worker
    must hold one complete source object while checksumming and uploading it.
    Typical papers retain sixteen-way object-store concurrency; unusually large
    files reduce the pool automatically so a submitter cannot transiently retain
    many gigabytes merely because the corpus contains several large documents.
    """

    largest_file = max(file.size_bytes for file in files)
    byte_limited_workers = max(
        1,
        _SOURCE_STAGING_MAX_IN_FLIGHT_BYTES // max(largest_file, 1),
    )
    return min(_SOURCE_STAGING_PARALLELISM, len(files), byte_limited_workers)


class _RevisionPlan:
    """What a revision keeps, per concrete run, and how to recognise what it holds."""

    def __init__(
        self,
        kept: dict[str, set[str]],
        checksums: dict[str, str],
        dropped: set[str],
    ) -> None:
        self.kept = kept
        self.checksums = checksums
        self.dropped = dropped
        self.skipped: list[InputFile] = []
        self.replaced: list[InputFile] = []

    @property
    def inherited(self) -> list[InheritedDocuments]:
        """The kept documents as the finalizer reads them, one entry per run."""

        return [
            InheritedDocuments(run_id=run_id, file_ids=sorted(file_ids))
            for run_id, file_ids in sorted(self.kept.items())
            if file_ids
        ]

    def changes_anything(self) -> bool:
        """Whether the revision would build a graph different from its base."""

        return bool(self.dropped or self.replaced)

    def without_known(self, files: Iterator[InputFile]) -> Iterator[InputFile]:
        """Skip files the graph already holds; let one with a held identity replace it.

        Equal bytes under any name are the same document, and re-extracting
        them would only cost the fleet. A file with a held identity but other
        bytes is the document updated, so the old version leaves the kept set.
        """

        for file in files:
            if file.checksum in self.checksums:
                self.skipped.append(file)
                continue
            replaced = False
            for file_ids in self.kept.values():
                if file.id in file_ids:
                    file_ids.discard(file.id)
                    replaced = True
            if replaced:
                self.replaced.append(file)
            yield file


def _file_ids_of(ref: ArtifactRef) -> list[str]:
    """The documents a stage artifact covers, from the metadata every stage writes."""

    file_ids = ref.metadata.get("file_ids")
    return [str(item) for item in file_ids] if isinstance(file_ids, list) else []


def _inherited_documents(payload: dict[str, object]) -> list[InheritedDocuments]:
    """Read a finalizer's inherited documents, absent on a run that inherits none."""

    entries = payload.get("inherit")
    if not isinstance(entries, list):
        return []
    return [InheritedDocuments.model_validate(entry) for entry in entries]


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
