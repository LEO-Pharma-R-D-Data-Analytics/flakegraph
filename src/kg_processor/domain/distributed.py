# SPDX-License-Identifier: Apache-2.0
"""Durable run, task, lease, and artifact models for distributed execution.

These records describe FlakeGraph workflow semantics without naming Kubernetes,
PostgreSQL, Snowflake, or a particular worker implementation. Backends may provide
at-least-once delivery, while deterministic task IDs and immutable artifacts make
successful retries idempotent.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskStage(StrEnum):
    """Name independently leased pipeline work that may run on separate workers."""

    PREPARE_DOCUMENT = "prepare_document"
    EXTRACT_DOCUMENT_CONTEXT = "extract_document_context"
    EXTRACT_ENTITY_WINDOW = "extract_entity_window"
    COMPACT_ENTITY_INVENTORY = "compact_entity_inventory"
    EXTRACT_RELATION_WINDOW = "extract_relation_window"
    COMPACT_DOCUMENT = "compact_document"
    FINALIZE_GRAPH = "finalize_graph"


# The order workers claim ready work in, higher first, across every run - the
# stage ladder, which stays within 0-20 so a run's priority offset keeps bulk
# and interactive work apart.
#
# The two compaction stages come first. They make no model call and finish in
# moments, and each one unlocks the next phase of its document: an entity
# inventory releases the document's relation windows, a compacted document
# brings its run closer to finalizing. Ranked below the model stages they
# share a pool with, they would wait behind every window of every other run -
# a finished run could sit for hours with nothing left but bookkeeping.
# Among the model stages, earlier ones rank higher so documents keep flowing
# into the pipeline. Preparation and finalizing have pools of their own.
STAGE_PRIORITY: dict[TaskStage, int] = {
    TaskStage.PREPARE_DOCUMENT: 20,
    TaskStage.COMPACT_ENTITY_INVENTORY: 19,
    TaskStage.COMPACT_DOCUMENT: 18,
    TaskStage.EXTRACT_DOCUMENT_CONTEXT: 15,
    TaskStage.EXTRACT_ENTITY_WINDOW: 10,
    TaskStage.EXTRACT_RELATION_WINDOW: 4,
    TaskStage.FINALIZE_GRAPH: 0,
}


class RunStatus(StrEnum):
    """Represent the externally visible lifecycle of one graph build."""

    PLANNING = "planning"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(StrEnum):
    """Represent durable task state under at-least-once worker execution."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PublicationStatus(StrEnum):
    """Represent durable delivery of a finalized graph to an external destination."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactKind(StrEnum):
    """Identify immutable payload schemas stored between worker stages."""

    SOURCE_DOCUMENT = "source_document"
    PREPARED_DOCUMENT = "prepared_document"
    DOCUMENT_CONTEXT = "document_context"
    EXTRACTION_WINDOW = "extraction_window"
    EXTRACTED_ENTITY_WINDOW = "extracted_entity_window"
    DOCUMENT_ENTITY_INVENTORY = "document_entity_inventory"
    EXTRACTED_RELATION_WINDOW = "extracted_relation_window"
    EXTRACTED_DOCUMENT = "extracted_document"
    GRAPH_RESULT = "graph_result"


class RunDefinition(BaseModel):
    """Describe a graph run before its task graph becomes executable."""

    id: str
    graph_id: str
    config: dict[str, Any]
    config_digest: str
    status: RunStatus = RunStatus.PLANNING

    @field_validator("id", "graph_id", "config_digest")
    @classmethod
    def identity_fields_must_not_be_blank(cls, value: str) -> str:
        """Reject ambiguous durable identities before database insertion."""

        if not value.strip():
            raise ValueError("distributed run identity fields must not be blank")
        return value


class InheritedDocuments(BaseModel):
    """Documents a run keeps from an earlier run of the same graph.

    A revision re-extracts only what it adds; everything it keeps is read from
    the stage outputs the earlier run already produced, so the finalizer builds
    the graph from both. Entries name concrete runs: a revision of a revision
    inherits from each original run directly rather than through a chain.
    """

    run_id: str
    file_ids: list[str] = Field(min_length=1)

    @field_validator("run_id")
    @classmethod
    def run_id_must_not_be_blank(cls, value: str) -> str:
        """Refuse an inheritance that names no run."""

        if not value.strip():
            raise ValueError("inherited documents must name their run")
        return value


class RevisionRequest(BaseModel):
    """Ask for a new run of a graph that keeps an earlier run's documents.

    ``drop_file_ids`` are removed from what is kept; whatever the file source
    discovers is added, with a document whose bytes the graph already holds
    skipped and one whose identity it holds replaced.
    """

    base_run_id: str
    drop_file_ids: list[str] = Field(default_factory=list)
    # Build a new version even though no document changes. The documents are
    # the same, but how a graph is built from them has changed - a finalizer
    # fix - and the only way to apply it to an existing graph is to finalize
    # it again. Nothing is extracted again: kept documents are read from the
    # base run's stage outputs, as for any revision.
    rebuild: bool = False


class TaskDefinition(BaseModel):
    """Describe one idempotent stage invocation and its dependency barrier."""

    id: str
    run_id: str
    stage: TaskStage
    scope_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    dependency_ids: list[str] = Field(default_factory=list)
    priority: int = 0
    max_attempts: int = Field(default=3, ge=1)

    @field_validator("id", "run_id", "scope_id")
    @classmethod
    def task_identity_fields_must_not_be_blank(cls, value: str) -> str:
        """Require stable identifiers for leases, retries, and audit events."""

        if not value.strip():
            raise ValueError("distributed task identity fields must not be blank")
        return value


class TaskLease(BaseModel):
    """Carry one claimed task plus completed dependency outputs to a worker."""

    task: TaskDefinition
    worker_id: str
    attempt: int = Field(ge=1)
    lease_expires_at: datetime
    dependency_outputs: dict[str, list[str]] = Field(default_factory=dict)


class PublicationLease(BaseModel):
    """Carry one durable graph-publication command and its fencing generation."""

    id: str
    run_id: str
    artifact_id: str
    task_payload: dict[str, Any]
    worker_id: str
    attempt: int = Field(ge=1)
    generation: int = Field(ge=1)
    lease_expires_at: datetime


class TaskProgress(BaseModel):
    """Describe bounded progress within one otherwise indivisible queue task.

    Most stages fan out into many independently countable tasks. Graph
    finalization is intentionally one run-wide barrier, so it publishes its
    current phase here instead of inventing queue tasks that could execute out of
    order or duplicate graph-wide work.
    """

    phase: str
    phase_index: int = Field(ge=1)
    phase_total: int = Field(ge=1)
    completed: int = Field(default=0, ge=0)
    total: int | None = Field(default=None, ge=1)
    message: str | None = None

    @field_validator("phase")
    @classmethod
    def phase_must_not_be_blank(cls, value: str) -> str:
        """Reject progress records that cannot identify their current work."""

        normalized = value.strip()
        if not normalized:
            raise ValueError("task progress phase must not be blank")
        return normalized

    @model_validator(mode="after")
    def counters_must_be_consistent(self) -> TaskProgress:
        """Keep phase and optional inner-work counters internally consistent."""

        if self.phase_index > self.phase_total:
            raise ValueError("task progress phase_index must not exceed phase_total")
        if self.total is not None and self.completed > self.total:
            raise ValueError("task progress completed must not exceed total")
        return self


class TaskSnapshot(BaseModel):
    """Expose task state for status commands, diagnostics, and progress UIs."""

    task: TaskDefinition
    status: TaskStatus
    attempts: int = Field(ge=0)
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    output_artifact_ids: list[str] = Field(default_factory=list)
    last_error: dict[str, Any] | None = None
    progress: TaskProgress | None = None


class RunSnapshot(BaseModel):
    """Expose a run and all of its tasks as one consistent status response."""

    run: RunDefinition
    tasks: list[TaskSnapshot] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class TaskCount(BaseModel):
    """Aggregate one stage/status bucket without materializing individual tasks."""

    stage: TaskStage
    status: TaskStatus
    count: int = Field(ge=0)
    # Of a queued bucket, how many a worker could claim right now: their
    # dependencies are met and no retry delay is pending. The rest wait on
    # earlier stages, so no pool is short of workers on their account.
    ready: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: TaskProgress | None = None


class RunSummary(BaseModel):
    """Expose constant-size run progress suitable for very large task graphs.

    Operators normally need stage progress, not hundreds of thousands of task
    payloads. Detailed snapshots remain available for explicit diagnosis while
    submit, status, cancel, and retry can return this bounded representation. A
    terminal run carries only its single redacted root error, never task payloads.
    """

    run: RunDefinition
    task_counts: list[TaskCount] = Field(default_factory=list)
    total_tasks: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    error: dict[str, Any] | None = None
    # Documents whose preparation gave up and recorded them as unreadable. Their
    # tasks succeed, so the task counts alone cannot say so.
    documents_failed: int = Field(default=0, ge=0)
    # The configuration digest each stage's worker fleet declared at startup.
    # A worker claims only tasks whose run digest equals its own, so a stage
    # served at another digest is one this run's remaining work cannot leave.
    fleet_config_digests: dict[str, str] = Field(default_factory=dict)


class RunOverview(BaseModel):
    """Describe one recent run without returning configuration or task payloads."""

    id: str
    graph_id: str
    status: RunStatus
    task_counts: list[TaskCount] = Field(default_factory=list)
    total_tasks: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


class GraphDeletion(BaseModel):
    """What deleting a graph removed: every run of it and all they stored.

    Counts are of what this call removed, so deleting a graph that is already
    gone reports zeros rather than failing.
    """

    graph_id: str
    run_ids: list[str] = Field(default_factory=list)
    tasks: int = Field(default=0, ge=0)
    artifacts: int = Field(default=0, ge=0)
    objects: int = Field(default=0, ge=0)


class ArtifactRef(BaseModel):
    """Identify an immutable content-addressed stage payload."""

    id: str
    run_id: str
    kind: ArtifactKind
    media_type: str
    checksum: str
    size_bytes: int = Field(ge=0)
    storage_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredArtifact(BaseModel):
    """Return artifact metadata and exact uncompressed payload bytes."""

    ref: ArtifactRef
    payload: bytes
