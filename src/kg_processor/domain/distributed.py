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


class RunOverview(BaseModel):
    """Describe one recent run without returning configuration or task payloads."""

    id: str
    graph_id: str
    status: RunStatus
    task_counts: list[TaskCount] = Field(default_factory=list)
    total_tasks: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime


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
