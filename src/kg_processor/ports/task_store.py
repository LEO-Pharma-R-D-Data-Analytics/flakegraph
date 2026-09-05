"""Durable task-store port used by distributed coordinators and workers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import timedelta
from typing import Any, Protocol, runtime_checkable

from kg_processor.domain.distributed import (
    PublicationLease,
    RunDefinition,
    RunOverview,
    RunSnapshot,
    RunSummary,
    TaskDefinition,
    TaskLease,
    TaskProgress,
    TaskStage,
)


class TaskStoreUnavailableError(RuntimeError):
    """Signal a transient coordination-store connectivity failure.

    Application workers use this provider-neutral exception to distinguish an
    unavailable queue from a malformed task or deterministic pipeline error.
    Long-running workers may therefore back off and reconnect without importing
    PostgreSQL-specific exception classes or restarting their containers.
    """


class TaskStore(Protocol):
    """Coordinate dependency-aware tasks through leases and atomic transitions."""

    def initialize(self) -> None:
        """Create or migrate durable coordination structures idempotently."""
        ...

    def create_run(self, run: RunDefinition) -> None:
        """Persist a planning run while rejecting conflicting run identities."""
        ...

    def add_tasks(self, run_id: str, tasks: list[TaskDefinition]) -> None:
        """Persist a complete validated task graph for a planning run."""
        ...

    def activate_run(self, run_id: str) -> None:
        """Make a fully planned run available to workers."""
        ...

    def claim_task(
        self,
        worker_id: str,
        stages: set[TaskStage],
        lease_duration: timedelta,
        expected_config_digest: str | None = None,
        skip_dependency_outputs_for_stages: set[TaskStage] | None = None,
    ) -> TaskLease | None:
        """Atomically claim a compatible highest-priority task for this worker.

        Stores must apply the optional configuration digest before incrementing a
        task attempt so a stale worker cannot exhaust healthy work's retry budget.
        Stages in ``skip_dependency_outputs_for_stages`` still wait for every
        dependency but omit their output IDs from the lease. This supports engines
        such as Spark that scan immutable stage prefixes directly.
        """
        ...

    def heartbeat(self, task_id: str, worker_id: str, lease_duration: timedelta) -> None:
        """Extend a running task lease only when the caller still owns it."""
        ...

    def complete_task(
        self,
        task_id: str,
        worker_id: str,
        output_artifact_ids: list[str],
        follow_up_tasks: list[TaskDefinition] | None = None,
        barrier_task_id: str | None = None,
    ) -> None:
        """Commit outputs and optional dynamically fanned-out work atomically.

        Follow-up tasks become dependencies of ``barrier_task_id`` in the same
        transaction that completes their parent. This prevents the barrier from
        racing ahead between preparation and extraction-window publication.
        """
        ...

    def fail_task(
        self,
        task_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
    ) -> None:
        """Record an error and either retry or terminally fail the owning run."""
        ...

    def cancel_run(self, run_id: str) -> None:
        """Cancel unfinished work while preserving completed artifacts and audit data."""
        ...

    def retry_run(self, run_id: str) -> None:
        """Requeue terminally failed tasks after an operator corrects their cause."""
        ...

    def get_run(self, run_id: str) -> RunSnapshot:
        """Return one consistent run and task snapshot or raise when unknown."""
        ...

    def get_run_summary(self, run_id: str) -> RunSummary:
        """Return bounded stage/status counts without loading every task payload."""
        ...

    def list_runs(self, limit: int = 100) -> list[RunOverview]:
        """Return recent bounded run overviews without configuration or task payloads."""
        ...


@runtime_checkable
class InitialTaskStreamWriter(Protocol):
    """Optional task-store capability for constant-memory corpus submission.

    The initial distributed plan has a deliberately restricted shape: one
    independent preparation task per source plus one run-wide final barrier.
    Stores that can validate and bulk-load that shape incrementally avoid forcing
    a coordinator to retain hundreds of thousands of Pydantic task objects.
    """

    def add_initial_tasks(self, run_id: str, tasks: Iterable[TaskDefinition]) -> None:
        """Validate and persist a streamed initial document plan atomically."""
        ...


@runtime_checkable
class TaskProgressWriter(Protocol):
    """Optional capability for durable progress within a long-running task."""

    def report_task_progress(
        self,
        task_id: str,
        worker_id: str,
        progress: TaskProgress,
    ) -> None:
        """Replace progress only while the caller owns the running task lease."""
        ...


@runtime_checkable
class FencedPublicationStore(Protocol):
    """Optional durable outbox for externally visible graph publication.

    ``publish_claimed`` must not hold a store lock across the callback: external
    delivery can outlast any lease, and blocking cancellation and lease
    reclamation behind it would make an unresponsive destination stall the run.

    A stale finalizer is fenced optimistically instead. The store validates
    ownership before the callback and publishes the lease's generation and
    attempt with it, then acknowledges completion only when the row still carries
    that same generation, attempt, owner, and live lease under a still-running
    run. A superseded or reclaimed finalizer therefore fails to acknowledge, and
    the destination uses the published generation to reject its late write.
    """

    def claim_publication(
        self,
        worker_id: str,
        lease_duration: timedelta,
    ) -> PublicationLease | None:
        """Lease one queued or expired publication command."""
        ...

    def publish_claimed(
        self,
        publication_id: str,
        worker_id: str,
        publish: Callable[[PublicationLease], None],
    ) -> None:
        """Run publication outside any store lock and acknowledge it under its fence."""
        ...

    def heartbeat_publication(
        self,
        publication_id: str,
        worker_id: str,
        lease_duration: timedelta,
    ) -> None:
        """Extend an active publication lease while external delivery is in progress."""
        ...

    def fail_publication(
        self,
        publication_id: str,
        worker_id: str,
        error: dict[str, Any],
        retry_delay: timedelta,
    ) -> None:
        """Record a failed delivery attempt and requeue it within its budget."""
        ...
