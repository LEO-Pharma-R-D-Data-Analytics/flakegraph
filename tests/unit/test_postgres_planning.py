"""Fast contract tests for PostgreSQL plan validation helpers."""

from __future__ import annotations

import pytest

from kg_processor.adapters.distributed.postgres import (
    PostgresDistributedStore,
    _is_initial_document_plan,
    _validate_acyclic,
)
from kg_processor.domain.distributed import TaskDefinition, TaskStage


def test_large_initial_plan_is_accepted_without_quadratic_graph_scans() -> None:
    """Validate the corpus-shaped DAG used by large distributed submissions.

    The size is large enough to make an accidental all-pairs implementation
    noticeable in the unit suite while remaining inexpensive for the intended
    linear algorithm. Runtime is not asserted because shared CI timing is noisy;
    completing this input is the regression contract.
    """

    run_id = "large-plan"
    preparations = [
        TaskDefinition(
            id=f"prepare-{index}",
            run_id=run_id,
            stage=TaskStage.PREPARE_DOCUMENT,
            scope_id=f"file-{index}",
        )
        for index in range(20_000)
    ]
    final = TaskDefinition(
        id="final",
        run_id=run_id,
        stage=TaskStage.FINALIZE_GRAPH,
        scope_id="graph",
    )

    _validate_acyclic([*preparations, final])


def test_initial_document_plan_uses_the_bounded_shape_validator() -> None:
    """Recognize only the exact large-corpus plan eligible for the memory fast path."""

    run_id = "document-plan"
    preparations = [
        TaskDefinition(
            id=f"prepare-{index}",
            run_id=run_id,
            stage=TaskStage.PREPARE_DOCUMENT,
            scope_id=f"file-{index}",
        )
        for index in range(3)
    ]
    final = TaskDefinition(
        id="final",
        run_id=run_id,
        stage=TaskStage.FINALIZE_GRAPH,
        scope_id="graph",
    )

    assert _is_initial_document_plan([*preparations, final]) is True
    assert (
        _is_initial_document_plan(
            [*preparations, final.model_copy(update={"dependency_ids": [preparations[0].id]})]
        )
        is False
    )


def test_cycle_is_rejected_before_tasks_reach_postgres() -> None:
    """Keep a malformed plan from creating workers that wait forever."""

    run_id = "cyclic-plan"
    first = TaskDefinition(
        id="first",
        run_id=run_id,
        stage=TaskStage.PREPARE_DOCUMENT,
        scope_id="first-file",
        dependency_ids=["second"],
    )
    second = TaskDefinition(
        id="second",
        run_id=run_id,
        stage=TaskStage.PREPARE_DOCUMENT,
        scope_id="second-file",
        dependency_ids=["first"],
    )

    with pytest.raises(ValueError, match="must be acyclic"):
        _validate_acyclic([first, second])


def test_recent_run_list_rejects_unbounded_limits_before_connecting() -> None:
    """Keep the operator-facing history query bounded independently of corpus size."""

    store = PostgresDistributedStore("postgresql://example.invalid/flakegraph")

    with pytest.raises(ValueError, match="between 1 and 500"):
        store.list_runs(0)
    with pytest.raises(ValueError, match="between 1 and 500"):
        store.list_runs(501)
