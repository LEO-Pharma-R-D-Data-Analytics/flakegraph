"""Contracts for bounded executor-side provider concurrency."""

from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.spark_finalization import (
    SparkFinalizationRequest,
    _adaptive_provider_partitions,
    _clear_executor_provider_caches,
    _connected_component_rows,
    _effective_shuffle_partitions,
    _executor_embedding_provider,
    _executor_enrichment_llm_provider,
    _parallel_map_partition,
    _row_batches,
    _spark_application_name,
    _target_output_partitions,
)
from kg_processor.config.settings import Settings
from kg_processor.ports.llm import DescriptionMergeRequest, DescriptionMergeResult


def test_spark_application_identity_fences_every_durable_attempt() -> None:
    """Prevent retry collisions without trusting user-controlled run IDs."""

    first = _spark_application_name(
        SparkFinalizationRequest(
            run_id="Customer Run/with invalid Kubernetes characters" * 4,
            graph_id="graph",
            attempt=1,
        )
    )
    repeated = _spark_application_name(
        SparkFinalizationRequest(
            run_id="Customer Run/with invalid Kubernetes characters" * 4,
            graph_id="graph",
            attempt=1,
        )
    )
    retry = _spark_application_name(
        SparkFinalizationRequest(
            run_id="Customer Run/with invalid Kubernetes characters" * 4,
            graph_id="graph",
            attempt=2,
        )
    )

    assert first == repeated
    assert first != retry
    assert first.startswith("flakegraph-")
    assert len(first) == 27
    assert first.replace("-", "").isalnum()


def test_parallel_partition_map_refills_capacity_behind_a_slow_request() -> None:
    """Keep submitting partition work while an earlier provider call is slow."""

    third_started = Event()

    def operation(value: int) -> int:
        """Make the first call depend on replacement work entering the pool."""

        if value == 0:
            assert third_started.wait(timeout=2)
        elif value == 2:
            third_started.set()
        return value * 10

    results = list(_parallel_map_partition([0, 1, 2], operation, parallelism=2))

    assert sorted(results) == [0, 10, 20]
    assert third_started.is_set()


def test_parallel_partition_map_rejects_invalid_worker_limit() -> None:
    """Fail configuration errors before consuming a Spark partition."""

    with pytest.raises(ValueError, match="parallelism must be positive"):
        list(_parallel_map_partition([1], lambda value: value, parallelism=0))


def test_row_batches_streams_a_short_final_batch() -> None:
    """Keep identity adjudication requests bounded without dropping tail rows."""

    assert list(_row_batches(iter(range(5)), batch_size=2)) == [[0, 1], [2, 3], [4]]

    with pytest.raises(ValueError, match="batch size must be positive"):
        list(_row_batches(iter([1]), batch_size=0))


def test_bounded_connected_components_preserve_singletons_and_transitive_links() -> None:
    """Match connected-component semantics without depending on edge input order."""

    vertices = ["d", "c", "b", "a", "single"]
    forward = _connected_component_rows(vertices, [("b", "c"), ("a", "b"), ("d", "c")])
    reverse = _connected_component_rows(vertices, [("d", "c"), ("a", "b"), ("b", "c")])

    expected = [
        ("a", "a"),
        ("b", "a"),
        ("c", "a"),
        ("d", "a"),
        ("single", "single"),
    ]
    assert forward == expected
    assert reverse == expected


def test_bounded_connected_components_reject_unknown_edge_endpoint() -> None:
    """Surface corrupt identity edges before creating incomplete component rows."""

    with pytest.raises(ValueError, match="outside the bounded graph"):
        _connected_component_rows(["known"], [("known", "missing")])


def test_provider_partitions_reduce_stragglers_without_unbounded_task_counts() -> None:
    """Size network work more finely while bounding scheduler load by fleet size."""

    # Four four-core executors get at least two work units per slot.
    assert _adaptive_provider_partitions(10, 4, 4, provider_batch_size=1) == 10
    assert _adaptive_provider_partitions(14_000, 4, 4, provider_batch_size=20) == 700
    # Partition counts are based on provider requests, not input rows. These
    # representative enrichment sizes retain four real batches per Spark task.
    assert _adaptive_provider_partitions(2_068, 4, 4, provider_batch_size=8) == 259
    assert _adaptive_provider_partitions(698, 4, 4, provider_batch_size=2) == 349
    # Extremely large provider stages stop at 256 tasks per execution slot.
    assert _adaptive_provider_partitions(1_000_000, 4, 4, provider_batch_size=1) == 4_096
    assert _adaptive_provider_partitions(0, 4, 4, provider_batch_size=1) == 1


def test_explicit_shuffle_partition_override_remains_authoritative() -> None:
    """Preserve measured operator tuning instead of replacing it from row counts."""

    assert _effective_shuffle_partitions(10_000_000, 4, 4, 73) == 73
    assert _effective_shuffle_partitions(10_000_000, 4, 4, 0) >= 16
    with pytest.raises(ValueError, match="must not be negative"):
        _effective_shuffle_partitions(10, 1, 1, -1)


def test_output_partition_policy_coalesces_without_adding_a_shuffle() -> None:
    """Avoid tiny files while allowing record caps to split oversized partitions."""

    assert _target_output_partitions(49, 16, 250_000) == 1
    assert _target_output_partitions(250_000, 16, 25_000) == 10
    assert _target_output_partitions(2_500_000, 16, 25_000) == 16
    assert _target_output_partitions(0, 16, 25_000) == 1


@pytest.mark.parametrize(
    ("rows", "instances", "cores", "batch_size"),
    [(-1, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)],
)
def test_provider_partitions_reject_invalid_capacity(
    rows: int,
    instances: int,
    cores: int,
    batch_size: int,
) -> None:
    """Reject invalid sizing inputs before they reach a Spark repartition call."""

    with pytest.raises(ValueError):
        _adaptive_provider_partitions(rows, instances, cores, batch_size)


def test_executor_embedding_provider_is_reused_for_identical_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid rebuilding an adapter or reloading model weights for every partition."""

    settings = Settings.load(
        env={},
        overrides={
            "llm": {"provider": "fake", "model": "fake"},
            "embedding": {"provider": "hash", "dimension": 8},
        },
    )
    built: list[object] = []

    def build(_settings: Settings) -> object:
        """Record factory calls and return a unique provider sentinel."""

        provider = object()
        built.append(provider)
        return provider

    _clear_executor_provider_caches()
    monkeypatch.setattr("kg_processor.factories.build_embedding_provider", build)
    try:
        first = _executor_embedding_provider(settings)
        second = _executor_embedding_provider(settings)
    finally:
        _clear_executor_provider_caches()

    assert first is second
    assert built == [first]


def test_spark_enrichment_provider_reuses_durable_cache_across_worker_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class CountingLlm(FakeLlmProvider):
        def __init__(self) -> None:
            self.calls = 0

        def merge_entity_description(
            self,
            request: DescriptionMergeRequest,
        ) -> DescriptionMergeResult:
            self.calls += 1
            return super().merge_entity_description(request)

    settings = Settings.load(
        env={},
        overrides={
            "job": {"graph_id": "graph"},
            "llm": {"provider": "fake", "model": "fake"},
            "cache": {"provider": "local", "path": tmp_path / "cache"},
        },
    )
    built: list[CountingLlm] = []

    def build(_settings: Settings) -> CountingLlm:
        provider = CountingLlm()
        built.append(provider)
        return provider

    request = DescriptionMergeRequest(
        entity_name="Alice",
        entity_type="PERSON",
        descriptions=["Alice", "Alice is an engineer."],
        evidence=["Alice works at Acme."],
    )
    monkeypatch.setattr("kg_processor.factories.build_llm_provider", build)
    _clear_executor_provider_caches()
    try:
        first = _executor_enrichment_llm_provider(settings)
        first.merge_entity_description(request)
        _clear_executor_provider_caches()
        second = _executor_enrichment_llm_provider(settings)
        second.merge_entity_description(request)
    finally:
        _clear_executor_provider_caches()

    assert len(built) == 2
    assert sum(provider.calls for provider in built) == 1
