"""Behavioral contracts for request-efficient graph enrichment batches."""

from __future__ import annotations

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.enrichment_batching import (
    merge_description_requests,
    summarize_community_requests,
)
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    DescriptionMergeRequest,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)


class _CountingFakeLlm(FakeLlmProvider):
    """Count batch and fallback calls while retaining deterministic fake behavior."""

    def __init__(self) -> None:
        self.structured_calls = 0
        self.description_fallback_calls = 0
        self.community_fallback_calls = 0

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        self.structured_calls += 1
        return super().complete_structured(request)

    def merge_entity_description(self, request: DescriptionMergeRequest):  # type: ignore[no-untyped-def]
        self.description_fallback_calls += 1
        return super().merge_entity_description(request)

    def summarize_community(self, request: CommunitySummaryRequest):  # type: ignore[no-untyped-def]
        self.community_fallback_calls += 1
        return super().summarize_community(request)


class _IncompleteBatchLlm(_CountingFakeLlm):
    """Return schema-valid but incomplete first records to exercise recovery."""

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        result = super().complete_structured(request)
        records = result.payload.get("results")
        assert isinstance(records, list) and isinstance(records[0], dict)
        if request.task_name == "entity_description_batch_merge":
            records[0]["description"] = ""
        elif request.task_name == "community_report_batch":
            records[0]["findings"] = []
        return result


def test_description_batch_uses_one_structured_call_and_preserves_order() -> None:
    """Reduce transport calls without changing input-to-output correspondence."""

    provider = _CountingFakeLlm()
    requests = [
        DescriptionMergeRequest(
            entity_name=f"Entity {index}",
            entity_type="CONCEPT",
            descriptions=["short", f"long description {index}"],
            evidence=[],
        )
        for index in range(16)
    ]

    results = merge_description_requests(
        provider,
        requests,
        model="fake",
        timeout_seconds=30,
        seed=17,
    )

    assert provider.structured_calls == 1
    assert provider.description_fallback_calls == 0
    assert [result.description for result in results] == [
        f"long description {index}" for index in range(16)
    ]


def test_community_batch_uses_one_structured_call_for_four_reports() -> None:
    """Batch four independently bounded community contexts in one request."""

    provider = _CountingFakeLlm()
    requests = [
        CommunitySummaryRequest(
            title_seed=title,
            members=[f"{title} member"],
            relations=[],
        )
        for title in ("First", "Second", "Third", "Fourth")
    ]

    results = summarize_community_requests(
        provider,
        requests,
        model="fake",
        timeout_seconds=30,
        seed=17,
    )

    assert provider.structured_calls == 1
    assert provider.community_fallback_calls == 0
    assert [result.title for result in results] == ["First", "Second", "Third", "Fourth"]


def test_incomplete_batch_records_fall_back_without_replaying_siblings() -> None:
    """Recover empty enrichment records individually while retaining valid rows."""

    provider = _IncompleteBatchLlm()
    descriptions = [
        DescriptionMergeRequest(
            entity_name=f"Entity {index}",
            entity_type="CONCEPT",
            descriptions=[f"description {index}"],
            evidence=[],
        )
        for index in range(2)
    ]
    communities = [
        CommunitySummaryRequest(
            title_seed=f"Community {index}",
            members=[f"member {index}"],
            # Co-mention communities can have grounded member context without an
            # internal relation. Empty findings must still recover through the
            # established single-community path.
            relations=[],
            evidence_quotes=[],
        )
        for index in range(2)
    ]

    description_results = merge_description_requests(
        provider,
        descriptions,
        model="fake",
        timeout_seconds=30,
        seed=17,
    )
    community_results = summarize_community_requests(
        provider,
        communities,
        model="fake",
        timeout_seconds=30,
        seed=17,
    )

    assert provider.structured_calls == 2
    assert provider.description_fallback_calls == 1
    assert provider.community_fallback_calls == 1
    assert all(result.description for result in description_results)
    assert all(result.summary for result in community_results)
