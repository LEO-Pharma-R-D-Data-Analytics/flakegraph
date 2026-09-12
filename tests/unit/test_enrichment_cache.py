"""Regression tests for durable post-extraction LLM checkpoints."""

from pathlib import Path

from kg_processor.adapters.cache.local_json import LocalJsonCache
from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.enrichment_cache import CachedEnrichmentLlmProvider
from kg_processor.ports.llm import DescriptionMergeRequest, DescriptionMergeResult


class _CountingLlm(FakeLlmProvider):
    def __init__(self) -> None:
        self.description_calls = 0

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        self.description_calls += 1
        return super().merge_entity_description(request)


def test_enrichment_wrapper_reuses_description_result_after_restart(tmp_path: Path) -> None:
    cache = LocalJsonCache(tmp_path)
    delegate = _CountingLlm()
    request = DescriptionMergeRequest(
        entity_name="Alice",
        entity_type="PERSON",
        descriptions=["Alice", "Alice is an engineer."],
        evidence=["Alice works at Acme."],
    )

    first = CachedEnrichmentLlmProvider(
        delegate,
        cache,
        graph_id="graph",
        provider="fake",
        model="fake",
    ).merge_entity_description(request)
    second = CachedEnrichmentLlmProvider(
        delegate,
        cache,
        graph_id="graph",
        provider="fake",
        model="fake",
    ).merge_entity_description(request)

    assert first == second
    assert delegate.description_calls == 1
