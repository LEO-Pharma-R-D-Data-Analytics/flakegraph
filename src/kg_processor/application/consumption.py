"""Collect what a run consumed while it runs.

Only real provider work is recorded. A cache hit returns the same artifact
without calling anybody, so recording it would inflate spend for work that was
never billed and would make a second run of an unchanged corpus look as expensive
as the first — destroying the comparison the cache exists to enable.

Collection is thread-safe because extraction fans windows out concurrently.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence

from kg_processor.domain.consumption import (
    ConsumptionEvent,
    ConsumptionTotals,
    Locality,
    RateCard,
    TokenUsage,
    group_by,
    summarize,
)

# Providers that run on infrastructure we own. Their calls are recorded exactly
# like hosted ones — the tokens are what make the avoided cost measurable — but
# they accrue avoided rather than billed cost.
_LOCAL_PROVIDERS = frozenset(
    {
        "ollama",
        "vllm_local",
        "sentence_transformers",
        "builtin_text",
        "mineru",
        "mineru_internal",
        "tesseract",
        "gliner",
    }
)


def locality_for(provider: str) -> Locality:
    """Classify a provider by where its work actually runs."""

    return Locality.LOCAL if provider in _LOCAL_PROVIDERS else Locality.HOSTED


class ConsumptionCollector:
    """Accumulate consumption events for one run."""

    def __init__(self, graph_id: str, run_id: str = "", job_id: str = "") -> None:
        self.graph_id = graph_id
        self.run_id = run_id
        self.job_id = job_id
        self._events: list[ConsumptionEvent] = []
        self._lock = threading.Lock()

    def record(
        self,
        *,
        stage: str,
        operation: str,
        provider: str,
        model: str,
        usage: TokenUsage | None = None,
        pages: int = 0,
        calls: int = 1,
        file_id: str | None = None,
    ) -> None:
        """Record one unit of provider work that actually executed."""

        event = ConsumptionEvent(
            graph_id=self.graph_id,
            run_id=self.run_id,
            job_id=self.job_id,
            stage=stage,
            operation=operation,
            provider=provider,
            model=model,
            locality=locality_for(provider),
            calls=calls,
            usage=usage or TokenUsage(),
            pages=pages,
            file_id=file_id,
        )
        with self._lock:
            self._events.append(event)

    def record_completion(
        self,
        *,
        stage: str,
        provider: str,
        model: str,
        metadata: Mapping[str, object] | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        """Record one LLM call, reading usage from the result or its metadata."""

        # An all-zero usage means the adapter reported nothing, not that the call
        # was free. Falling back to the provider's own metadata keeps a forgotten
        # ``usage=`` from silently costing nothing — which is indistinguishable
        # from a genuinely free call once it reaches a total.
        resolved = usage if usage and usage.total_tokens else None
        if resolved is None and metadata is not None:
            resolved = TokenUsage.from_provider_payload(metadata.get("usage")) or usage
        self.record(
            stage=stage,
            operation="completion",
            provider=provider,
            model=model,
            usage=resolved,
        )

    @property
    def events(self) -> Sequence[ConsumptionEvent]:
        """Return the events recorded so far."""

        with self._lock:
            return list(self._events)

    def running_totals(self, card: RateCard) -> dict[str, object]:
        """Summarize what the run has spent so far, without its event detail.

        The full report carries every recorded call so the answer can be
        re-derived later. That is the right shape to persist once beside a
        finished graph and the wrong shape to write every few seconds while the
        run is still going, so progress reporting takes the totals alone.
        """

        return dict(summarize(self.events, card).model_dump(mode="json"))

    def report(self, card: RateCard) -> dict[str, object]:
        """Summarize the run for persistence beside its graph.

        Both the totals and the events are returned. Totals answer "what did this
        graph cost"; the events are what let that answer be re-derived when the
        rate table is corrected, or broken down by stage, provider or model
        without re-running anything.
        """

        events = self.events
        totals: ConsumptionTotals = summarize(events, card)
        return {
            "totals": totals.model_dump(mode="json"),
            "by_stage": {
                name: value.model_dump(mode="json")
                for name, value in group_by(events, "stage", card).items()
            },
            "by_provider": {
                name: value.model_dump(mode="json")
                for name, value in group_by(events, "provider", card).items()
            },
            "by_model": {
                name: value.model_dump(mode="json")
                for name, value in group_by(events, "model", card).items()
            },
            "by_locality": {
                name: value.model_dump(mode="json")
                for name, value in group_by(events, "locality", card).items()
            },
            "events": [event.model_dump(mode="json") for event in events],
        }


class NullConsumptionCollector(ConsumptionCollector):
    """Record nothing, for callers that do not track consumption."""

    def record(self, **_kwargs: object) -> None:
        """Ignore the event."""

        return None
