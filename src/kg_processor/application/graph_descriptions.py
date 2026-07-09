"""LLM-assisted synthesis of canonical entity descriptions.

Descriptions are merged only after observations have survived filtering and
canonicalization. The trace records request sizes and provider metadata so a
reviewer can connect a node description back to the exact LLM prompt contract.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from kg_processor.application.graph_merge import normalize_entity_name
from kg_processor.domain.graph import EntitySource, Evidence, ExtractedEntity, GraphNode
from kg_processor.ports.llm import DescriptionMergeRequest, LlmProvider

MAX_DESCRIPTION_LENGTH = 1000


@dataclass(frozen=True)
class DescriptionSynthesisResult:
    """Count and trace events from optional entity-description synthesis."""

    merged_count: int
    trace_events: list[dict[str, Any]]


def synthesize_entity_descriptions(
    nodes: list[GraphNode],
    entity_sources: list[EntitySource],
    observations: list[ExtractedEntity],
    evidence: list[Evidence],
    llm: LlmProvider,
    min_observations: int,
    max_descriptions: int,
    max_evidence: int,
) -> DescriptionSynthesisResult:
    """Use the LLM port to synthesize richer descriptions for repeated entities."""

    descriptions_by_key = _descriptions_by_node_key(observations)
    evidence_by_subject = _evidence_by_subject(evidence)
    sources_by_node_id = _sources_by_node_id(entity_sources)
    merged_count = 0
    trace_events: list[dict[str, Any]] = []
    for node in nodes:
        descriptions = descriptions_by_key.get((node.normalized_name, node.primary_type), [])
        if len(descriptions) < min_observations:
            continue
        descriptions_used = descriptions[:max_descriptions]
        evidence_used = evidence_by_subject.get(node.id, [])[:max_evidence]
        result = llm.merge_entity_description(
            DescriptionMergeRequest(
                entity_name=node.name,
                entity_type=node.primary_type,
                descriptions=descriptions_used,
                evidence=evidence_used,
            )
        )
        description = result.description.strip()[:MAX_DESCRIPTION_LENGTH]
        if not description:
            trace_events.append(
                _trace_event(
                    node,
                    "skipped_empty_description",
                    descriptions,
                    descriptions_used,
                    evidence_used,
                    result.provider_metadata,
                    description_length=0,
                )
            )
            continue
        node.description = description
        for source in sources_by_node_id.get(node.id, []):
            source.per_file_description = description
        merged_count += 1
        trace_events.append(
            _trace_event(
                node,
                "merged",
                descriptions,
                descriptions_used,
                evidence_used,
                result.provider_metadata,
                description_length=len(description),
            )
        )
    return DescriptionSynthesisResult(merged_count=merged_count, trace_events=trace_events)


def _trace_event(
    node: GraphNode,
    action: str,
    descriptions: list[str],
    descriptions_used: list[str],
    evidence_used: list[str],
    provider_metadata: dict[str, Any],
    description_length: int,
) -> dict[str, Any]:
    return {
        "stage": "description_merge",
        "action": action,
        "node_id": node.id,
        "entity_name": node.name,
        "entity_type": node.primary_type,
        "observations_available": len(descriptions),
        "descriptions_used": len(descriptions_used),
        "evidence_used": len(evidence_used),
        "description_length": description_length,
        "provider_metadata": provider_metadata,
    }


def _descriptions_by_node_key(
    observations: list[ExtractedEntity],
) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for observation in observations:
        description = observation.description.strip()
        if not description:
            continue
        key = (normalize_entity_name(observation.name), observation.type)
        if description not in grouped[key]:
            grouped[key].append(description)
    return dict(grouped)


def _evidence_by_subject(evidence: list[Evidence]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in evidence:
        quote = item.quote.strip()
        if quote and quote not in grouped[item.subject_id]:
            grouped[item.subject_id].append(quote)
    return dict(grouped)


def _sources_by_node_id(entity_sources: list[EntitySource]) -> dict[str, list[EntitySource]]:
    grouped: dict[str, list[EntitySource]] = defaultdict(list)
    for source in entity_sources:
        grouped[source.node_id].append(source)
    return dict(grouped)
