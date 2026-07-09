"""Canonical graph assembly from grounded extraction observations.

Merge decisions are explicit trace records. Stable ids are derived from graph id,
normalized entity/relation keys, and file scope so reruns and Snowflake MERGE
operations can be idempotent.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from kg_processor.domain.graph import (
    Chunk,
    EntitySource,
    Evidence,
    ExtractedEntity,
    ExtractedRelation,
    GraphEdge,
    GraphNode,
)
from kg_processor.domain.ids import stable_id

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_QUOTE_SEED_MIN_LEN = 3


@dataclass(frozen=True)
class MergeDecision:
    """Traceable record explaining how one observation affected graph assembly."""

    kind: Literal["entity", "relation"]
    action: Literal["created", "merged", "aggregated", "dropped"]
    reason: str
    graph_id: str
    node_id: str | None = None
    edge_id: str | None = None
    normalized_name: str | None = None
    primary_type: str | None = None
    source_name: str | None = None
    target_name: str | None = None
    relation_type: str | None = None
    source_chunk_id: str | None = None
    source_file_id: str | None = None
    observation_count: int | None = None
    source_chunk_ids: list[str] | None = None
    observed_names: list[str] | None = None
    original_weight: float | None = None
    effective_weight: float | None = None
    prior_weight: float | None = None
    new_weight: float | None = None

    def to_trace_event(self) -> dict[str, object]:
        """Serialize the merge decision into the extraction trace format."""

        event: dict[str, object] = {
            "stage": "merge_decision",
            "kind": self.kind,
            "action": self.action,
            "reason": self.reason,
            "graph_id": self.graph_id,
        }
        _set_if_present(event, "node_id", self.node_id)
        _set_if_present(event, "edge_id", self.edge_id)
        _set_if_present(event, "normalized_name", self.normalized_name)
        _set_if_present(event, "primary_type", self.primary_type)
        _set_if_present(event, "source_name", self.source_name)
        _set_if_present(event, "target_name", self.target_name)
        _set_if_present(event, "relation_type", self.relation_type)
        _set_if_present(event, "source_chunk_id", self.source_chunk_id)
        _set_if_present(event, "source_file_id", self.source_file_id)
        _set_if_present(event, "observation_count", self.observation_count)
        _set_if_present(event, "source_chunk_ids", self.source_chunk_ids)
        _set_if_present(event, "observed_names", self.observed_names)
        _set_if_present(event, "original_weight", self.original_weight)
        _set_if_present(event, "effective_weight", self.effective_weight)
        _set_if_present(event, "prior_weight", self.prior_weight)
        _set_if_present(event, "new_weight", self.new_weight)
        return event


@dataclass(frozen=True)
class GraphAssemblyResult:
    """Canonical graph rows plus merge evidence and review decisions."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    evidence: list[Evidence]
    entity_sources: list[EntitySource]
    decisions: list[MergeDecision]

    def decision_reason_counts(self) -> dict[str, int]:
        """Return sorted counts of merge decision reasons."""

        counter = Counter(decision.reason for decision in self.decisions)
        return dict(sorted(counter.items()))

    def decision_action_counts(self) -> dict[str, int]:
        """Return sorted counts of merge decision actions."""

        counter = Counter(decision.action for decision in self.decisions)
        return dict(sorted(counter.items()))


@dataclass(frozen=True)
class EntityAssembly:
    """Intermediate entity assembly state consumed by relation assembly."""

    nodes: list[GraphNode]
    node_by_name: dict[str, GraphNode]
    evidence: list[Evidence]
    entity_sources: list[EntitySource]
    decisions: list[MergeDecision]


@dataclass(frozen=True)
class RelationAssembly:
    """Intermediate relation assembly state returned before final graph sorting."""

    edges: list[GraphEdge]
    evidence: list[Evidence]
    decisions: list[MergeDecision]


def normalize_entity_name(name: str) -> str:
    """Normalize entity names into stable keys used for merging."""

    stripped = _NON_ALNUM_RE.sub("", name.lower())
    return stripped or name.strip().lower()


def normalize_relation_type(relation_type: str) -> str:
    """Normalize relation labels into bounded Snowflake-friendly identifiers."""

    return re.sub(r"\s+", "_", relation_type.strip().lower())[:80]


def assemble_graph(
    graph_id: str,
    chunks: list[Chunk],
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    relation_weight_max: float,
) -> tuple[list[GraphNode], list[GraphEdge], list[Evidence], list[EntitySource]]:
    """Assemble canonical graph rows without returning merge decisions."""

    result = assemble_graph_with_decisions(
        graph_id,
        chunks,
        entities,
        relations,
        relation_weight_max,
    )
    return result.nodes, result.edges, result.evidence, result.entity_sources


def assemble_graph_with_decisions(
    graph_id: str,
    chunks: list[Chunk],
    entities: list[ExtractedEntity],
    relations: list[ExtractedRelation],
    relation_weight_max: float,
) -> GraphAssemblyResult:
    """Merge grounded observations into canonical nodes, edges, and evidence."""

    # Entities are grouped before relation assembly so endpoint resolution can
    # use the same canonical node map that will be persisted.
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    grouped_entities: dict[tuple[str, str], list[ExtractedEntity]] = defaultdict(list)
    for entity in entities:
        grouped_entities[(normalize_entity_name(entity.name), entity.type)].append(entity)

    entity_assembly = _assemble_entities(graph_id, chunks_by_id, grouped_entities)
    relation_assembly = _assemble_relations(
        graph_id,
        chunks_by_id,
        entity_assembly.node_by_name,
        relations,
        relation_weight_max,
    )
    nodes = sorted(entity_assembly.nodes, key=lambda node: node.id)
    edges = relation_assembly.edges
    _assign_degrees(nodes, edges)
    return GraphAssemblyResult(
        nodes=nodes,
        edges=edges,
        evidence=[*entity_assembly.evidence, *relation_assembly.evidence],
        entity_sources=entity_assembly.entity_sources,
        decisions=[*entity_assembly.decisions, *relation_assembly.decisions],
    )


def _assemble_entities(
    graph_id: str,
    chunks_by_id: dict[str, Chunk],
    grouped_entities: dict[tuple[str, str], list[ExtractedEntity]],
) -> EntityAssembly:
    nodes: list[GraphNode] = []
    node_by_name: dict[str, GraphNode] = {}
    evidence: list[Evidence] = []
    entity_sources: list[EntitySource] = []
    decisions: list[MergeDecision] = []

    for (normalized_name, primary_type), observed in sorted(grouped_entities.items()):
        display_name = _best_display_name([entity.name for entity in observed])
        description = _best_description([entity.description for entity in observed])
        types = sorted({entity.type for entity in observed})
        source_chunk_ids = sorted({entity.source_chunk_id for entity in observed})
        node = GraphNode(
            id=stable_id("node", graph_id, normalized_name, primary_type),
            graph_id=graph_id,
            normalized_name=normalized_name,
            name=display_name,
            primary_type=primary_type,
            types=types,
            description=description,
            source_chunk_ids=source_chunk_ids,
        )
        nodes.append(node)
        node_by_name[normalized_name] = node
        decisions.append(
            MergeDecision(
                kind="entity",
                action="merged" if len(observed) > 1 else "created",
                reason=(
                    "entity_observations_merged" if len(observed) > 1 else "canonical_node_created"
                ),
                graph_id=graph_id,
                node_id=node.id,
                normalized_name=normalized_name,
                primary_type=primary_type,
                observation_count=len(observed),
                source_chunk_ids=source_chunk_ids,
                observed_names=sorted({entity.name for entity in observed}),
            )
        )
        for chunk_id in source_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk:
                observation = _best_observation_for_chunk(observed, chunk_id)
                evidence.append(
                    _evidence_for_observation(
                        graph_id,
                        node.id,
                        "node",
                        chunk,
                        display_name,
                        observation.quote if observation else None,
                        observation.start_offset if observation else None,
                        observation.end_offset if observation else None,
                    )
                )
        counts_by_file = Counter(
            chunks_by_id[cid].file_id for cid in source_chunk_ids if cid in chunks_by_id
        )
        for file_id, count in sorted(counts_by_file.items()):
            entity_sources.append(
                EntitySource(
                    id=stable_id("entity_source", graph_id, node.id, file_id),
                    graph_id=graph_id,
                    node_id=node.id,
                    file_id=file_id,
                    per_file_description=description,
                    mention_count=count,
                )
            )

    return EntityAssembly(
        nodes=nodes,
        node_by_name=node_by_name,
        evidence=evidence,
        entity_sources=entity_sources,
        decisions=decisions,
    )


def _assemble_relations(
    graph_id: str,
    chunks_by_id: dict[str, Chunk],
    node_by_name: dict[str, GraphNode],
    relations: list[ExtractedRelation],
    relation_weight_max: float,
) -> RelationAssembly:
    edges_by_key: dict[tuple[str, str, str, str], GraphEdge] = {}
    evidence: list[Evidence] = []
    decisions: list[MergeDecision] = []
    for relation in relations:
        source = node_by_name.get(normalize_entity_name(relation.source_name))
        target = node_by_name.get(normalize_entity_name(relation.target_name))
        chunk = chunks_by_id.get(relation.source_chunk_id)
        if not source:
            decisions.append(_dropped_relation_decision(graph_id, relation, "source_node_missing"))
            continue
        if not target:
            decisions.append(_dropped_relation_decision(graph_id, relation, "target_node_missing"))
            continue
        if not chunk:
            decisions.append(_dropped_relation_decision(graph_id, relation, "source_chunk_missing"))
            continue
        relation_type = normalize_relation_type(relation.relation_type)
        key = (source.id, target.id, relation_type, chunk.file_id)
        weight = min(max(relation.weight, 0.0), relation_weight_max)
        prior = edges_by_key.get(key)
        if prior is None:
            edge_id = stable_id(
                "edge",
                graph_id,
                source.id,
                target.id,
                relation_type,
                chunk.file_id,
            )
            edge = GraphEdge(
                id=edge_id,
                graph_id=graph_id,
                source_node_id=source.id,
                target_node_id=target.id,
                relation_type=relation_type,
                description=relation.description[:1000],
                weight=weight,
                source_file_id=chunk.file_id,
                source_chunk_ids=[chunk.id],
            )
            edges_by_key[key] = edge
            decisions.append(
                MergeDecision(
                    kind="relation",
                    action="created",
                    reason=(
                        "canonical_edge_created_weight_clamped"
                        if weight != relation.weight
                        else "canonical_edge_created"
                    ),
                    graph_id=graph_id,
                    edge_id=edge.id,
                    source_name=relation.source_name,
                    target_name=relation.target_name,
                    relation_type=relation_type,
                    source_chunk_id=chunk.id,
                    source_file_id=chunk.file_id,
                    original_weight=relation.weight,
                    effective_weight=weight,
                    new_weight=edge.weight,
                )
            )
            evidence.append(
                _evidence_for_observation(
                    graph_id,
                    edge.id,
                    "edge",
                    chunk,
                    relation.description,
                    relation.quote,
                    relation.start_offset,
                    relation.end_offset,
                )
            )
        else:
            prior_weight = prior.weight
            new_chunk = chunk.id not in prior.source_chunk_ids
            if new_chunk:
                prior.source_chunk_ids.append(chunk.id)
            prior.weight = min(prior.weight + weight, relation_weight_max)
            if len(relation.description) > len(prior.description):
                prior.description = relation.description[:1000]
            decisions.append(
                MergeDecision(
                    kind="relation",
                    action="aggregated",
                    reason=(
                        "relation_evidence_aggregated_weight_clamped"
                        if prior_weight + weight > relation_weight_max
                        else "relation_evidence_aggregated"
                    ),
                    graph_id=graph_id,
                    edge_id=prior.id,
                    source_name=relation.source_name,
                    target_name=relation.target_name,
                    relation_type=relation_type,
                    source_chunk_id=chunk.id,
                    source_file_id=chunk.file_id,
                    original_weight=relation.weight,
                    effective_weight=weight,
                    prior_weight=prior_weight,
                    new_weight=prior.weight,
                )
            )
            if new_chunk:
                evidence.append(
                    _evidence_for_observation(
                        graph_id,
                        prior.id,
                        "edge",
                        chunk,
                        relation.description,
                        relation.quote,
                        relation.start_offset,
                        relation.end_offset,
                    )
                )

    edges = sorted(edges_by_key.values(), key=lambda edge: edge.id)
    return RelationAssembly(
        edges=edges,
        evidence=evidence,
        decisions=decisions,
    )


def _best_display_name(names: list[str]) -> str:
    return sorted(names, key=lambda name: (-_display_score(name), len(name), name))[0]


def _display_score(name: str) -> int:
    score = 0
    if any(not char.isalnum() and not char.isspace() for char in name):
        score += 3
    alpha = "".join(char for char in name if char.isalpha())
    if alpha and not alpha.islower() and not alpha.isupper():
        score += 2
    elif alpha and alpha.isupper():
        score += 1
    return score


def _best_description(descriptions: list[str]) -> str:
    return max(descriptions, key=len, default="")


def _best_observation_for_chunk(
    observations: list[ExtractedEntity],
    chunk_id: str,
) -> ExtractedEntity | None:
    matching = [
        observation for observation in observations if observation.source_chunk_id == chunk_id
    ]
    if not matching:
        return None
    return sorted(
        matching,
        key=lambda observation: (
            observation.quote is None,
            observation.start_offset is None,
            -len(observation.description),
        ),
    )[0]


def _evidence_for_observation(
    graph_id: str,
    subject_id: str,
    subject_kind: str,
    chunk: Chunk,
    quote_seed: str,
    quote: str | None,
    start_offset: int | None,
    end_offset: int | None,
) -> Evidence:
    evidence_quote, evidence_start, evidence_end = _evidence_quote_and_offsets(
        chunk,
        quote,
        start_offset,
        end_offset,
        quote_seed,
    )
    return Evidence(
        id=stable_id("evidence", graph_id, subject_id, chunk.id),
        graph_id=graph_id,
        subject_id=subject_id,
        subject_kind=subject_kind,
        file_id=chunk.file_id,
        chunk_id=chunk.id,
        page_number=chunk.page_number,
        start_offset=evidence_start,
        end_offset=evidence_end,
        quote=evidence_quote,
    )


def _evidence_quote_and_offsets(
    chunk: Chunk,
    quote: str | None,
    start_offset: int | None,
    end_offset: int | None,
    quote_seed: str,
) -> tuple[str, int, int]:
    if start_offset is not None and end_offset is not None:
        local_quote = quote or chunk.content[start_offset:end_offset]
        return (
            local_quote.strip(),
            chunk.start_offset + start_offset,
            chunk.start_offset + end_offset,
        )
    if quote:
        local_start = chunk.content.lower().find(quote.strip().lower())
        if local_start >= 0:
            local_end = local_start + len(quote.strip())
            return quote.strip(), chunk.start_offset + local_start, chunk.start_offset + local_end
        return quote.strip(), chunk.start_offset, chunk.end_offset
    inferred = _quote(chunk.content, quote_seed)
    local_start = chunk.content.find(inferred)
    if local_start >= 0:
        return (
            inferred,
            chunk.start_offset + local_start,
            chunk.start_offset + local_start + len(inferred),
        )
    return inferred, chunk.start_offset, chunk.end_offset


def _quote(content: str, quote_seed: str) -> str:
    seed_words = [word.lower() for word in quote_seed.split() if len(word) > _QUOTE_SEED_MIN_LEN]
    lowered = content.lower()
    for word in seed_words:
        index = lowered.find(word)
        if index >= 0:
            start = max(0, index - 120)
            end = min(len(content), index + 240)
            return content[start:end].strip()
    return content[:360].strip()


def _assign_degrees(nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
    degree: Counter[str] = Counter()
    for edge in edges:
        degree[edge.source_node_id] += 1
        degree[edge.target_node_id] += 1
    for node in nodes:
        node.degree = degree[node.id]
        node.rank = float(node.degree)


def _dropped_relation_decision(
    graph_id: str,
    relation: ExtractedRelation,
    reason: str,
) -> MergeDecision:
    return MergeDecision(
        kind="relation",
        action="dropped",
        reason=reason,
        graph_id=graph_id,
        source_name=relation.source_name,
        target_name=relation.target_name,
        relation_type=normalize_relation_type(relation.relation_type),
        source_chunk_id=relation.source_chunk_id,
        original_weight=relation.weight,
    )


def _set_if_present(event: dict[str, object], key: str, value: object) -> None:
    if value is not None:
        event[key] = value
