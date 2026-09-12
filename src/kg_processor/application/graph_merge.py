"""Canonical graph assembly from grounded extraction observations.

Merge decisions are explicit trace records. Canonical edge ids are independent
of file scope, while edge observations preserve each file/chunk assertion so
reruns and Snowflake MERGE operations remain auditable and idempotent.
"""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from kg_processor.domain.graph import (
    Chunk,
    EdgeObservation,
    EntitySource,
    Evidence,
    ExtractedEntity,
    ExtractedRelation,
    GraphEdge,
    GraphNode,
)
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import normalize_ontology_label

_QUOTE_SEED_MIN_LEN = 3
# Persisted relation labels stay within a bounded identifier length so canonical
# predicates remain usable as Snowflake identifiers and stable join keys.
MAX_RELATION_LABEL_LENGTH = 80


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
    """Bundle canonical graph rows, provenance, and explicit merge decisions.

    Returning diagnostics with persisted artifacts allows run reports to explain
    which observations were created, merged, repaired, or dropped.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    edge_observations: list[EdgeObservation]
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
    node_by_name_and_type: dict[tuple[str, str], GraphNode]
    evidence: list[Evidence]
    entity_sources: list[EntitySource]
    decisions: list[MergeDecision]


@dataclass(frozen=True)
class RelationAssembly:
    """Hold canonical relation artifacts before final graph-level reconciliation.

    Edge assertions remain separate from canonical edges so repeated evidence can
    accumulate support without duplicating graph topology.
    """

    edges: list[GraphEdge]
    edge_observations: list[EdgeObservation]
    evidence: list[Evidence]
    decisions: list[MergeDecision]


def prune_isolated_entities(assembly: GraphAssemblyResult) -> GraphAssemblyResult:
    """Remove canonical entities that participate in no accepted graph relation.

    Window extraction intentionally favors recall and can surface headings,
    citation metadata, or incidental names. Requiring at least one grounded edge
    converts that broad observation inventory into an explorable knowledge graph,
    avoids expensive enrichment for disconnected records, and removes only rows
    whose provenance remains available in extraction traces.
    """

    connected_ids = {
        node_id for edge in assembly.edges for node_id in (edge.source_node_id, edge.target_node_id)
    }
    removed = [node for node in assembly.nodes if node.id not in connected_ids]
    if not removed:
        return assembly
    decisions = [
        *assembly.decisions,
        *[
            MergeDecision(
                kind="entity",
                action="dropped",
                reason="isolated_entity_without_relation",
                graph_id=node.graph_id,
                node_id=node.id,
                normalized_name=node.normalized_name,
                primary_type=node.primary_type,
            )
            for node in removed
        ],
    ]
    return GraphAssemblyResult(
        nodes=[node for node in assembly.nodes if node.id in connected_ids],
        edges=assembly.edges,
        edge_observations=assembly.edge_observations,
        evidence=[
            item
            for item in assembly.evidence
            if item.subject_kind != "node" or item.subject_id in connected_ids
        ],
        entity_sources=[item for item in assembly.entity_sources if item.node_id in connected_ids],
        decisions=decisions,
    )


def normalize_entity_name(name: str) -> str:
    """Normalize entity names into stable keys used for merging.

    This is the single canonical-name identity rule. Node IDs are hashed from its
    result, so every engine that assembles graph rows must call this function
    rather than reimplement it: NFC composition and full case folding have no
    equivalent expression in SQL dialects, and any approximation mints different
    IDs for the same entity and breaks idempotent merges into shared tables.
    """

    # Keep Unicode letters/digits so names such as "Muller", "Müller", and
    # "東京" do not collapse into the same ASCII-only key or disappear entirely.
    normalized = unicodedata.normalize("NFC", name)
    stripped = "".join(char for char in normalized.casefold().strip() if char.isalnum())
    return stripped or normalized.strip().casefold()


def normalize_relation_type(relation_type: str) -> str:
    """Normalize relation labels into bounded Snowflake-friendly identifiers.

    Canonical edge IDs are hashed from this value, so it is the single relation
    label rule for every engine. The bound applies wherever a persisted
    ``relation_type`` is produced or re-derived, keeping label identity stable
    across engines even for pathologically long extracted predicates.
    """

    return normalize_ontology_label(relation_type)[:MAX_RELATION_LABEL_LENGTH]


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
    """Merge grounded observations into canonical graph rows and audit decisions.

    Entity grouping runs first to establish endpoint identity. Relation assembly
    then consolidates repeated triples while retaining per-source assertions,
    exact evidence, bounded weights, and reasons for every dropped observation.
    """

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
        entity_assembly.node_by_name_and_type,
        relations,
        relation_weight_max,
    )
    nodes = sorted(entity_assembly.nodes, key=lambda node: node.id)
    edges = relation_assembly.edges
    _assign_degrees(nodes, edges)
    return GraphAssemblyResult(
        nodes=nodes,
        edges=edges,
        edge_observations=relation_assembly.edge_observations,
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
    node_by_name_and_type: dict[tuple[str, str], GraphNode] = {}
    evidence: list[Evidence] = []
    entity_sources: list[EntitySource] = []
    decisions: list[MergeDecision] = []

    for (normalized_name, grouped_type), observed in sorted(grouped_entities.items()):
        primary_type = _primary_type(observed)
        display_name = _best_display_name([entity.name for entity in observed])
        description = _best_description([entity.description for entity in observed])
        types = sorted({entity.type for entity in observed})
        aliases = _node_aliases(observed, display_name)
        source_chunk_ids = sorted({entity.source_chunk_id for entity in observed})
        observations_by_chunk: dict[str, ExtractedEntity] = {}
        for observation in observed:
            current = observations_by_chunk.get(observation.source_chunk_id)
            if current is None or _observation_quality_key(observation) < _observation_quality_key(
                current
            ):
                observations_by_chunk[observation.source_chunk_id] = observation
        node = GraphNode(
            id=stable_id("node", graph_id, normalized_name, primary_type),
            graph_id=graph_id,
            normalized_name=normalized_name,
            name=display_name,
            primary_type=primary_type,
            types=types,
            aliases=aliases,
            description=description,
            source_chunk_ids=source_chunk_ids,
        )
        nodes.append(node)
        node_by_name_and_type[(normalized_name, grouped_type)] = node
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
                selected_observation = observations_by_chunk.get(chunk_id)
                evidence.append(
                    _evidence_for_observation(
                        graph_id,
                        node.id,
                        "node",
                        chunk,
                        display_name,
                        selected_observation.quote if selected_observation else None,
                        selected_observation.start_offset if selected_observation else None,
                        selected_observation.end_offset if selected_observation else None,
                    )
                )
        counts_by_file = Counter(
            chunks_by_id[cid].file_id for cid in source_chunk_ids if cid in chunks_by_id
        )
        for file_id, count in sorted(counts_by_file.items()):
            file_descriptions = [
                observation.description
                for observation in observed
                if (
                    (chunk := chunks_by_id.get(observation.source_chunk_id)) is not None
                    and chunk.file_id == file_id
                )
            ]
            entity_sources.append(
                EntitySource(
                    id=stable_id("entity_source", graph_id, node.id, file_id),
                    graph_id=graph_id,
                    node_id=node.id,
                    file_id=file_id,
                    per_file_description=_best_description(file_descriptions),
                    mention_count=count,
                )
            )

    return EntityAssembly(
        nodes=nodes,
        node_by_name_and_type=_with_unique_alias_endpoints(node_by_name_and_type, nodes),
        evidence=evidence,
        entity_sources=entity_sources,
        decisions=decisions,
    )


def _with_unique_alias_endpoints(
    canonical_nodes: dict[tuple[str, str], GraphNode],
    nodes: list[GraphNode],
) -> dict[tuple[str, str], GraphNode]:
    """Extend the endpoint index with aliases that identify one canonical node.

    Relations are extracted from local text and may therefore use an acronym or
    another accepted surface after entity resolution selected a longer canonical
    name. Aliases are lookup keys only; nodes, IDs, and persisted display names
    remain canonical. A colliding alias is deliberately omitted so it cannot join
    a relation to an arbitrary same-typed entity.
    """

    endpoint_nodes = dict(canonical_nodes)
    alias_candidates: dict[tuple[str, str], dict[str, GraphNode]] = defaultdict(dict)
    for node in nodes:
        for alias in node.aliases:
            normalized_alias = normalize_entity_name(alias)
            for node_type in node.types:
                alias_candidates[(normalized_alias, node_type)][node.id] = node
    for key, candidates in alias_candidates.items():
        if key not in endpoint_nodes and len(candidates) == 1:
            endpoint_nodes[key] = next(iter(candidates.values()))
    untyped_candidates: dict[str, dict[str, GraphNode]] = defaultdict(dict)
    for (normalized_name, _node_type), node in endpoint_nodes.items():
        untyped_candidates[normalized_name][node.id] = node
    for normalized_name, candidates in untyped_candidates.items():
        if len(candidates) == 1:
            endpoint_nodes[(normalized_name, "")] = next(iter(candidates.values()))
    return endpoint_nodes


def _node_aliases(observed: list[ExtractedEntity], display_name: str) -> list[str]:
    """Collect deterministic accepted surfaces other than the display name.

    Exact normalized duplicates are removed so casing-only variants do not
    inflate persisted metadata. Every value already passed entity grounding and
    alias validation before graph assembly.
    """

    display_key = normalize_entity_name(display_name)
    by_key: dict[str, str] = {}
    for entity in observed:
        for surface in [entity.name, *entity.aliases]:
            stripped = surface.strip()
            key = normalize_entity_name(stripped)
            if stripped and key and key != display_key:
                by_key.setdefault(key, stripped)
    return [by_key[key] for key in sorted(by_key)]


def _primary_type(observed: list[ExtractedEntity]) -> str:
    counts = Counter(entity.type for entity in observed)
    first_seen = {entity.type: index for index, entity in enumerate(observed)}
    return sorted(
        counts,
        key=lambda entity_type: (-counts[entity_type], first_seen[entity_type]),
    )[0]


def _assemble_relations(  # noqa: PLR0915 - each branch records a distinct audit decision.
    graph_id: str,
    chunks_by_id: dict[str, Chunk],
    node_by_name_and_type: dict[tuple[str, str], GraphNode],
    relations: list[ExtractedRelation],
    relation_weight_max: float,
) -> RelationAssembly:
    """Assemble grounded relation observations into canonical directed edges.

    Missing endpoints or chunks become explicit drop decisions. Repeated triples
    share one canonical edge while each source assertion and evidence record is
    preserved independently for provenance and reindexing.
    """

    edges_by_key: dict[tuple[str, str, str], GraphEdge] = {}
    edge_chunk_ids: dict[str, set[str]] = {}
    edge_observations: list[EdgeObservation] = []
    evidence: list[Evidence] = []
    decisions: list[MergeDecision] = []
    for relation in relations:
        source = _relation_endpoint_node(
            node_by_name_and_type,
            relation.source_name,
            relation.source_type,
        )
        target = _relation_endpoint_node(
            node_by_name_and_type,
            relation.target_name,
            relation.target_type,
        )
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
        if source.id == target.id:
            # Distinct extracted spellings can resolve to the same canonical node
            # after punctuation and alias normalization. Such a relation was not a
            # source-level self-loop and therefore could not be rejected earlier.
            decisions.append(
                _dropped_relation_decision(graph_id, relation, "self_loop_after_entity_merge")
            )
            continue
        relation_type = normalize_relation_type(relation.relation_type)
        key = (source.id, target.id, relation_type)
        weight = min(max(relation.weight, 0.0), relation_weight_max)
        prior = edges_by_key.get(key)
        if prior is None:
            edge_id = stable_id(
                "edge",
                graph_id,
                source.id,
                target.id,
                relation_type,
            )
            edge = GraphEdge(
                id=edge_id,
                graph_id=graph_id,
                source_node_id=source.id,
                target_node_id=target.id,
                relation_type=relation_type,
                description=relation.description[:1000],
                weight=weight,
                confidence=relation.confidence,
                source_file_id=chunk.file_id,
                source_file_ids=[chunk.file_id],
                source_chunk_ids=[chunk.id],
                evidence_count=1,
            )
            edges_by_key[key] = edge
            edge_chunk_ids[edge.id] = {chunk.id}
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
            edge_evidence = _evidence_for_observation(
                graph_id,
                edge.id,
                "edge",
                chunk,
                relation.description,
                relation.quote,
                relation.start_offset,
                relation.end_offset,
            )
            evidence.append(edge_evidence)
            edge_observations.append(
                _edge_observation(graph_id, edge, chunk, relation, weight, edge_evidence.id)
            )
        else:
            prior_weight = prior.weight
            new_chunk = chunk.id not in edge_chunk_ids[prior.id]
            if not new_chunk:
                decisions.append(
                    _dropped_relation_decision(
                        graph_id,
                        relation,
                        "duplicate_edge_observation",
                    )
                )
                continue
            prior.source_chunk_ids.append(chunk.id)
            edge_chunk_ids[prior.id].add(chunk.id)
            prior.source_file_ids.append(chunk.file_id)
            prior.source_file_ids = sorted(set(prior.source_file_ids))
            prior.source_file_id = prior.source_file_ids[0]
            prior.evidence_count += 1
            prior.weight = min(prior.weight + weight, relation_weight_max)
            prior.confidence = max(prior.confidence, relation.confidence)
            if (len(relation.description), relation.description) > (
                len(prior.description),
                prior.description,
            ):
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
            edge_evidence = _evidence_for_observation(
                graph_id,
                prior.id,
                "edge",
                chunk,
                relation.description,
                relation.quote,
                relation.start_offset,
                relation.end_offset,
            )
            evidence.append(edge_evidence)
            edge_observations.append(
                _edge_observation(
                    graph_id,
                    prior,
                    chunk,
                    relation,
                    weight,
                    edge_evidence.id,
                )
            )

    edges = sorted(edges_by_key.values(), key=lambda edge: edge.id)
    for edge in edges:
        edge.source_chunk_ids.sort()
        edge.source_file_ids = sorted(set(edge.source_file_ids))
    return RelationAssembly(
        edges=edges,
        edge_observations=sorted(edge_observations, key=lambda item: item.id),
        evidence=sorted(evidence, key=lambda item: item.id),
        decisions=decisions,
    )


def _relation_endpoint_node(
    nodes: dict[tuple[str, str], GraphNode],
    name: str,
    entity_type: str | None,
) -> GraphNode | None:
    """Resolve a relation endpoint without conflating same-named typed entities.

    Two-pass extraction supplies the endpoint type and therefore resolves exactly.
    Older/provider-neutral extraction records may omit it; those are accepted only
    when the normalized name identifies a single node unambiguously.
    """

    normalized_name = normalize_entity_name(name)
    if entity_type is not None:
        return nodes.get((normalized_name, entity_type))
    return nodes.get((normalized_name, ""))


def _edge_observation(
    graph_id: str,
    edge: GraphEdge,
    chunk: Chunk,
    relation: ExtractedRelation,
    weight: float,
    evidence_id: str,
) -> EdgeObservation:
    """Preserve one supporting assertion independently from its canonical edge.

    This record retains file, chunk, description, weight, confidence, and evidence
    identity even when many observations consolidate into one graph relation.
    """

    return EdgeObservation(
        id=stable_id("edge_observation", graph_id, edge.id, chunk.id),
        graph_id=graph_id,
        edge_id=edge.id,
        file_id=chunk.file_id,
        chunk_id=chunk.id,
        description=relation.description[:1000],
        weight=weight,
        confidence=relation.confidence,
        evidence_id=evidence_id,
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
    return max(descriptions, key=lambda description: (len(description), description), default="")


def _observation_quality_key(observation: ExtractedEntity) -> tuple[bool, bool, int]:
    """Rank one observation without rescanning its full canonical group."""

    return (
        observation.quote is None,
        observation.start_offset is None,
        -len(observation.description),
    )


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
        local_start, local_end = _casefold_span(chunk.content, quote.strip())
        if local_start >= 0:
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
    seed_words = [word for word in quote_seed.split() if len(word) > _QUOTE_SEED_MIN_LEN]
    for word in seed_words:
        index = _casefold_find(content, word)
        if index >= 0:
            start = max(0, index - 120)
            end = min(len(content), index + 240)
            return content[start:end].strip()
    return content[:360].strip()


def _casefold_find(content: str, needle: str) -> int:
    return _casefold_span(content, needle)[0]


def _casefold_span(content: str, needle: str) -> tuple[int, int]:
    """Map a casefolded match back to its exact original-codepoint span."""

    if not needle:
        return -1, -1
    folded_content, starts, ends = _casefold_with_offsets(content)
    folded_needle = unicodedata.normalize("NFC", needle).casefold()
    folded_index = folded_content.find(folded_needle)
    if folded_index < 0 or folded_index >= len(starts):
        return -1, -1
    folded_end = folded_index + len(folded_needle)
    if folded_end <= folded_index or folded_end > len(starts):
        return -1, -1
    return starts[folded_index], ends[folded_end - 1]


def _casefold_with_offsets(value: str) -> tuple[str, list[int], list[int]]:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(value):
        cluster_end = index + 1
        while cluster_end < len(value) and unicodedata.combining(value[cluster_end]):
            cluster_end += 1
        folded = unicodedata.normalize("NFC", value[index:cluster_end]).casefold()
        chars.append(folded)
        starts.extend([index] * len(folded))
        ends.extend([cluster_end] * len(folded))
        index = cluster_end
    return "".join(chars), starts, ends


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
