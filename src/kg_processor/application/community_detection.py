"""Deterministic community detection over canonical weighted graph rows."""

from __future__ import annotations

from itertools import combinations

import igraph as ig  # type: ignore[import-untyped]
import leidenalg  # type: ignore[import-untyped]
import networkx as nx

from kg_processor.application.community_reports import (
    CommunityProgressCallback,
    CommunityReportResult,
    CommunitySeed,
    generate_community_reports,
)
from kg_processor.domain.graph import Evidence, GraphEdge, GraphNode
from kg_processor.domain.ids import stable_id
from kg_processor.ports.llm import DEFAULT_LLM_TIMEOUT_SECONDS, LlmProvider

_MAX_HIERARCHY_LEVEL = 2
_MAX_CO_MENTION_NODES = 30
_CHILD_RESOLUTION_MULTIPLIER = 1.5


def detect_communities(
    graph_id: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    llm: LlmProvider,
    min_community_size: int,
    report_parallelism: int = 1,
    report_progress: CommunityProgressCallback | None = None,
    evidence: list[Evidence] | None = None,
    deterministic_seed: int = 17,
    max_community_size: int = 50,
    resolution: float = 1.0,
    co_mention_weight: float = 0.05,
    model: str = "default",
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
) -> CommunityReportResult:
    """Detect deterministic weighted communities and generate grounded reports.

    Canonical relations provide the primary graph, weak chunk co-mentions help
    stabilize sparse regions, and seeded Leiden partitions are converted into
    provider-neutral report requests.
    """

    if not nodes:
        return CommunityReportResult(communities=[], findings=[], trace_events=[])
    graph: nx.Graph[str] = nx.Graph()
    for node in nodes:
        graph.add_node(node.id)
    for edge in edges:
        if edge.source_node_id != edge.target_node_id:
            _add_weighted_edge(graph, edge.source_node_id, edge.target_node_id, edge.weight)
    _add_co_mention_projection(graph, nodes, co_mention_weight)

    raw_communities = _community_seeds(
        graph,
        min_community_size,
        max_community_size,
        resolution,
        deterministic_seed,
    )
    return generate_community_reports(
        graph_id,
        raw_communities,
        nodes,
        edges,
        llm,
        report_parallelism=report_parallelism,
        report_progress=report_progress,
        evidence=evidence,
        model=model,
        timeout_seconds=timeout_seconds,
        seed=deterministic_seed,
    )


def _community_seeds(
    graph: nx.Graph[str],
    min_community_size: int,
    max_community_size: int,
    resolution: float,
    deterministic_seed: int,
) -> list[CommunitySeed]:
    """Build top-level and bounded child community seeds from a weighted graph.

    Edgeless or non-positive graphs produce no communities. Oversized partitions
    are recursively subdivided so report prompts remain useful and bounded.
    """

    if graph.number_of_edges() == 0:
        return []
    total_weight = sum(float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True))
    if total_weight <= 0:
        return []
    top_level = [
        members
        for members in _leiden_partition(graph, resolution, deterministic_seed)
        if len(members) >= min_community_size
    ]
    seeds: list[CommunitySeed] = []
    for members in top_level:
        seeds.append(CommunitySeed(member_ids=members))
        _append_child_seeds(
            seeds,
            graph,
            members,
            level=1,
            parent_stable_key=_stable_key(members, 0),
            min_community_size=min_community_size,
            max_community_size=max_community_size,
            resolution=resolution * _CHILD_RESOLUTION_MULTIPLIER,
            deterministic_seed=deterministic_seed,
        )
    return seeds


def _append_child_seeds(
    seeds: list[CommunitySeed],
    graph: nx.Graph[str],
    members: set[str],
    *,
    level: int,
    parent_stable_key: str,
    min_community_size: int,
    max_community_size: int,
    resolution: float,
    deterministic_seed: int,
) -> None:
    """Recursively split oversized communities into reportable child groups.

    Recursion is bounded by hierarchy depth and stops when Leiden cannot produce
    more than one proper child, preventing duplicate or unhelpful report levels.
    """

    if len(members) <= max_community_size or level > _MAX_HIERARCHY_LEVEL:
        return
    induced = graph.subgraph(members).copy()
    children = [
        child
        for child in _leiden_partition(induced, resolution, deterministic_seed + level)
        if min_community_size <= len(child) < len(members)
    ]
    if len(children) <= 1:
        return
    for child in children:
        seeds.append(
            CommunitySeed(
                member_ids=child,
                level=level,
                parent_stable_key=parent_stable_key,
            )
        )
        _append_child_seeds(
            seeds,
            induced,
            child,
            level=level + 1,
            parent_stable_key=_stable_key(child, level),
            min_community_size=min_community_size,
            max_community_size=max_community_size,
            resolution=resolution * _CHILD_RESOLUTION_MULTIPLIER,
            deterministic_seed=deterministic_seed,
        )


def _leiden_partition(
    graph: nx.Graph[str],
    resolution: float,
    deterministic_seed: int,
) -> list[set[str]]:
    """Run seeded Leiden after imposing stable vertex and edge ordering.

    Sorting removes iteration-order variance before crossing into igraph, and
    the final communities are sorted again to make artifact IDs reproducible.
    """

    node_ids = sorted(graph.nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    ordered_edges = sorted(
        (
            min(source, target),
            max(source, target),
            float(data.get("weight", 1.0)),
        )
        for source, target, data in graph.edges(data=True)
        if source != target
    )
    leiden_graph = ig.Graph(
        n=len(node_ids),
        edges=[(node_index[source], node_index[target]) for source, target, _ in ordered_edges],
        directed=False,
    )
    partition = leidenalg.find_partition(
        leiden_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=[weight for _source, _target, weight in ordered_edges],
        resolution_parameter=resolution,
        seed=deterministic_seed,
    )
    communities = [{node_ids[index] for index in community} for community in partition]
    return sorted(communities, key=lambda members: (min(members), len(members)))


def _add_co_mention_projection(
    graph: nx.Graph[str],
    nodes: list[GraphNode],
    weight: float,
) -> None:
    """Project bounded chunk co-mentions as weak community-only graph edges.

    These links never become knowledge-graph facts. They help sparse but related
    entities cluster while a node cap prevents broad chunks creating dense cliques.
    """

    nodes_by_chunk: dict[str, list[str]] = {}
    for node in nodes:
        for chunk_id in node.source_chunk_ids:
            nodes_by_chunk.setdefault(chunk_id, []).append(node.id)
    for node_ids in nodes_by_chunk.values():
        # Very broad chunks should not create a dense artificial clique.
        bounded_ids = sorted(set(node_ids))[:_MAX_CO_MENTION_NODES]
        for source, target in combinations(bounded_ids, 2):
            _add_weighted_edge(graph, source, target, weight)


def _stable_key(members: set[str], level: int) -> str:
    """Derive a reproducible community hierarchy key from sorted member IDs and depth."""

    return stable_id("community_key", level, ",".join(sorted(members)), length=40)


def _add_weighted_edge(
    graph: nx.Graph[str],
    source_node_id: str,
    target_node_id: str,
    weight: float,
) -> None:
    """Add an undirected community edge, accumulating repeated graph facts."""

    if graph.has_edge(source_node_id, target_node_id):
        graph[source_node_id][target_node_id]["weight"] += weight
        return
    graph.add_edge(source_node_id, target_node_id, weight=weight)
