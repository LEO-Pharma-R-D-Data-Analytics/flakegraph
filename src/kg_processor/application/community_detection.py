"""Deterministic community detection over canonical weighted graph rows."""

from __future__ import annotations

import networkx as nx

from kg_processor.application.community_reports import (
    CommunityReportResult,
    generate_community_reports,
)
from kg_processor.domain.graph import GraphEdge, GraphNode
from kg_processor.ports.llm import LlmProvider


def detect_communities(
    graph_id: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    llm: LlmProvider,
    min_community_size: int,
) -> CommunityReportResult:
    """Detect weighted communities and enrich them through the LLM port."""

    if not nodes:
        return CommunityReportResult(communities=[], findings=[], trace_events=[])
    graph: nx.Graph[str] = nx.Graph()
    for node in nodes:
        graph.add_node(node.id)
    for edge in edges:
        graph.add_edge(edge.source_node_id, edge.target_node_id, weight=edge.weight)

    raw_communities = _community_sets(graph, min_community_size)
    return generate_community_reports(graph_id, raw_communities, nodes, edges, llm)


def _community_sets(graph: nx.Graph[str], min_community_size: int) -> list[set[str]]:
    if graph.number_of_edges() == 0:
        return [{node} for node in graph.nodes]
    communities = list(
        nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    )
    result: list[set[str]] = []
    for community in communities:
        if len(community) >= min_community_size:
            result.append(set(community))
        else:
            result.extend({node} for node in community)
    return sorted(result, key=lambda members: (min(members), len(members)))
