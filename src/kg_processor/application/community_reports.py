"""Generate grounded community summaries from canonical nodes and edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kg_processor.domain.graph import Community, CommunityFinding, GraphEdge, GraphNode
from kg_processor.domain.ids import stable_id
from kg_processor.ports.llm import CommunitySummaryRequest, LlmProvider

DEFAULT_MAX_RELATIONS_PER_COMMUNITY = 25


@dataclass(frozen=True)
class CommunityReportResult:
    """Communities, findings, and trace events generated from graph clusters."""

    communities: list[Community]
    findings: list[CommunityFinding]
    trace_events: list[dict[str, Any]]


def generate_community_reports(
    graph_id: str,
    community_member_sets: list[set[str]],
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    llm: LlmProvider,
    max_relations_per_community: int = DEFAULT_MAX_RELATIONS_PER_COMMUNITY,
) -> CommunityReportResult:
    """Generate stable community rows and LLM-authored findings."""

    # Community ids are based on sorted member ids rather than LLM output. That
    # makes report regeneration safe even when summaries or ratings change.
    node_by_id = {node.id: node for node in nodes}
    communities: list[Community] = []
    findings: list[CommunityFinding] = []
    trace_events: list[dict[str, Any]] = []
    for index, member_ids in enumerate(community_member_sets):
        members = _ordered_members(member_ids, node_by_id)
        if not members:
            continue
        internal_edges = _internal_edges(member_ids, edges, max_relations_per_community)
        request = CommunitySummaryRequest(
            title_seed=_title_seed(members, index),
            members=[_member_line(node) for node in members],
            relations=[_relation_line(edge, node_by_id) for edge in internal_edges],
        )
        summary = llm.summarize_community(request)
        stable_key = stable_id("community_key", 0, ",".join(sorted(member_ids)), length=40)
        community = Community(
            id=stable_id("community", graph_id, stable_key),
            graph_id=graph_id,
            stable_key=stable_key,
            level=0,
            title=summary.title,
            summary=summary.summary,
            rating=summary.rating,
            rating_explanation=summary.rating_explanation,
            member_node_ids=sorted(member_ids),
            suggested_questions=summary.suggested_questions,
        )
        communities.append(community)
        for finding_index, (finding_summary, explanation) in enumerate(summary.findings):
            findings.append(
                CommunityFinding(
                    id=stable_id("finding", community.id, finding_index, finding_summary),
                    graph_id=graph_id,
                    community_id=community.id,
                    summary=finding_summary,
                    explanation=explanation,
                )
            )
        trace_events.append(
            {
                "stage": "community_report",
                "community_id": community.id,
                "stable_key": stable_key,
                "level": community.level,
                "members": len(member_ids),
                "relations_used": len(internal_edges),
                "findings": len(summary.findings),
                "suggested_questions": len(summary.suggested_questions),
                "provider_metadata": summary.provider_metadata,
            }
        )
    return CommunityReportResult(
        communities=communities,
        findings=findings,
        trace_events=trace_events,
    )


def _ordered_members(member_ids: set[str], node_by_id: dict[str, GraphNode]) -> list[GraphNode]:
    members = [node_by_id[node_id] for node_id in member_ids if node_id in node_by_id]
    return sorted(members, key=lambda node: (-node.degree, node.name.lower(), node.id))


def _title_seed(members: list[GraphNode], index: int) -> str:
    if members:
        return members[0].name
    return f"Community {index + 1}"


def _member_line(node: GraphNode) -> str:
    return f"{node.name} [{node.primary_type}] degree={node.degree}: {node.description[:500]}"


def _internal_edges(
    member_ids: set[str],
    edges: list[GraphEdge],
    max_relations_per_community: int,
) -> list[GraphEdge]:
    internal = [
        edge
        for edge in edges
        if edge.source_node_id in member_ids and edge.target_node_id in member_ids
    ]
    return sorted(
        internal,
        key=lambda edge: (-edge.weight, edge.relation_type, edge.source_node_id, edge.id),
    )[:max_relations_per_community]


def _relation_line(edge: GraphEdge, node_by_id: dict[str, GraphNode]) -> str:
    source = node_by_id.get(edge.source_node_id)
    target = node_by_id.get(edge.target_node_id)
    source_name = source.name if source else edge.source_node_id
    target_name = target.name if target else edge.target_node_id
    return (
        f"{source_name} --{edge.relation_type}--> {target_name}"
        f" weight={edge.weight}: {edge.description[:500]}"
    )
