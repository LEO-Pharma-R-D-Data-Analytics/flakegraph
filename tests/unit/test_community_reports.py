from __future__ import annotations

from kg_processor.application.community_reports import generate_community_reports
from kg_processor.domain.graph import Chunk, ExtractionResult, GraphEdge, GraphNode
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    GraphRepairRequest,
    LlmOptions,
)


def test_generate_community_reports_uses_grounded_member_and_relation_context() -> None:
    nodes = [
        _node("node_alice", "Alice Smith", "PERSON", degree=2),
        _node("node_acme", "Acme Corp", "ORGANIZATION", degree=1),
        _node("node_copenhagen", "Copenhagen", "LOCATION", degree=1),
    ]
    edges = [
        _edge(
            "edge_low",
            "node_acme",
            "node_copenhagen",
            "located_in",
            "Acme Corp is located in Copenhagen.",
            weight=2.0,
        ),
        _edge(
            "edge_high",
            "node_alice",
            "node_acme",
            "works_at",
            "Alice Smith works at Acme Corp.",
            weight=9.0,
        ),
    ]
    provider = _RecordingLlm()

    result = generate_community_reports(
        "graph",
        [{"node_alice", "node_acme", "node_copenhagen"}],
        nodes,
        edges,
        provider,
        max_relations_per_community=1,
    )
    communities = result.communities
    findings = result.findings

    assert len(communities) == 1
    assert len(findings) == 1
    assert communities[0].title == "Alice Smith network"
    assert communities[0].rating_explanation == "High-weight employment relation."
    assert communities[0].suggested_questions == ["Where does Alice Smith work?"]
    assert communities[0].member_node_ids == ["node_acme", "node_alice", "node_copenhagen"]
    assert provider.requests[0].title_seed == "Alice Smith"
    assert provider.requests[0].members[0].startswith("Alice Smith [PERSON] degree=2")
    assert provider.requests[0].relations == [
        "Alice Smith --works_at--> Acme Corp weight=9.0: Alice Smith works at Acme Corp."
    ]
    assert result.trace_events == [
        {
            "stage": "community_report",
            "community_id": communities[0].id,
            "stable_key": communities[0].stable_key,
            "level": 0,
            "members": 3,
            "relations_used": 1,
            "findings": 1,
            "suggested_questions": 1,
            "provider_metadata": {
                "provider": "recording_llm",
                "prompt_name": "community_report",
            },
        }
    ]


class _RecordingLlm:
    def __init__(self) -> None:
        self.requests: list[CommunitySummaryRequest] = []

    def extract_graph(self, chunks: list[Chunk], options: LlmOptions) -> ExtractionResult:
        raise NotImplementedError

    def repair_graph_extraction(self, request: GraphRepairRequest) -> ExtractionResult:
        raise NotImplementedError

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        raise NotImplementedError

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        self.requests.append(request)
        return CommunitySummaryResult(
            title=f"{request.title_seed} network",
            summary="Grounded summary.",
            rating=8.0,
            rating_explanation="High-weight employment relation.",
            findings=[("Strong connection", request.relations[0])],
            suggested_questions=["Where does Alice Smith work?"],
            provider_metadata={
                "provider": "recording_llm",
                "prompt_name": "community_report",
            },
        )


def _node(node_id: str, name: str, primary_type: str, degree: int) -> GraphNode:
    return GraphNode(
        id=node_id,
        graph_id="graph",
        normalized_name=name.lower().replace(" ", ""),
        name=name,
        primary_type=primary_type,
        types=[primary_type],
        description=f"{name} appears in the source graph.",
        source_chunk_ids=["chunk_1"],
        degree=degree,
        rank=float(degree),
    )


def _edge(
    edge_id: str,
    source_node_id: str,
    target_node_id: str,
    relation_type: str,
    description: str,
    weight: float,
) -> GraphEdge:
    return GraphEdge(
        id=edge_id,
        graph_id="graph",
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        relation_type=relation_type,
        description=description,
        weight=weight,
        source_file_id="file_1",
        source_chunk_ids=["chunk_1"],
    )
