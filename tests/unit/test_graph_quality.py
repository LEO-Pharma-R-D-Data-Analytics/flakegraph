from __future__ import annotations

from kg_processor.application.graph_quality import evaluate_graph_quality_rows


def test_graph_quality_passes_for_grounded_graph() -> None:
    result = evaluate_graph_quality_rows(
        nodes=[
            {"id": "node_1", "embedding": [0.1, 0.2]},
            {"id": "node_2", "embedding": [0.3, 0.4]},
        ],
        edges=[
            {
                "id": "edge_1",
                "source_node_id": "node_1",
                "target_node_id": "node_2",
                "embedding": [0.5, 0.6],
            }
        ],
        evidence=[
            {"id": "ev_1", "subject_id": "node_1"},
            {"id": "ev_2", "subject_id": "node_2"},
            {"id": "ev_3", "subject_id": "edge_1"},
        ],
        expected_embedding_dimension=2,
    )

    assert result.ok
    assert {check.name: check.ok for check in result.checks} == {
        "no_orphan_edges": True,
        "nodes_have_evidence": True,
        "edges_have_evidence": True,
        "node_embedding_dimensions": True,
        "edge_embedding_dimensions": True,
    }


def test_graph_quality_reports_orphans_missing_evidence_and_embedding_mismatches() -> None:
    result = evaluate_graph_quality_rows(
        nodes=[{"id": "node_1", "embedding": [0.1]}],
        edges=[
            {
                "id": "edge_1",
                "source_node_id": "node_1",
                "target_node_id": "missing_node",
                "embedding": None,
            }
        ],
        evidence=[],
        expected_embedding_dimension=2,
    )

    checks = {check.name: check for check in result.checks}
    assert not result.ok
    assert checks["no_orphan_edges"].details == ["edge_1"]
    assert checks["nodes_have_evidence"].details == ["node_1"]
    assert checks["edges_have_evidence"].details == ["edge_1"]
    assert checks["node_embedding_dimensions"].details == ["node_1:1"]
    assert checks["edge_embedding_dimensions"].details == ["edge_1:missing"]
