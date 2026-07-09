"""Graph quality gates that run before writer persistence by default."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kg_processor.domain.graph import Evidence, GraphEdge, GraphNode


class GraphQualityCheck(BaseModel):
    """Single quality gate result with bounded details for reports."""

    name: str
    ok: bool
    count: int = 0
    details: list[str] = Field(default_factory=list)


class GraphQualityResult(BaseModel):
    """Aggregate quality result used to decide whether persistence may continue."""

    ok: bool
    checks: list[GraphQualityCheck]


class GraphQualityError(RuntimeError):
    """Raised when configured graph quality gates fail."""

    def __init__(self, result: GraphQualityResult) -> None:
        self.result = result
        failed_checks = [check.name for check in result.checks if not check.ok]
        message = "Graph quality checks failed"
        if failed_checks:
            message = f"{message}: {', '.join(failed_checks)}"
        super().__init__(message)


def evaluate_graph_quality(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    evidence: list[Evidence],
    expected_embedding_dimension: int | None,
) -> GraphQualityResult:
    """Evaluate quality gates over domain graph models."""

    return evaluate_graph_quality_rows(
        [node.model_dump(mode="json") for node in nodes],
        [edge.model_dump(mode="json") for edge in edges],
        [item.model_dump(mode="json") for item in evidence],
        expected_embedding_dimension,
    )


def evaluate_graph_quality_rows(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    expected_embedding_dimension: int | None,
) -> GraphQualityResult:
    """Evaluate quality gates over serialized graph rows."""

    checks = [
        _orphan_edge_check(nodes, edges),
        _evidence_coverage_check("nodes_have_evidence", nodes, evidence),
        _evidence_coverage_check("edges_have_evidence", edges, evidence),
    ]
    if expected_embedding_dimension is not None:
        checks.extend(
            [
                _embedding_dimension_check(
                    "node_embedding_dimensions",
                    nodes,
                    expected_embedding_dimension,
                ),
                _embedding_dimension_check(
                    "edge_embedding_dimensions",
                    edges,
                    expected_embedding_dimension,
                ),
            ]
        )
    return GraphQualityResult(ok=all(check.ok for check in checks), checks=checks)


def _orphan_edge_check(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> GraphQualityCheck:
    node_ids = {str(node.get("id")) for node in nodes}
    orphan_edges = [
        str(edge.get("id"))
        for edge in edges
        if str(edge.get("source_node_id")) not in node_ids
        or str(edge.get("target_node_id")) not in node_ids
    ]
    return GraphQualityCheck(
        name="no_orphan_edges",
        ok=not orphan_edges,
        count=len(orphan_edges),
        details=orphan_edges[:20],
    )


def _evidence_coverage_check(
    name: str,
    subjects: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> GraphQualityCheck:
    evidence_subject_ids = {str(item.get("subject_id")) for item in evidence}
    missing = [
        str(subject.get("id"))
        for subject in subjects
        if str(subject.get("id")) not in evidence_subject_ids
    ]
    return GraphQualityCheck(
        name=name,
        ok=not missing,
        count=len(missing),
        details=missing[:20],
    )


def _embedding_dimension_check(
    name: str,
    rows: list[dict[str, Any]],
    expected_dimension: int,
) -> GraphQualityCheck:
    mismatches = []
    for row in rows:
        embedding = row.get("embedding")
        if embedding is None:
            mismatches.append(f"{row.get('id')}:missing")
        elif len(embedding) != expected_dimension:
            mismatches.append(f"{row.get('id')}:{len(embedding)}")
    return GraphQualityCheck(
        name=name,
        ok=not mismatches,
        count=len(mismatches),
        details=mismatches[:20],
    )
