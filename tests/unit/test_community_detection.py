from __future__ import annotations

import networkx as nx

from kg_processor.application.community_detection import _add_weighted_edge, _community_seeds


def test_community_detection_omits_reports_when_total_weight_is_zero() -> None:
    """Ensure a graph with no positive support cannot produce misleading communities.

    Mere node adjacency with zero weight is insufficient.
    """

    graph: nx.Graph[str] = nx.Graph()
    graph.add_nodes_from(["node_a", "node_b"])
    graph.add_edge("node_a", "node_b", weight=0.0)

    assert (
        _community_seeds(
            graph,
            min_community_size=2,
            max_community_size=50,
            resolution=1.0,
            deterministic_seed=17,
        )
        == []
    )


def test_community_detection_accumulates_parallel_edge_weights() -> None:
    graph: nx.Graph[str] = nx.Graph()

    _add_weighted_edge(graph, "node_a", "node_b", 2.0)
    _add_weighted_edge(graph, "node_a", "node_b", 3.5)

    assert graph["node_a"]["node_b"]["weight"] == 5.5


def test_seeded_leiden_partition_is_stable_and_omits_isolated_nodes() -> None:
    """Ensure seeded Leiden is repeatable and does not report unsupported isolated nodes."""

    graph: nx.Graph[str] = nx.Graph()
    graph.add_weighted_edges_from(
        [
            ("a", "b", 5.0),
            ("b", "c", 5.0),
            ("a", "c", 5.0),
            ("d", "e", 5.0),
            ("e", "f", 5.0),
            ("d", "f", 5.0),
            ("c", "d", 0.01),
        ]
    )
    graph.add_node("isolated")

    first = _community_seeds(graph, 2, 50, 1.0, 17)
    second = _community_seeds(graph, 2, 50, 1.0, 17)

    assert first == second
    assert {frozenset(seed.member_ids) for seed in first} == {
        frozenset({"a", "b", "c"}),
        frozenset({"d", "e", "f"}),
    }
    assert all("isolated" not in seed.member_ids for seed in first)
