"""Interactive Plotly graph projection and filter logic for Streamlit."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import networkx as nx
import plotly.graph_objects as go
from flakegraph_app.graph_store import variant_sequence
from flakegraph_app.models import GraphDataset

_TYPE_COLORS = (
    "#16806A",
    "#D5654F",
    "#D39A25",
    "#3468A4",
    "#8056A6",
    "#6A7B2C",
    "#B34D78",
    "#397E8E",
    "#9A5C2D",
    "#536273",
)
_MAX_CANVAS_NODES = 1_200
_SMALL_LAYOUT_NODE_LIMIT = 300
# Groups this small are placed directly rather than handed to a solver.
_DIRECT_GROUP_LIMIT = 2
# NetworkX solves a spring layout densely below this size and switches to its
# SciPy sparse solver at it. Kamada-Kawai needs SciPy at any size.
_SPARSE_LAYOUT_NODE_LIMIT = 500
# Kamada-Kawai solves all-pairs distances, so its cost climbs steeply: measured on
# a real graph, 300 members take 0.8s and 945 take 9.3s, which the reader waits
# through on every filter change. Larger groups are placed by the force-directed
# solver instead, which handles the same 945 members in 0.6s.
_HIERARCHY_SOLVER_LIMIT = 300


def filter_graph(
    dataset: GraphDataset,
    *,
    search: str = "",
    node_types: Sequence[str] = (),
    relation_types: Sequence[str] = (),
    community_ids: Sequence[str] = (),
    minimum_confidence: float = 0.0,
    include_isolates: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply review filters while preserving only edges with visible endpoints."""

    normalized_search = search.strip().casefold()
    allowed_types = set(node_types)
    allowed_relations = set(relation_types)
    allowed_communities = set(community_ids)
    membership = _community_membership(dataset.communities)
    nodes = []
    for raw in dataset.nodes:
        node = dict(raw)
        node_id = str(node.get("id", ""))
        haystack = f"{node.get('name', '')} {node.get('description', '')}".casefold()
        if normalized_search and normalized_search not in haystack:
            continue
        if allowed_types and str(node.get("primary_type", "Unknown")) not in allowed_types:
            continue
        if allowed_communities and not (set(membership.get(node_id, ())) & allowed_communities):
            continue
        nodes.append(node)
    visible = {str(node.get("id")) for node in nodes}
    edges = [
        dict(edge)
        for edge in dataset.edges
        if str(edge.get("source_node_id")) in visible
        and str(edge.get("target_node_id")) in visible
        and (not allowed_relations or str(edge.get("relation_type")) in allowed_relations)
        and _float(edge.get("confidence")) >= minimum_confidence
    ]
    if not include_isolates:
        connected = {
            str(endpoint)
            for edge in edges
            for endpoint in (edge.get("source_node_id"), edge.get("target_node_id"))
        }
        nodes = [node for node in nodes if str(node.get("id")) in connected]
    nodes = sorted(
        nodes,
        key=lambda node: (-_float(node.get("degree")), str(node.get("name", "")).casefold()),
    )[:_MAX_CANVAS_NODES]
    visible = {str(node.get("id")) for node in nodes}
    edges = [
        edge
        for edge in edges
        if str(edge.get("source_node_id")) in visible and str(edge.get("target_node_id")) in visible
    ]
    return nodes, edges


# The canvas is transparent in both themes so it inherits whatever background the
# host paints. Streamlit does not report its theme on every supported version, so
# a canvas that had to be told which theme it was in would render a lit panel on a
# dark page precisely where detection is unavailable.
_LIGHT_CANVAS = {
    "paper": "rgba(0,0,0,0)",
    "plot": "rgba(0,0,0,0)",
    # Outlines separate overlapping nodes. A translucent dark edge reads on a
    # light background and disappears into a dark one, which is the intent.
    "node_outline": "rgba(23,33,43,0.35)",
    "edge": "rgba(83,98,115,0.28)",
    "font": "#17212b",
}
_DARK_CANVAS = {
    "paper": "rgba(0,0,0,0)",
    "plot": "rgba(0,0,0,0)",
    "node_outline": "rgba(0,0,0,0.45)",
    "edge": "rgba(154,167,180,0.34)",
    "font": "#e6ecf2",
}


def build_graph_figure(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    *,
    layout: str = "Communities",
    dark: bool = False,
) -> go.Figure:
    """Build a stable, selectable graph canvas with edge and entity hover detail.

    Plotly paints its own canvas, so the surrounding theme has to be passed in
    explicitly. A fixed white canvas reads as a lit panel on a dark page and
    washes out the node palette it is meant to separate.
    """

    canvas = _DARK_CANVAS if dark else _LIGHT_CANVAS

    node_by_id = {str(node.get("id")): node for node in nodes}
    graph = _projection(nodes, edges)
    positions = _positions(graph, layout)
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    midpoint_x: list[float] = []
    midpoint_y: list[float] = []
    edge_hover: list[str] = []
    edge_ids: list[str] = []
    for edge in edges:
        source = str(edge.get("source_node_id"))
        target = str(edge.get("target_node_id"))
        if source not in positions or target not in positions:
            continue
        source_position, target_position = positions[source], positions[target]
        edge_x.extend((source_position[0], target_position[0], None))
        edge_y.extend((source_position[1], target_position[1], None))
        midpoint_x.append((source_position[0] + target_position[0]) / 2)
        midpoint_y.append((source_position[1] + target_position[1]) / 2)
        edge_ids.append(str(edge.get("id", "")))
        edge_hover.append(
            f"<b>{_escape(node_by_id[source].get('name'))}</b> → "
            f"<b>{_escape(node_by_id[target].get('name'))}</b><br>"
            f"{_escape(edge.get('relation_type'))}<br>"
            f"Confidence {_float(edge.get('confidence')):.0%}"
        )

    node_types = sorted({str(node.get("primary_type", "Unknown")) for node in nodes})
    type_colors = {
        name: _TYPE_COLORS[index % len(_TYPE_COLORS)] for index, name in enumerate(node_types)
    }
    degrees = dict(graph.degree())
    node_ids = list(node_by_id)
    # Deliberately SVG rather than Scattergl: Snowsight renders the app inside a
    # sandboxed frame without WebGL, where a WebGL trace shows only "WebGL is not
    # supported by your browser" in place of the graph. The canvas is capped at
    # 1,200 entities, which SVG handles comfortably.
    node_trace = go.Scatter(
        x=[positions[node_id][0] for node_id in node_ids],
        y=[positions[node_id][1] for node_id in node_ids],
        mode="markers",
        customdata=[["node", node_id] for node_id in node_ids],
        marker={
            "size": [
                min(28, 9 + math.sqrt(max(0, degrees.get(node_id, 0))) * 3.2)
                for node_id in node_ids
            ],
            "color": [
                type_colors[str(node_by_id[node_id].get("primary_type", "Unknown"))]
                for node_id in node_ids
            ],
            "line": {"color": canvas["node_outline"], "width": 1.2},
            "opacity": 0.94,
        },
        text=[
            f"<b>{_escape(node.get('name'))}</b><br>"
            f"{_escape(node.get('primary_type'))}<br>"
            f"Degree {degrees.get(node_id, 0)}<br>"
            f"{_escape(str(node.get('description', ''))[:240])}"
            for node_id, node in ((node_id, node_by_id[node_id]) for node_id in node_ids)
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Entities",
    )
    figure = go.Figure(
        data=[
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": canvas["edge"], "width": 1.1},
                hoverinfo="skip",
                name="Relations",
            ),
            go.Scatter(
                x=midpoint_x,
                y=midpoint_y,
                mode="markers",
                customdata=[["edge", edge_id] for edge_id in edge_ids],
                marker={"size": 10, "color": "rgba(0,0,0,0)"},
                text=edge_hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            node_trace,
        ]
    )
    figure.update_layout(
        height=680,
        margin={"l": 8, "r": 8, "t": 12, "b": 8},
        paper_bgcolor=canvas["paper"],
        plot_bgcolor=canvas["plot"],
        font={"color": canvas["font"]},
        hovermode="closest",
        dragmode="pan",
        showlegend=False,
        xaxis={"visible": False, "fixedrange": False},
        yaxis={"visible": False, "fixedrange": False, "scaleanchor": "x", "scaleratio": 1},
    )
    return figure


def graph_facets(dataset: GraphDataset) -> dict[str, list[str]]:
    """Return sorted filter values from the loaded review projection."""

    return {
        "node_types": sorted({str(node.get("primary_type", "Unknown")) for node in dataset.nodes}),
        "relation_types": sorted(
            {str(edge.get("relation_type", "Unknown")) for edge in dataset.edges}
        ),
        "community_ids": sorted(str(community.get("id")) for community in dataset.communities),
    }


def layout_fallback_notice(
    layout: str,
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> str | None:
    """Say when a chosen layout cannot be solved and a ring is drawn instead.

    The circular fallback keeps the explorer usable where the numeric package is
    absent, but a ring is not a degraded hierarchy — it carries no structure at
    all. Presented without a word, a graph whose communities and hierarchy both
    render as the same evenly spaced circle reads as a finding about the graph.

    Layouts are solved per connected group, so it is the largest group rather than
    the entity total that decides whether a numeric solver is reached at all.
    """

    if _numeric_layout_available():
        return None
    largest = max(
        (len(members) for members in nx.connected_components(_projection(nodes, edges))),
        default=0,
    )
    if layout == "Hierarchy" and largest > _DIRECT_GROUP_LIMIT:
        reason = "A hierarchy layout is solved numerically"
    elif layout != "Radial" and largest >= _SPARSE_LAYOUT_NODE_LIMIT:
        reason = (
            f"A community layout with a connected group of "
            f"{_SPARSE_LAYOUT_NODE_LIMIT:,} entities or more is solved numerically"
        )
    else:
        return None
    return (
        f"{reason}, and this deployment has no SciPy installed. The entities below "
        "are placed in a ring, so their positions carry no meaning. Add `scipy` to "
        "the environment to restore the layout; filters, details, and every count "
        "on this page are unaffected."
    )


@lru_cache(maxsize=1)
def _numeric_layout_available() -> bool:
    """Report whether the optional numeric solver behind the layouts is installed."""

    try:
        return importlib.util.find_spec("scipy") is not None
    except (ImportError, ValueError):
        return False


def _projection(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> nx.Graph[Any]:
    """Build the undirected projection the canvas is laid out and sized from."""

    graph: nx.Graph[Any] = nx.Graph()
    graph.add_nodes_from(str(node.get("id")) for node in nodes)
    for edge in edges:
        graph.add_edge(
            str(edge.get("source_node_id")),
            str(edge.get("target_node_id")),
            weight=max(_float(edge.get("weight")), 0.05),
        )
    return graph


def _positions(graph: nx.Graph[Any], layout: str) -> Mapping[str, tuple[float, float]]:
    """Compute deterministic positions suited to the selected inspection intent.

    NetworkX delegates its force-directed layouts to SciPy for larger graphs. The
    application extra installs that optimized path, and a circular fallback keeps
    the explorer usable where an incomplete environment omits the optional numeric
    package instead of exposing a dependency traceback in the UI.
    ``layout_fallback_notice`` tells the reader when that has happened.
    """

    if not graph:
        return {}
    if layout == "Radial":
        return _plain(nx.circular_layout(graph))
    try:
        return _packed(graph, layout)
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        return _plain(nx.circular_layout(graph))


def _packed(graph: nx.Graph[Any], layout: str) -> Mapping[str, tuple[float, float]]:
    """Solve each connected group on its own, then pack the results side by side.

    Solving the whole projection at once lets unconnected entities decide the
    extent: nothing pulls them back towards the rest, so every connected group is
    squeezed until its relations are drawn shorter than the markers sitting on top
    of them, and a graph with relations reads as a field of loose dots. Each group
    is given canvas area in proportion to how many entities it holds, so a
    relation stays visible however many entities stand alone.
    """

    cells = [
        (math.sqrt(len(members)), _group_positions(graph.subgraph(members), layout))
        for members in sorted(nx.connected_components(graph), key=len, reverse=True)
    ]
    # Wrap into rows at the width of a square of the same total area, which packs
    # the groups into a roughly square block whatever their individual sizes.
    row_width = math.sqrt(sum((2 * radius) ** 2 for radius, _ in cells))
    positions: dict[str, tuple[float, float]] = {}
    left = top = row_height = 0.0
    for radius, group in cells:
        width = 2 * radius
        if left and left + width > row_width:
            left, top, row_height = 0.0, top - row_height, 0.0
        for node_id, (x, y) in group.items():
            positions[node_id] = (left + radius + x * radius, top - radius + y * radius)
        left += width
        row_height = max(row_height, width)
    return positions


def _group_positions(
    group: nx.Graph[Any],
    layout: str,
) -> Mapping[str, tuple[float, float]]:
    """Place one connected group in the unit square, deterministically.

    Groups of one or two entities are placed directly: a solver adds nothing to a
    single point or a single pair, and Kamada-Kawai has no distances to minimize.
    """

    members = sorted(group)
    if len(members) == 1:
        return {members[0]: (0.0, 0.0)}
    if len(members) == _DIRECT_GROUP_LIMIT:
        return {members[0]: (-1.0, 0.0), members[1]: (1.0, 0.0)}
    # Both solvers place nodes by their position in the graph's iteration order,
    # and a subgraph view iterates a set once the group is small relative to the
    # whole graph. Rebuilding the group in sorted order keeps one graph drawn the
    # same way in every process rather than only within one.
    ordered: nx.Graph[Any] = nx.Graph()
    ordered.add_nodes_from(members)
    ordered.add_edges_from(group.edges(data=True))
    if layout == "Hierarchy" and len(members) <= _HIERARCHY_SOLVER_LIMIT:
        solved = nx.kamada_kawai_layout(ordered, weight="weight")
    else:
        seed = int(hashlib.sha256("|".join(members).encode()).hexdigest()[:8], 16)
        iterations = 100 if len(members) <= _SMALL_LAYOUT_NODE_LIMIT else 40
        solved = nx.spring_layout(ordered, seed=seed, weight="weight", iterations=iterations)
    return _unit_square(solved)


def _unit_square(solved: Mapping[Any, Any]) -> Mapping[str, tuple[float, float]]:
    """Centre solved positions on the origin and scale them to fill [-1, 1].

    Both axes take the same scale, so a group that a solver laid out wide and flat
    stays wide and flat instead of being stretched to a square.
    """

    xs = [float(value[0]) for value in solved.values()]
    ys = [float(value[1]) for value in solved.values()]
    centre_x, centre_y = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    extent = max(max(xs) - min(xs), max(ys) - min(ys))
    scale = 2.0 / extent if extent else 1.0
    return {
        str(key): ((float(value[0]) - centre_x) * scale, (float(value[1]) - centre_y) * scale)
        for key, value in solved.items()
    }


def _plain(solved: Mapping[Any, Any]) -> Mapping[str, tuple[float, float]]:
    """Convert a NetworkX layout's arrays into plain plotting coordinates."""

    return {str(key): (float(value[0]), float(value[1])) for key, value in solved.items()}


def _community_membership(
    communities: Sequence[Mapping[str, Any]],
) -> Mapping[str, list[str]]:
    """Build node-to-community membership from canonical community rows."""

    result: dict[str, list[str]] = defaultdict(list)
    for community in communities:
        community_id = str(community.get("id", ""))
        for node_id in variant_sequence(community.get("member_node_ids")):
            result[str(node_id)].append(community_id)
    return result


def _float(value: Any) -> float:
    """Convert optional numeric artifact values to a safe plotting value."""

    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _escape(value: Any) -> str:
    """Escape the small HTML subset interpreted by Plotly hover labels."""

    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
