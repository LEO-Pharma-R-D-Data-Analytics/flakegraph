"""Knowledge graph exploration page."""

from __future__ import annotations

import hashlib
import html
import json
from typing import Any

import pandas as pd
import streamlit as st
from flakegraph_app.explorer import (
    build_graph_figure,
    filter_graph,
    graph_facets,
    layout_fallback_notice,
)
from flakegraph_app.graph_store import variant_sequence
from flakegraph_app.models import GraphDataset
from flakegraph_app.ui.theme import is_dark_theme


@st.fragment
def render_graph_dataset(dataset: GraphDataset) -> None:
    """Render summary, filters, graph canvas, details, communities, and evidence."""

    counts = dataset.counts
    summary = st.columns(5)
    for column, (label, key) in zip(
        summary,
        (
            ("Documents", "documents"),
            ("Entities", "nodes"),
            ("Relations", "edges"),
            ("Communities", "communities"),
            ("Evidence", "evidence"),
        ),
        strict=True,
    ):
        column.metric(label, f"{counts[key]:,}")

    facets = graph_facets(dataset)
    community_labels = {
        str(community.get("id")): str(community.get("title") or community.get("id"))
        for community in dataset.communities
    }
    filter_columns = st.columns([2, 2, 2])
    search = filter_columns[0].text_input(
        "Search",
        placeholder="Entity name or description",
        help="Filter visible entities by canonical name, alias, type, or description.",
    )
    node_types = filter_columns[1].multiselect(
        "Entity types",
        facets["node_types"],
        help="Show only selected ontology entity types; leave empty to show every type.",
    )
    relation_types = filter_columns[2].multiselect(
        "Relation types",
        facets["relation_types"],
        help="Show only selected edge types; leave empty to show every relation type.",
    )
    secondary = st.columns([2, 2, 1, 1])
    community_ids = secondary[0].multiselect(
        "Communities",
        facets["community_ids"],
        format_func=lambda value: community_labels.get(value, value),
        help="Focus the graph on one or more detected communities.",
    )
    confidence = secondary[1].slider(
        "Minimum confidence",
        0.0,
        1.0,
        0.0,
        0.05,
        help=(
            "Hide relations below this extraction confidence. Confidence is a "
            "property of an asserted relation; canonical entities carry no such "
            "score and are never hidden by it."
        ),
    )
    layout = secondary[2].selectbox(
        "Layout",
        ["Communities", "Hierarchy", "Radial"],
        help="Choose how the visible graph projection is positioned in the explorer.",
    )
    include_isolates = secondary[3].toggle(
        "Isolates",
        value=True,
        help="Include entities with no visible relationship after the current filters.",
    )
    nodes, edges = filter_graph(
        dataset,
        search=search,
        node_types=node_types,
        relation_types=relation_types,
        community_ids=community_ids,
        minimum_confidence=confidence,
        include_isolates=include_isolates,
    )
    st.caption(
        f"Showing {len(nodes):,} entities and {len(edges):,} relations. "
        "The canvas is capped at 1,200 entities; summary metrics cover the full graph."
    )
    fallback = layout_fallback_notice(str(layout), nodes, edges)
    if fallback:
        st.warning(fallback)
    canvas, details = st.columns([3, 1])
    with canvas:
        selection = st.plotly_chart(
            _cached_graph_figure(
                _projection_signature(nodes, edges),
                str(layout),
                is_dark_theme(),
                _nodes=nodes,
                _edges=edges,
            ),
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            config={"displaylogo": False, "scrollZoom": True},
            key="graph_canvas",
        )
    selected_kind, selected_id = _selection(selection)
    with details:
        _render_selection(dataset, selected_kind, selected_id, nodes, edges)

    review = st.segmented_control(
        "Review panel",
        ["Entities", "Relations", "Communities", "Evidence", "Run"],
        default="Entities",
        label_visibility="collapsed",
        key=f"graph_review_{dataset.graph_id}",
        help=(
            "Choose one graph review surface. Only the selected surface is loaded, "
            "which keeps large evidence and community collections responsive."
        ),
    )
    _render_review_panel(dataset, nodes, edges, str(review or "Entities"))


def _render_review_panel(
    dataset: GraphDataset,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    review: str,
) -> None:
    """Render only the selected review surface instead of every hidden tab body.

    Streamlit eagerly executes all ``st.tabs`` bodies. A graph with thousands of
    evidence rows and hundreds of communities therefore paid the serialization
    cost for every view after each filter or selection. This explicit switch keeps
    the same five review capabilities while materializing exactly one at a time.
    """

    if review == "Entities":
        columns = ["name", "primary_type", "description", "degree", "rank", "id"]
        st.dataframe(_review_frame(nodes, columns), width="stretch", hide_index=True)
        return
    if review == "Relations":
        columns = [
            "relation_type",
            "description",
            "confidence",
            "weight",
            "evidence_count",
            "id",
        ]
        st.dataframe(_review_frame(edges, columns), width="stretch", hide_index=True)
        return
    if review == "Communities":
        _render_communities(dataset)
        return
    if review == "Evidence":
        columns = [
            "quote",
            "subject_kind",
            "file_id",
            "page_number",
            "chunk_id",
            "subject_id",
        ]
        st.dataframe(
            _review_frame(dataset.evidence, columns),
            width="stretch",
            hide_index=True,
        )
        return
    # Both documents are collapsed, so without a heading the panel presents two
    # identical "{...}" handles and the reader has to open one to learn which.
    report, metrics = st.columns(2)
    report.markdown("**Run report**")
    report.caption("Stages, timings, and adapters recorded while the graph was built.")
    report.json(dict(dataset.run_report), expanded=False)
    metrics.markdown("**Graph metrics**")
    metrics.caption("Counts and structural measures computed over the finished graph.")
    metrics.json(dict(dataset.graph_metrics), expanded=False)


def _render_selection(
    dataset: GraphDataset,
    kind: str | None,
    selected_id: str | None,
    visible_nodes: list[dict[str, Any]],
    visible_edges: list[dict[str, Any]],
) -> None:
    """Show a selected node or relation with its directly linked evidence."""

    st.subheader("Details")
    if not selected_id:
        st.caption("Select an entity or relation on the canvas.")
        return
    rows = visible_nodes if kind == "node" else visible_edges
    selected = next((row for row in rows if str(row.get("id")) == selected_id), None)
    if selected is None:
        st.caption("The selected item is outside the current filter projection.")
        return
    title = selected.get("name") or selected.get("relation_type") or selected_id
    st.subheader(str(title))
    if selected.get("primary_type"):
        st.caption(str(selected["primary_type"]))
    st.text(str(selected.get("description") or "No description"))
    st.json(selected, expanded=False)
    evidence = [row for row in dataset.evidence if str(row.get("subject_id")) == selected_id]
    if evidence:
        st.markdown("**Evidence**")
        for row in evidence[:8]:
            st.html(
                '<div class="fg-detail">'
                f"{_escape(row.get('quote', ''))}"
                f'<div class="fg-subtle">Page {_escape(row.get("page_number", "—"))}</div>'
                "</div>"
            )


def _render_communities(dataset: GraphDataset) -> None:
    """Present community summaries and findings as readable expandable sections."""

    if not dataset.communities:
        st.info("This graph does not contain community reports.")
        return
    for community in sorted(
        dataset.communities,
        key=lambda row: (-float(row.get("rating") or 0), str(row.get("title", ""))),
    ):
        title = str(community.get("title") or community.get("id"))
        member_count = len(variant_sequence(community.get("member_node_ids")))
        with st.expander(f"{title} · {member_count} entities"):
            st.text(str(community.get("summary") or "No summary"))
            if community.get("rating") is not None:
                st.caption(f"Rating {community['rating']}")
            questions = variant_sequence(community.get("suggested_questions"))
            if questions:
                st.markdown("**Suggested questions**")
                st.dataframe(
                    pd.DataFrame({"Question": [str(question) for question in questions]}),
                    width="stretch",
                    hide_index=True,
                )


def _selection(selection: Any) -> tuple[str | None, str | None]:
    """Extract the custom entity identity from Streamlit's Plotly selection event."""

    try:
        points = selection.selection.points
    except (AttributeError, TypeError):
        return None, None
    if not points:
        return None, None
    custom = points[-1].get("customdata")
    if isinstance(custom, list | tuple) and len(custom) >= _SELECTION_ID_PARTS:
        return str(custom[0]), str(custom[1])
    return None, None


def _escape(value: Any) -> str:
    """Escape evidence text before rendering it in an HTML detail block."""

    return html.escape(str(value or ""))


def _review_frame(rows: Any, columns: list[str]) -> pd.DataFrame:
    """Build an Arrow-safe table projection from heterogeneous artifact history."""

    return pd.DataFrame(
        [
            {
                column: (
                    json.dumps(row.get(column), sort_keys=True)
                    if isinstance(row.get(column), list | tuple | dict)
                    else row.get(column)
                )
                for column in columns
            }
            for row in rows
        ],
        columns=columns,
    )


_SELECTION_ID_PARTS = 2


@st.cache_data(show_spinner="Laying out the graph…", max_entries=32)
def _cached_graph_figure(
    signature: str,
    layout: str,
    dark: bool,
    *,
    _nodes: list[dict[str, Any]],
    _edges: list[dict[str, Any]],
) -> Any:
    """Reuse deterministic Plotly layouts across explorer-only fragment reruns.

    Community layouts run a force-directed solver and are intentionally the most
    expensive explorer operation, so selecting a node or switching a details tab
    must not solve the same layout again.

    The projection is passed unhashed and identified by ``signature`` instead.
    Streamlit hashes every argument on each lookup, and walking twelve hundred
    node dictionaries costs more than a tenth of a second — paid on every rerun,
    including the ones the cache answers.

    ``dark`` participates in the key: the canvas colours are baked into the
    figure, so a cached light figure would survive a switch to a dark host and
    keep painting a white panel.
    """

    return build_graph_figure(_nodes, _edges, layout=layout, dark=dark)


def _projection_signature(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    """Identify one filtered projection cheaply enough to do on every rerun.

    Identity, count and order decide the layout, so hashing the ids alone answers
    the same question as hashing every field of every row.
    """

    digest = hashlib.blake2b(digest_size=16)
    for node in nodes:
        digest.update(f"{node.get('id', '')}\x1f".encode())
    digest.update(b"\x1e")
    for edge in edges:
        digest.update(f"{edge.get('id', '')}\x1f".encode())
    return digest.hexdigest()
