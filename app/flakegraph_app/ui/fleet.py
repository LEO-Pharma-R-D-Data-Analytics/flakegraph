"""Professional Kubernetes fleet inventory and workload observability view."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st
from flakegraph_app.backends.base import ControlPlaneBackend
from flakegraph_app.models import ClusterSnapshot, NodeStatus
from flakegraph_app.ui.cache_state import invalidate_fleet_snapshot
from flakegraph_app.ui.shared import format_relative_time
from flakegraph_app.ui.theme import page_heading


def render_fleet(
    backend: ControlPlaneBackend,
    namespace: str,
    snapshot: ClusterSnapshot | None,
    error: str | None,
) -> None:
    """Render registered nodes, hardware, model placement, and live utilization.

    The backend has already normalized optional Kubernetes and Metrics Server data.
    This page therefore remains useful on a minimal cluster while progressively
    adding utilization and accelerator details when those APIs are available.
    """

    page_heading(
        "Infrastructure",
        "Compute fleet",
        "Registered Kubernetes nodes, colocated model servers, and FlakeGraph workers.",
    )
    controls = st.columns([3, 1, 0.8], vertical_alignment="bottom")
    if snapshot is not None:
        controls[0].caption(f"Context: {snapshot.context}")
    else:
        controls[0].caption("Kubernetes connection")
    # The namespace belongs to the selected cluster, and fleet reads are confined
    # to it. Offering it as a free-text field here would present a control whose
    # only accepted value is the one already shown.
    controls[1].caption(f"Namespace: `{namespace}`")
    if controls[2].button(
        "Refresh",
        icon=":material/refresh:",
        width="stretch",
        help="Refresh nodes, workloads, and resource telemetry.",
    ):
        invalidate_fleet_snapshot()
        st.rerun()

    if snapshot is None:
        st.info("Connect kubectl to a Kubernetes cluster to view its registered fleet.")
        if error:
            with st.expander("Connection details"):
                st.code(error)
        return

    _render_summary(snapshot)
    for warning in snapshot.warnings:
        st.warning(warning)

    st.subheader("Registered nodes")
    if not snapshot.nodes:
        st.info(f"No Kubernetes nodes are registered in context {snapshot.context}.")
    else:
        for node in sorted(snapshot.nodes, key=lambda item: (not item.ready, item.name)):
            st.html(_node_markup(node))
            _render_node_assignments(backend, namespace, node)

    show_workloads = st.toggle(
        f"Show workloads ({len(snapshot.workloads)})",
        value=False,
        key=f"show_fleet_workloads_{namespace}",
        help="Load the detailed pod table only when it is needed.",
    )
    if show_workloads:
        if not snapshot.workloads:
            st.caption(f"No FlakeGraph workloads were found in namespace {namespace}.")
            return
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Workload": item.name,
                        "Component": item.component,
                        "Status": item.phase,
                        "Ready": item.ready,
                        "Node": item.node,
                        "Restarts": item.restarts,
                        "CPU": item.cpu,
                        "Memory": item.memory,
                        "Model": item.model,
                        "Image": item.image,
                    }
                    for item in snapshot.workloads
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def _render_summary(snapshot: ClusterSnapshot) -> None:
    """Show the four fleet totals operators need before inspecting a node."""

    model_servers = sum(node.model_server_ready for node in snapshot.nodes)
    configured_models = sum(bool(node.model or node.model_image) for node in snapshot.nodes)
    summaries = st.columns(4)
    summaries[0].metric(
        "Nodes ready",
        f"{snapshot.nodes_ready}/{snapshot.nodes_total}",
        help="Kubernetes nodes currently reporting Ready versus all registered nodes.",
    )
    summaries[1].metric(
        "GPUs",
        sum(node.gpu_count for node in snapshot.nodes),
        help="Accelerators advertised by registered nodes.",
    )
    summaries[2].metric(
        "Workers",
        sum(node.worker_count for node in snapshot.nodes),
        help="FlakeGraph worker pods currently assigned across the fleet.",
    )
    summaries[3].metric(
        "Model servers",
        f"{model_servers}/{configured_models}",
        help="Ready colocated model servers versus nodes configured to host one.",
    )


def _node_markup(node: NodeStatus) -> str:
    """Build one escaped, responsive inventory row for a Kubernetes node."""

    status = "Ready" if node.ready else "Not ready"
    status_class = "ready" if node.ready else "offline"
    node_class = _human_node_class(node.node_class)
    gpu = (
        f"{node.gpu_count} × {node.gpu_model}"
        if node.gpu_count and node.gpu_model
        else (str(node.gpu_count) if node.gpu_count else "Not reported")
    )
    cpu = _resource_usage(node.cpu_usage, node.cpu_capacity, node.cpu_percent)
    memory = _resource_usage(node.memory_usage, node.memory_capacity, node.memory_percent)
    if node.model:
        model_state = "Ready" if node.model_server_ready else "Starting"
        model = f"{node.model} · {model_state}"
    elif node.model_image:
        model = "Ready" if node.model_server_ready else "Starting"
    else:
        model = "Not assigned"
    return (
        '<div class="fg-node">'
        '<div class="fg-node-identity">'
        f'<span class="fg-status-dot {status_class}"></span>'
        "<div>"
        f"<strong>{html.escape(node.name)}</strong>"
        f"<span>{html.escape(node_class)} · {html.escape(status)}</span>"
        "</div></div>"
        f"{_node_stat('GPU', gpu)}"
        f"{_node_stat('GPU usage', node.gpu_percent or 'Not reported')}"
        f"{_node_stat('Workers', str(node.worker_count))}"
        f"{_node_stat('CPU', cpu)}"
        f"{_node_stat('Memory', memory)}"
        f"{_node_stat('Local model', model, wide=True)}"
        "</div>"
    )


def _render_node_assignments(
    backend: ControlPlaneBackend,
    namespace: str,
    node: NodeStatus,
) -> None:
    """Lazily reveal bounded active tasks for one node on explicit request.

    Collapsing an expander does not prevent Streamlit from evaluating its body, so
    the database lookup sits behind a button and is cached only in session state.
    This keeps the default fleet page limited to Kubernetes inventory and metrics.
    """

    if "node_assignments" not in backend.capabilities or not node.worker_count:
        return
    cache_key = f"node_assignments_{namespace}_{node.name}"
    with st.expander(f"Active work · {node.name}", icon=":material/account_tree:"):
        loaded = cache_key in st.session_state
        if st.button(
            "Refresh active work" if loaded else "Load active work",
            icon=":material/refresh:" if loaded else ":material/visibility:",
            help=(
                "Query only tasks currently leased by workers on this node. The "
                "overview does not load task payloads, artifacts, or completed history."
            ),
            key=f"load_assignments_{namespace}_{node.name}",
        ):
            try:
                st.session_state[cache_key] = list(backend.node_assignments(namespace, node.name))
            except Exception as exc:
                st.error(f"Active work is unavailable: {exc}")
        assignments = st.session_state.get(cache_key)
        if assignments is None:
            st.caption("Load this node's current task assignments on demand.")
        elif not assignments:
            st.caption("No tasks are currently leased by workers on this node.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Run": item.run_id,
                            "Graph": item.graph_id,
                            "Stage": item.stage.replace("_", " ").title(),
                            "Task": item.task_id,
                            "Scope": item.scope_id,
                            "Worker": item.worker_id,
                            "Updated": format_relative_time(item.updated_at),
                        }
                        for item in assignments
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Updated": st.column_config.TextColumn(
                        help="Time elapsed since the worker last renewed or changed this task."
                    )
                },
            )


def _node_stat(label: str, value: str, *, wide: bool = False) -> str:
    """Render one compact label/value pair inside a node inventory row."""

    modifier = " fg-node-stat-wide" if wide else ""
    return (
        f'<div class="fg-node-stat{modifier}">'
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong>"
        "</div>"
    )


def _resource_usage(usage: str | None, capacity: str | None, percent: str | None) -> str:
    """Format optional resource usage without implying unavailable data is zero."""

    if usage and capacity:
        suffix = f" ({percent})" if percent else ""
        return f"{usage} / {capacity}{suffix}"
    return usage or capacity or "Not reported"


def _human_node_class(value: str) -> str:
    """Turn scheduling labels into concise operator-facing hardware names."""

    if value == "nvidia-spark":
        return "NVIDIA DGX Spark"
    return value.replace("-", " ").replace("_", " ").title()
