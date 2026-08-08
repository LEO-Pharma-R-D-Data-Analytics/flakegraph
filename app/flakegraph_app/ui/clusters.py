"""Register, select and manage the Kubernetes clusters the app runs against."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from flakegraph_app.cluster_catalog import (
    ClusterCatalog,
    ClusterProfile,
    read_catalog,
    remove_cluster,
    save_cluster,
    select_cluster,
    validate_cluster_name,
    validate_namespace,
)

_PENDING_REMOVAL = "pending_cluster_removal"


def render_cluster_selector(state_root: Path) -> ClusterProfile | None:
    """Choose the active cluster in the sidebar, and offer to register one.

    Returns the active cluster so callers can target it without re-reading the
    catalog. ``None`` means none is registered yet, which the fleet view explains
    rather than failing against an absent cluster.
    """

    catalog = read_catalog(state_root)
    if not catalog.clusters:
        st.caption("No clusters registered yet.")
        if st.button(
            "Add a cluster",
            icon=":material/add:",
            width="stretch",
            key="cluster_add_first",
        ):
            st.session_state["active_page"] = "clusters"
            st.rerun()
        return None

    names = [cluster.name for cluster in catalog.clusters]
    active = catalog.active()
    chosen = st.selectbox(
        "Cluster",
        names,
        index=names.index(active.name) if active and active.name in names else 0,
        key="cluster_selector",
        help="Kubernetes cluster that runs FlakeGraph workers and model servers.",
    )
    if chosen and (active is None or chosen != active.name):
        select_cluster(state_root, chosen)
        st.rerun()
    if st.button(
        "Manage clusters",
        icon=":material/settings:",
        width="stretch",
        key="cluster_manage",
    ):
        st.session_state["active_page"] = "clusters"
        st.rerun()
    return read_catalog(state_root).active()


def render_cluster_manager(state_root: Path) -> None:
    """Add, edit and remove clusters without leaving the application."""

    st.html('<div class="fg-kicker">Kubernetes</div>')
    st.title("Clusters")
    st.caption(
        "Register each cluster once. FlakeGraph targets the selected cluster's "
        "context and namespace for every command it runs."
    )

    catalog = read_catalog(state_root)
    existing = {cluster.name: cluster for cluster in catalog.clusters}

    edit_names = ["New cluster", *existing]
    selected = st.selectbox("Editing", edit_names, key="cluster_editor_target")
    current = existing.get(str(selected))

    with st.form("cluster_form"):
        name = st.text_input(
            "Name",
            value=current.name if current else "",
            disabled=current is not None,
            help="Stable identifier for this cluster within FlakeGraph.",
        )
        namespace = st.text_input(
            "Namespace",
            value=current.namespace if current else "flakegraph",
            help="Namespace holding FlakeGraph workers and model servers.",
        )
        context = st.text_input(
            "Kubeconfig context",
            value=current.context if current else "",
            help=(
                "Context to select before each command. Leave empty to use "
                "whichever context the kubeconfig already points at."
            ),
        )
        kubeconfig = st.text_input(
            "Kubeconfig path",
            value=current.kubeconfig if current else "",
            help=(
                "Path to the kubeconfig file. It is referenced, never copied — it "
                "holds cluster credentials, which do not belong in app state."
            ),
        )
        description = st.text_input(
            "Description",
            value=current.description if current else "",
            help="What this cluster is for, so the choice is obvious later.",
        )
        submitted = st.form_submit_button(
            "Save cluster" if current else "Register cluster",
            type="primary",
        )

    if submitted:
        try:
            profile = ClusterProfile(
                name=current.name if current else validate_cluster_name(name),
                namespace=validate_namespace(namespace),
                context=context.strip(),
                kubeconfig=kubeconfig.strip(),
                description=description.strip(),
            )
        except ValueError as error:
            st.error(str(error))
        else:
            save_cluster(state_root, profile)
            st.rerun()

    if not catalog.clusters:
        return

    st.subheader("Registered clusters")
    for cluster in catalog.clusters:
        row = st.columns([3, 2, 2, 1])
        active_marker = " (selected)" if cluster.name == catalog.selected else ""
        row[0].markdown(f"**{cluster.name}**{active_marker}")
        if cluster.description:
            row[0].caption(cluster.description)
        row[1].caption(f"namespace `{cluster.namespace}`")
        row[2].caption(f"context `{cluster.context or 'default'}`")
        if row[3].button(
            "",
            icon=":material/delete:",
            key=f"cluster_delete_{cluster.name}",
            help=f"Remove {cluster.name} from FlakeGraph. The cluster is untouched.",
        ):
            st.session_state[_PENDING_REMOVAL] = cluster.name
    _render_removal_confirmation(state_root, catalog)


def _render_removal_confirmation(state_root: Path, catalog: ClusterCatalog) -> None:
    """Confirm removal, as every other destructive control in the app does.

    Removing the selected cluster moves the selection, so a mis-click retargets
    every subsequent fleet command at a different cluster.
    """

    pending = str(st.session_state.get(_PENDING_REMOVAL) or "")
    if not pending or catalog.get(pending) is None:
        return

    def _dismiss() -> None:
        """Disarm the confirmation when it is closed without an answer.

        Streamlit ignores a dismissal by default, which would leave the pending
        selection set and reopen the dialog on the operator's next interaction.
        """

        st.session_state[_PENDING_REMOVAL] = ""

    @st.dialog("Remove cluster", on_dismiss=_dismiss)
    def _confirm() -> None:
        st.write(
            f"Remove **{pending}** from FlakeGraph? The cluster itself is untouched; "
            "only this app forgets how to reach it."
        )
        if pending == catalog.selected:
            st.warning("This is the selected cluster. Removing it moves the selection.")
        remove, keep = st.columns(2)
        if remove.button("Remove", type="primary", width="stretch"):
            remove_cluster(state_root, pending)
            st.session_state[_PENDING_REMOVAL] = ""
            st.rerun()
        if keep.button("Keep", width="stretch"):
            st.session_state[_PENDING_REMOVAL] = ""
            st.rerun()

    _confirm()
