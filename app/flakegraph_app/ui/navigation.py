"""Run-centric sidebar navigation shared by every application runtime."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import streamlit as st
from flakegraph_app.backends.base import ControlPlaneBackend
from flakegraph_app.models import (
    ARTIFACTS_UNAVAILABLE_STATUS,
    ClusterSnapshot,
    RunSnapshot,
    RuntimeMode,
    StorageKind,
)
from flakegraph_app.ui.cache_state import invalidate_run_history
from flakegraph_app.ui.clusters import render_cluster_selector

_HISTORY_PAGE_SIZE = 15
_BULK_PREVIEW_LIMIT = 5


def render_run_navigation(
    backend: ControlPlaneBackend,
    runtime: RuntimeMode,
    runs: Sequence[RunSnapshot],
    cluster: ClusterSnapshot | None = None,
    cluster_error: str | None = None,
) -> tuple[str, RunSnapshot | None]:
    """Render runtime destinations and recent runs like a conversation history."""

    runtime_key = runtime.value.lower()
    if st.session_state.get("navigation_runtime") != runtime_key:
        st.session_state["navigation_runtime"] = runtime_key
        st.session_state["active_page"] = "fleet" if runtime == RuntimeMode.KUBERNETES else "new"
        st.session_state["selected_run_id"] = None

    active_page = str(st.session_state.get("active_page") or "new")
    if st.button(
        "New graph",
        icon=":material/add_circle:",
        type="primary" if active_page == "new" else "secondary",
        width="stretch",
        key=f"new_graph_{runtime_key}",
        help="Configure a new source, processing providers, and graph destination.",
    ):
        st.session_state["active_page"] = "new"
        st.session_state["selected_run_id"] = None
        st.rerun()

    if runtime == RuntimeMode.KUBERNETES:
        # A fleet is rarely one cluster, and machines are added over time, so the
        # target is chosen here rather than fixed in the environment.
        render_cluster_selector(_state_root(backend))
        fleet_label = _fleet_label(cluster, cluster_error)
        if st.button(
            fleet_label,
            icon=":material/dns:",
            type="primary" if active_page == "fleet" else "tertiary",
            width="stretch",
            key="kubernetes_fleet",
            help=(
                "Inspect registered nodes, model servers, workers, utilization, and "
                "on-demand active task assignments."
            ),
        ):
            st.session_state["active_page"] = "fleet"
            st.session_state["selected_run_id"] = None
            st.rerun()

    _render_graph_history_heading(runtime_key, "forget" in backend.capabilities)
    if not runs:
        st.caption("Submitted graphs will appear here.")
        if active_page == "clusters":
            return ("clusters", None)
        return ("fleet", None) if active_page == "fleet" else ("new", None)

    visible_runs = _render_run_filters(runtime_key, runs)
    selected_id = st.session_state.get("selected_run_id")
    selected = next((run for run in runs if run.run_id == selected_id), None)
    can_forget = "forget" in backend.capabilities
    bulk_mode = can_forget and _bulk_mode(runtime_key)
    if not visible_runs:
        st.caption("No graphs match the current filters.")
    history_limit_key = f"graph_history_limit_{runtime_key}"
    history_limit = int(st.session_state.get(history_limit_key, _HISTORY_PAGE_SIZE))
    paged_runs = _history_page(visible_runs, history_limit, selected_id)
    if bulk_mode:
        _render_bulk_actions(runtime_key, runs, paged_runs)
    _render_run_list(
        runtime_key,
        paged_runs,
        selected_id,
        active_page,
        can_forget,
        bulk_mode,
    )
    if len(visible_runs) > history_limit:
        remaining = len(visible_runs) - history_limit
        if st.button(
            f"Show {min(_HISTORY_PAGE_SIZE, remaining)} older",
            icon=":material/expand_more:",
            type="tertiary",
            width="stretch",
            key=f"more_graphs_{runtime_key}",
            help="Load the next page of older graph runs into the sidebar.",
        ):
            st.session_state[history_limit_key] = history_limit + _HISTORY_PAGE_SIZE
            st.rerun()
    if can_forget:
        _render_forget_confirmation(backend, runtime_key, runs, selected_id)
        _render_bulk_forget_confirmation(backend, runtime_key, runs, selected_id)

    if active_page == "clusters":
        return ("clusters", None)
    return (
        ("run", selected)
        if active_page == "run" and selected is not None
        else (("fleet", None) if active_page == "fleet" else ("new", None))
    )


def _render_graph_history_heading(runtime_key: str, can_forget: bool) -> None:
    """Render the graph heading and an unobtrusive bulk-selection mode toggle."""

    columns = st.columns([7, 1], vertical_alignment="center") if can_forget else [st.container()]
    with columns[0]:
        st.markdown("#### Graphs")
    if not can_forget:
        return
    with columns[1]:
        bulk_mode = _bulk_mode(runtime_key)
        if st.button(
            "",
            icon=":material/close:" if bulk_mode else ":material/checklist:",
            type="tertiary",
            width="stretch",
            key=f"toggle_bulk_graphs_{runtime_key}",
            help=(
                "Finish selecting graphs and return to normal history navigation."
                if bulk_mode
                else "Select multiple terminal graphs to remove from app history together."
            ),
        ):
            st.session_state[f"bulk_graph_mode_{runtime_key}"] = not bulk_mode
            if bulk_mode:
                _clear_bulk_selection(runtime_key)
            st.rerun()


def _render_run_filters(
    runtime_key: str,
    runs: Sequence[RunSnapshot],
) -> list[RunSnapshot]:
    """Render compact history filters and return the matching newest-first runs."""

    query = st.text_input(
        "Search graphs",
        placeholder="Search by graph or run ID",
        label_visibility="collapsed",
        key=f"graph_search_{runtime_key}",
        icon=":material/search:",
        help="Filter history by graph ID, run ID, or storage location.",
    )
    storage_options: list[str | StorageKind] = ["All", *list(StorageKind)]
    selected_storage = st.selectbox(
        "Storage",
        storage_options,
        format_func=lambda item: item.value if isinstance(item, StorageKind) else item,
        key=f"graph_storage_filter_{runtime_key}",
        label_visibility="collapsed",
        help="Show graphs from every destination or only local/Snowflake storage.",
    )
    return _filter_runs(runs, query, selected_storage)


def _render_run_list(
    runtime_key: str,
    runs: Sequence[RunSnapshot],
    selected_id: object,
    active_page: str,
    can_forget: bool,
    bulk_mode: bool,
) -> None:
    """Render graph navigation with single or opt-in bulk removal controls."""

    with st.container(key="graph_run_list"):
        for run in runs:
            columns = (
                st.columns([1, 8], gap="small")
                if bulk_mode
                else (st.columns([5, 1], gap="small") if can_forget else [st.container()])
            )
            if bulk_mode:
                with columns[0]:
                    st.checkbox(
                        f"Select {_short_name(run.display_name)}",
                        key=_bulk_checkbox_key(runtime_key, run.run_id),
                        disabled=_active_status(run.status),
                        label_visibility="collapsed",
                        help=(
                            "Active graphs cannot be removed. Cancel this run before "
                            "removing it from history."
                            if _active_status(run.status)
                            else f"Include {_short_name(run.display_name)} in the bulk action."
                        ),
                    )
                with columns[1]:
                    _render_run_button(runtime_key, run, selected_id, active_page)
            else:
                with columns[0]:
                    _render_run_button(runtime_key, run, selected_id, active_page)
            if can_forget and not bulk_mode:
                with columns[1]:
                    if st.button(
                        "",
                        icon=":material/delete_outline:",
                        type="tertiary",
                        width="stretch",
                        disabled=_active_status(run.status),
                        help=(
                            "Remove this terminal run from app history. Graph artifacts "
                            "and distributed audit data are preserved. Active runs must "
                            "be cancelled first."
                        ),
                        key=f"forget_{runtime_key}_{run.run_id}",
                    ):
                        st.session_state["pending_forget_run_id"] = run.run_id
                        # Re-enter the script so the confirmation dialog opens at
                        # once. Rendering it below a long history list made the
                        # first click appear inert because the confirmation was
                        # outside the visible sidebar viewport.
                        st.rerun()


def _render_bulk_actions(
    runtime_key: str,
    all_runs: Sequence[RunSnapshot],
    shown_runs: Sequence[RunSnapshot],
) -> None:
    """Render compact selection helpers only while bulk-selection mode is active."""

    selected = _selected_bulk_ids(runtime_key, all_runs)
    selectable = [run for run in shown_runs if not _active_status(run.status)]
    shown_selected = {
        run.run_id
        for run in selectable
        if bool(st.session_state.get(_bulk_checkbox_key(runtime_key, run.run_id)))
    }
    st.caption(f"{len(selected)} selected")
    actions = st.columns([1, 1, 5], gap="small")
    if actions[0].button(
        "",
        icon=":material/select_all:",
        type="tertiary",
        width="stretch",
        disabled=not selectable or len(shown_selected) == len(selectable),
        help="Select every removable graph currently shown in the sidebar.",
        key=f"bulk_select_all_{runtime_key}",
    ):
        for run in selectable:
            st.session_state[_bulk_checkbox_key(runtime_key, run.run_id)] = True
        st.rerun()
    if actions[1].button(
        "",
        icon=":material/deselect:",
        type="tertiary",
        width="stretch",
        disabled=not selected,
        help="Clear the current graph selection without leaving selection mode.",
        key=f"bulk_select_clear_{runtime_key}",
    ):
        _clear_bulk_selection(runtime_key)
        st.rerun()
    if actions[2].button(
        f"Remove {len(selected)}",
        icon=":material/delete_outline:",
        type="primary",
        width="stretch",
        disabled=not selected,
        help="Review and confirm removal of the selected graphs from app history.",
        key=f"bulk_remove_{runtime_key}",
    ):
        st.session_state[f"pending_bulk_forget_{runtime_key}"] = list(selected)
        st.rerun()


def _render_run_button(
    runtime_key: str,
    run: RunSnapshot,
    selected_id: object,
    active_page: str,
) -> None:
    """Render one full-width history entry and open its run workspace when selected."""

    label = f"{_short_name(run.display_name)}\n{run.storage_kind.value}"
    status_key = "active" if _active_status(run.status) else "terminal"
    st.button(
        label,
        icon=_status_icon(run.status),
        width="stretch",
        type=("primary" if run.run_id == selected_id and active_page == "run" else "tertiary"),
        help=(
            f"Open {run.display_name} (graph ID {run.graph_id}). Stored in "
            f"{run.storage_location or run.storage_kind.value}; run {run.run_id}; "
            f"updated {run.updated_at or 'not yet observed'}."
        ),
        key=f"run_{status_key}_{runtime_key}_{run.run_id}",
        on_click=_select_run,
        args=(run.run_id,),
    )


def _select_run(run_id: str) -> None:
    """Select a run before rerendering so remote fleet reads can be skipped."""

    st.session_state["active_page"] = "run"
    st.session_state["selected_run_id"] = run_id


def _render_forget_confirmation(
    backend: ControlPlaneBackend,
    runtime_key: str,
    runs: Sequence[RunSnapshot],
    selected_id: object,
) -> None:
    """Open a visible confirmation dialog for the selected terminal history entry."""

    pending_id = str(st.session_state.get("pending_forget_run_id") or "")
    pending = next((run for run in runs if run.run_id == pending_id), None)
    if pending is None:
        return
    _forget_dialog(backend, runtime_key, pending, selected_id)


def _render_bulk_forget_confirmation(
    backend: ControlPlaneBackend,
    runtime_key: str,
    runs: Sequence[RunSnapshot],
    selected_id: object,
) -> None:
    """Open the bulk confirmation dialog when a selected run set is pending."""

    pending_key = f"pending_bulk_forget_{runtime_key}"
    pending_ids = {str(run_id) for run_id in st.session_state.get(pending_key, []) if str(run_id)}
    pending = [run for run in runs if run.run_id in pending_ids and not _active_status(run.status)]
    if not pending:
        st.session_state.pop(pending_key, None)
        st.session_state.pop(f"bulk_forget_errors_{runtime_key}", None)
        return
    _bulk_forget_dialog(backend, runtime_key, pending, selected_id)


def _dismiss_forget() -> None:
    """Disarm the single-graph confirmation when its dialog is closed unconfirmed.

    Closing with the ✕, Escape, or a click outside is a decision to keep the
    graph. The pending selection is what makes the dialog open, so leaving it in
    place would reopen the same confirmation on the operator's next click
    anywhere in the sidebar.
    """

    st.session_state.pop("pending_forget_run_id", None)


def _dismiss_bulk_forget() -> None:
    """Disarm bulk confirmation state when its dialog is closed unconfirmed.

    A dismissal callback receives no arguments, and only one dialog can be open
    at a time, so every runtime's pending selection and retry errors are cleared
    rather than guessing which runtime armed the one being closed.
    """

    for prefix in ("pending_bulk_forget_", "bulk_forget_errors_"):
        for key in [str(key) for key in st.session_state if str(key).startswith(prefix)]:
            st.session_state.pop(key, None)


@st.dialog("Remove this graph?", on_dismiss=_dismiss_forget)
def _forget_dialog(
    backend: ControlPlaneBackend,
    runtime_key: str,
    pending: RunSnapshot,
    selected_id: object,
) -> None:
    """Confirm one deletion, and say exactly what it takes with it.

    Deleting is the only removal offered. A bin icon beside a graph reads as
    "delete this graph", and an option that merely hid it left an operator
    believing their documents were gone while every entity, quote and page of
    source text remained in the account.
    """

    st.write(f"**{_short_name(pending.display_name)}**")
    stored = _stored_row_counts(backend, pending.graph_id)
    if stored:
        total = sum(stored.values())
        st.caption(
            f"Deletes **{total:,} rows** across {len(stored)} tables — its entities, "
            "evidence and extracted source text — along with the documents uploaded "
            "for it. This cannot be undone."
        )
    else:
        st.caption("Deletes this graph and everything recorded about it. This cannot be undone.")
    confirmation = st.columns(2)
    if confirmation[0].button(
        "Delete graph",
        icon=":material/delete_forever:",
        type="primary",
        width="stretch",
        help="Remove the graph, its documents, and every row recorded about it.",
        key=f"confirm_delete_{runtime_key}_{pending.run_id}",
    ):
        try:
            with st.spinner("Deleting graph and its data"):
                removed = _remove_graph(backend, pending)
        except Exception as exc:
            st.error(str(exc))
        else:
            _finish_removal(pending, selected_id)
            if removed:
                st.toast(f"Deleted {removed:,} rows")
            st.rerun()
    if confirmation[1].button(
        "Keep",
        width="stretch",
        help="Close this confirmation without changing anything.",
        key=f"cancel_forget_{runtime_key}_{pending.run_id}",
    ):
        st.session_state.pop("pending_forget_run_id", None)
        st.rerun()


def _remove_graph(backend: ControlPlaneBackend, run: RunSnapshot) -> int:
    """Delete one graph, and return how many rows went with it.

    Runtimes that keep their graphs in Snowflake delete the graph itself. Local
    and fleet runs store their graphs as files this application does not own, so
    there the run leaves the history and the artifacts stay where the operator
    put them.
    """

    if "delete_graph" in backend.capabilities:
        return sum(backend.delete_graph(run.graph_id).values())
    backend.forget(run.run_id, _config_path(run))
    return 0


def _stored_row_counts(backend: ControlPlaneBackend, graph_id: str) -> dict[str, int]:
    """Return what the destination still stores for one graph, if it can say."""

    if "delete_graph" not in backend.capabilities:
        return {}
    try:
        return dict(backend.graph_row_counts(graph_id))
    except Exception:
        # The count is here to inform the choice, not to gate it.
        return {}


def _finish_removal(pending: RunSnapshot, selected_id: object) -> None:
    """Clear the confirmation and leave the reader somewhere that still exists."""

    invalidate_run_history()
    if selected_id == pending.run_id:
        st.session_state["selected_run_id"] = None
        st.session_state["active_page"] = "new"
    st.session_state.pop("pending_forget_run_id", None)


@st.dialog("Remove selected graphs from history?", on_dismiss=_dismiss_bulk_forget)
def _bulk_forget_dialog(
    backend: ControlPlaneBackend,
    runtime_key: str,
    pending: Sequence[RunSnapshot],
    selected_id: object,
) -> None:
    """Confirm a bounded batch of history-only removals and report partial failures."""

    errors_key = f"bulk_forget_errors_{runtime_key}"
    prior_errors = st.session_state.get(errors_key)
    if isinstance(prior_errors, dict) and prior_errors:
        st.error(f"{len(prior_errors)} graph removal(s) failed. You can retry or keep them.")
    st.write(f"Remove **{len(pending)} graph{'s' if len(pending) != 1 else ''}**?")
    for run in pending[:_BULK_PREVIEW_LIMIT]:
        st.caption(f"• {_short_name(run.display_name, limit=48)}")
    if len(pending) > _BULK_PREVIEW_LIMIT:
        st.caption(f"• and {len(pending) - _BULK_PREVIEW_LIMIT} more")
    st.caption(
        "Only app history entries are removed. Stored graph artifacts and distributed "
        "audit data remain available."
    )
    confirmation = st.columns(2)
    if confirmation[0].button(
        f"Remove {len(pending)}",
        icon=":material/delete_outline:",
        type="primary",
        width="stretch",
        help="Confirm removal of every listed graph from this app's history.",
        key=f"confirm_bulk_forget_{runtime_key}",
    ):
        failures = _forget_runs(backend, pending)
        removed_ids = {run.run_id for run in pending} - set(failures)
        invalidate_run_history()
        for run_id in removed_ids:
            st.session_state.pop(_bulk_checkbox_key(runtime_key, run_id), None)
        if selected_id in removed_ids:
            st.session_state["selected_run_id"] = None
            st.session_state["active_page"] = "new"
        if failures:
            st.session_state[f"pending_bulk_forget_{runtime_key}"] = list(failures)
            st.session_state[errors_key] = failures
        else:
            st.session_state.pop(f"pending_bulk_forget_{runtime_key}", None)
            st.session_state.pop(errors_key, None)
            st.session_state[f"bulk_graph_mode_{runtime_key}"] = False
            _clear_bulk_selection(runtime_key)
        st.rerun()
    if confirmation[1].button(
        "Keep",
        width="stretch",
        help="Close this confirmation without removing the listed graphs.",
        key=f"cancel_bulk_forget_{runtime_key}",
    ):
        st.session_state.pop(f"pending_bulk_forget_{runtime_key}", None)
        st.session_state.pop(errors_key, None)
        st.rerun()


def _forget_runs(
    backend: ControlPlaneBackend,
    runs: Sequence[RunSnapshot],
) -> dict[str, str]:
    """Delete every graph independently so one backend error cannot abort the batch."""

    failures: dict[str, str] = {}
    # Each deletion clears every table the graph touches, so a bulk removal is
    # long enough that silence reads as a hung dialog. The bar names the graph.
    progress = st.progress(0.0, text="Deleting graphs")
    for index, run in enumerate(runs, start=1):
        progress.progress((index - 1) / len(runs), text=f"Deleting {run.graph_name}")
        try:
            _remove_graph(backend, run)
        except Exception as exc:
            failures[run.run_id] = str(exc)
    progress.empty()
    return failures


def _fleet_label(cluster: ClusterSnapshot | None, error: str | None) -> str:
    """Summarize Kubernetes reachability without crowding the sidebar."""

    if cluster is not None:
        return f"Fleet · {cluster.nodes_ready}/{cluster.nodes_total} ready"
    return "Fleet · unavailable" if error else "Fleet"


def _bulk_mode(runtime_key: str) -> bool:
    """Return whether the current runtime history is in explicit selection mode."""

    return bool(st.session_state.get(f"bulk_graph_mode_{runtime_key}", False))


def _bulk_checkbox_key(runtime_key: str, run_id: str) -> str:
    """Build the namespaced widget key used by one graph-selection checkbox."""

    return f"bulk_graph_selected_{runtime_key}_{run_id}"


def _selected_bulk_ids(
    runtime_key: str,
    runs: Sequence[RunSnapshot],
) -> list[str]:
    """Return selected terminal runs in the backend's stable history order."""

    return [
        run.run_id
        for run in runs
        if not _active_status(run.status)
        and bool(st.session_state.get(_bulk_checkbox_key(runtime_key, run.run_id)))
    ]


def _clear_bulk_selection(runtime_key: str) -> None:
    """Remove only checkbox state belonging to the current runtime selection mode."""

    prefix = f"bulk_graph_selected_{runtime_key}_"
    for key in [str(key) for key in st.session_state if str(key).startswith(prefix)]:
        st.session_state.pop(key, None)


def _filter_runs(
    runs: Sequence[RunSnapshot],
    query: str,
    storage: str | StorageKind,
) -> list[RunSnapshot]:
    """Filter graph history without changing its newest-first backend ordering."""

    needle = query.strip().casefold()
    expected_storage = storage if isinstance(storage, StorageKind) else None
    return [
        run
        for run in runs
        if (expected_storage is None or run.storage_kind == expected_storage)
        and (
            not needle
            or needle in run.display_name.casefold()
            or needle in run.graph_id.casefold()
            or needle in run.run_id.casefold()
            or needle in (run.storage_location or "").casefold()
        )
    ]


def _history_page(
    runs: Sequence[RunSnapshot],
    limit: int,
    selected_id: object,
) -> list[RunSnapshot]:
    """Bound sidebar widgets while keeping an older selected graph navigable."""

    page = list(runs[: max(1, limit)])
    selected = next((run for run in runs if run.run_id == selected_id), None)
    if selected is not None and selected not in page:
        page.append(selected)
    return page


def _short_name(value: str, limit: int = 24) -> str:
    """Keep long generated graph identifiers readable in a narrow sidebar."""

    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def _status_icon(status: str) -> str:
    """Choose familiar Material symbols for graph-run status."""

    normalized = status.lower()
    if normalized in {"succeeded", "completed", "done", "success"}:
        return ":material/hub:"
    if normalized in {"failed", "error"}:
        return ":material/error_outline:"
    if normalized == ARTIFACTS_UNAVAILABLE_STATUS:
        # Distinct from both success and failure: the graph exists, but not on
        # this machine, so the row must not read as one that will open.
        return ":material/cloud_off:"
    if normalized in {"cancelled", "canceled", "interrupted"}:
        return ":material/block:"
    return ":material/progress_activity:"


def _active_status(status: str) -> bool:
    """Return whether removing the run would hide work that is still executing."""

    return status.lower() in {
        "planning",
        "pending",
        "queued",
        "running",
        "processing",
        "claimed",
        "submitted",
    }


def _config_path(snapshot: RunSnapshot) -> Path | None:
    """Return the optional generated configuration used by backend run actions."""

    value = str(snapshot.raw.get("config_path") or "")
    return Path(value) if value else None


def _state_root(backend: ControlPlaneBackend) -> Path:
    """Return the app-owned catalog directory for the active backend."""

    state_root = getattr(backend, "state_root", None)
    if isinstance(state_root, Path):
        return state_root
    return Path(".flakegraph") / "app"
