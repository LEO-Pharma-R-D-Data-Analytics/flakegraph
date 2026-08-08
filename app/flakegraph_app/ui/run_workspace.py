"""One status-aware workspace for an individual graph run."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from flakegraph_app.backends.base import ControlPlaneBackend
from flakegraph_app.models import RunSnapshot
from flakegraph_app.ui.cache_state import invalidate_run_history
from flakegraph_app.ui.consumption import render_consumption
from flakegraph_app.ui.graph_explorer import render_graph_dataset
from flakegraph_app.ui.shared import concise_error, render_run_snapshot
from flakegraph_app.ui.theme import page_heading

_SUCCESS_STATUSES = {"succeeded", "completed", "done", "success"}
_ACTIVE_STATUSES = {
    "pending",
    "planning",
    "queued",
    "running",
    "processing",
    "claimed",
    "submitted",
}
_RUN_GRAPH_CACHE_LIMIT = 4


def render_run_workspace(
    backend: ControlPlaneBackend,
    listed_snapshot: RunSnapshot,
) -> None:
    """Show live progress for active work and the graph explorer after success."""

    config_path = _config_path(listed_snapshot)
    snapshot = listed_snapshot
    if _requires_status_refresh(listed_snapshot.status):
        try:
            snapshot = backend.status(listed_snapshot.run_id, config_path)
        except Exception as exc:
            st.warning(
                "Live run status is temporarily unavailable. The last known state is "
                f"shown below; distributed work remains durable. {concise_error(exc)}"
            )

    normalized = snapshot.status.lower()
    if normalized in _SUCCESS_STATUSES:
        _render_completed_run(backend, snapshot, config_path)
        return
    if normalized in _ACTIVE_STATUSES:
        _render_active_run(backend, snapshot, config_path)
        return
    _render_terminal_run(backend, snapshot, config_path)


def _render_completed_run(
    backend: ControlPlaneBackend,
    snapshot: RunSnapshot,
    config_path: Path | None,
) -> None:
    """Open a finished run directly in the explorer with diagnostics secondary."""

    _render_graph_heading(
        backend,
        snapshot,
        "Graph",
        "Explore the finalized graph and its source evidence.",
    )
    _render_sharing(backend, snapshot.graph_id)
    explorer_tab, consumption_tab, details_tab = st.tabs(
        ["Explorer", "Consumption", "Run details"]
    )
    load_error: str | None = None
    with explorer_tab:
        cache_key = f"{snapshot.run_id}:{snapshot.updated_at or 'complete'}"
        cache = st.session_state.setdefault("run_graph_cache", {})
        dataset = cache.get(cache_key)
        if cache_key not in cache:
            with st.spinner("Loading graph artifacts"):
                try:
                    dataset = backend.load_run_graph(snapshot, config_path)
                except Exception as exc:
                    load_error = f"The graph could not be loaded: {exc}"
                    st.error(load_error)
                else:
                    _cache_graph_dataset(cache, cache_key, dataset)
        if dataset is not None:
            render_graph_dataset(dataset)
    with consumption_tab:
        if dataset is not None:
            render_consumption(dataset.graph_metrics, snapshot.graph_id)
        else:
            # Without the graph there is nothing to report on. Every other
            # explanation this tab can offer describes the graph, so falling
            # through to one would blame a missing feature for a failed read.
            st.error(load_error or "The graph could not be loaded.")
    with details_tab:
        _render_completed_details(backend, snapshot, config_path)


def _render_completed_details(
    backend: ControlPlaneBackend,
    snapshot: RunSnapshot,
    config_path: Path | None,
) -> None:
    """Show listed completion data and load detailed diagnostics only on request."""

    cache_key = f"run_status_{snapshot.run_id}_{snapshot.updated_at or 'complete'}"
    detailed = st.session_state.get(cache_key)
    if st.button(
        "Refresh run diagnostics",
        icon=":material/refresh:",
        help=(
            "Query the runtime for detailed stages, warnings, and recent events. "
            "The graph opens without this additional remote round trip."
        ),
        key=f"refresh_run_diagnostics_{snapshot.run_id}",
    ):
        try:
            detailed = backend.status(snapshot.run_id, config_path)
        except Exception as exc:
            st.error(f"Run diagnostics are unavailable: {exc}")
        else:
            st.session_state[cache_key] = detailed
    render_run_snapshot(detailed if isinstance(detailed, RunSnapshot) else snapshot)


def _render_active_run(
    backend: ControlPlaneBackend,
    initial: RunSnapshot,
    config_path: Path | None,
) -> None:
    """Refresh processing progress until the graph is finalized."""

    _render_graph_heading(
        backend,
        initial,
        "Ingestion",
        "Documents are being transformed into a finalized, evidence-backed graph.",
    )

    # A fragment updates only this status region. Five seconds leaves the page
    # responsive between durable coordinator queries while still feeling live.
    @st.fragment(run_every="5s")
    def run_progress() -> None:
        snapshot_key = f"last_live_snapshot_{initial.run_id}"
        status_available = True
        try:
            current = backend.status(initial.run_id, config_path)
        except Exception as exc:
            status_available = False
            cached = st.session_state.get(snapshot_key)
            current = cached if isinstance(cached, RunSnapshot) else initial
            st.warning(
                "Live status is temporarily unavailable. The last known state is shown; "
                "queued work and completed artifacts remain durable. "
                f"{concise_error(exc)}"
            )
        else:
            st.session_state[snapshot_key] = current
        if current.status.lower() not in _ACTIVE_STATUSES:
            st.rerun()
        render_run_snapshot(current)
        _render_live_consumption(current)
        recovery_notice_key = f"recovery_notice_{current.run_id}"
        if not _needs_recovery(current):
            st.session_state.pop(recovery_notice_key, None)
        command_columns = st.columns([1, 1, 4])
        if command_columns[0].button(
            "Cancel",
            icon=":material/stop_circle:",
            disabled=(not status_available or current.status.lower() not in _ACTIVE_STATUSES),
            help=(
                "Stop queued work and request cancellation of active tasks. Completed "
                "artifacts remain available for diagnostics and safe retries."
            ),
        ):
            try:
                with st.spinner("Cancelling run"):
                    backend.cancel(current.run_id, config_path)
            except Exception as exc:
                st.error(concise_error(exc))
            else:
                invalidate_run_history()
                st.rerun()
        if (
            status_available
            and _needs_recovery(current)
            and command_columns[1].button(
                "Recover",
                icon=":material/healing:",
                help=(
                    "Check the Kubernetes workers required by queued stages. Missing "
                    "prerequisites are reported; unavailable pools are restarted without "
                    "duplicating completed work or queued tasks."
                ),
                key=f"recover_run_{current.run_id}",
            )
        ):
            try:
                message = backend.recover(current.run_id, config_path)
            except Exception as exc:
                st.session_state[recovery_notice_key] = {
                    "level": "error",
                    "message": (f"Recovery could not proceed: {concise_error(exc)}"),
                }
            else:
                st.session_state[recovery_notice_key] = {
                    "level": "success",
                    "message": message,
                }
        command_columns[2].html(
            '<div class="fg-run-timestamp">'
            f"Run {current.run_id} · "
            f"{'refreshed' if status_available else 'status unavailable at'} "
            f"{datetime.now(UTC).strftime('%H:%M:%S UTC')}"
            "</div>"
        )
        recovery_notice = st.session_state.get(recovery_notice_key)
        if isinstance(recovery_notice, dict):
            notice_level = recovery_notice.get("level")
            notice_message = str(recovery_notice.get("message") or "")
            (st.error if notice_level == "error" else st.success)(notice_message)

    run_progress()


def _needs_recovery(snapshot: RunSnapshot) -> bool:
    """Return whether bounded diagnostics identify inactive queued work."""

    diagnostics = snapshot.raw.get("diagnostics")
    return (
        isinstance(diagnostics, dict)
        and diagnostics.get("state") == "attention_required"
        and snapshot.status.lower() in _ACTIVE_STATUSES
    )


def _render_terminal_run(
    backend: ControlPlaneBackend,
    snapshot: RunSnapshot,
    config_path: Path | None,
) -> None:
    """Explain terminal work and offer a safe retry for durable failed runs."""

    _render_graph_heading(
        backend,
        snapshot,
        "Run",
        "This run did not produce a finalized graph. Review the diagnostics below.",
    )
    render_run_snapshot(snapshot)
    _render_live_consumption(snapshot)
    _render_document_failures(backend, snapshot)
    technical_details = _technical_error_details(snapshot)
    if technical_details:
        with st.expander("Technical details"):
            st.code(technical_details, language="text")
    if snapshot.status.lower() != "failed" or "retry" not in backend.capabilities:
        return
    if st.button(
        "Retry failed work",
        icon=":material/refresh:",
        type="primary",
        help=(
            "Requeue only the tasks that exhausted their retry budget. Successful "
            "stages and immutable artifacts are preserved, so completed work is not repeated."
        ),
        key=f"retry_run_{snapshot.run_id}",
    ):
        try:
            backend.retry(snapshot.run_id, config_path)
        except Exception as exc:
            st.error(f"Retry could not be started: {exc}")
        else:
            invalidate_run_history()
            st.rerun()


def _render_live_consumption(snapshot: RunSnapshot) -> None:
    """Show what the run has spent so far, while it is still spending it.

    A finished graph publishes its full consumption beside the graph itself. The
    run that most needs watching is the one still going, and it is the only one
    where waiting for the total has a cost of its own.
    """

    totals = snapshot.raw.get("consumption")
    if not isinstance(totals, Mapping) or not totals:
        return
    calls = int(totals.get("calls") or 0)
    if not calls:
        return
    unpriced = int(totals.get("unpriced_calls") or 0)
    st.caption("So far this run")
    columns = st.columns(4)
    columns[0].metric("Tokens", f"{int(totals.get('total_tokens') or 0):,}")
    columns[1].metric("Pages parsed", f"{int(totals.get('pages') or 0):,}")
    columns[2].metric("Provider calls", f"{calls:,}")
    columns[3].metric(
        "Billed",
        f"${float(totals.get('billed_usd') or 0.0):,.2f}",
        help=(
            "Cost of the calls a rate covers. Totals grow while the run does and "
            "are republished in full beside the finished graph."
        ),
    )
    if unpriced:
        st.caption(
            f"{unpriced:,} of {calls:,} calls have no rate covering what they used, "
            "so the billed figure excludes them."
        )


def _render_document_failures(backend: ControlPlaneBackend, snapshot: RunSnapshot) -> None:
    """Show why documents failed, grouped by reason.

    A count alone leaves the reader unable to tell a corrupt document from a
    session that expired under the whole run, which are opposite problems: one
    needs the document replaced, the other only needs the work requeued.
    """

    reader = getattr(backend, "document_failures", None)
    if reader is None or not snapshot.documents_failed:
        return
    try:
        failures = reader(snapshot.run_id)
    except Exception as exc:
        st.caption(f"Failure reasons are unavailable: {concise_error(exc)}")
        return
    if not failures:
        return
    for message, documents in failures:
        label = "document" if documents == 1 else "documents"
        st.warning(f"**{documents} {label}** · {message}")


def _technical_error_details(snapshot: RunSnapshot) -> str | None:
    """Return the complete redacted worker failure for opt-in diagnostics.

    The Kubernetes backend keeps the bounded status payload in ``raw``. Only the
    concise cause is rendered by default; this helper preserves task identity,
    exception type, and the full remote traceback for troubleshooting without
    allowing it to dominate the normal failed-run workspace.
    """

    error = snapshot.raw.get("error")
    if not isinstance(error, Mapping):
        return None
    error_type = str(error.get("error_type") or error.get("type") or "Run failure")
    message = str(error.get("error_message") or error.get("message") or "").strip()
    task_id = str(error.get("task_id") or error.get("publication_id") or "").strip()
    if not message:
        return None
    heading = f"Task: {task_id}\n" if task_id else ""
    return f"{heading}Type: {error_type}\n\n{message}"


def _render_graph_heading(
    backend: ControlPlaneBackend,
    snapshot: RunSnapshot,
    kicker: str,
    description: str,
) -> None:
    """Render a graph title with an optional, backend-neutral rename action."""

    heading, action = st.columns([10, 2], vertical_alignment="top")
    with heading:
        page_heading(kicker, snapshot.display_name, description)
    if "rename" not in backend.capabilities:
        return
    with action:
        rename_panel = st.popover(
            "Rename",
            icon=":material/edit:",
            type="tertiary",
            help=(
                f"Rename {snapshot.display_name}. This changes only the display name; "
                f"the stable graph ID remains {snapshot.graph_id}."
            ),
        )
    with rename_panel:
        _render_graph_rename_form(backend, snapshot)


def _render_graph_rename_form(
    backend: ControlPlaneBackend,
    snapshot: RunSnapshot,
) -> None:
    """Render a compact rename form that preserves the graph's stable identity."""

    st.markdown("**Rename graph**")
    display_name = st.text_input(
        "Graph name",
        value=snapshot.display_name,
        max_chars=120,
        help="Use a concise name that makes this graph easy to recognize in the sidebar.",
        key=f"graph_name_input_{snapshot.graph_id}",
    )
    st.caption(f"Stable graph ID: {snapshot.graph_id}")
    if st.button(
        "Save",
        icon=":material/check:",
        type="primary",
        width="stretch",
        key=f"save_graph_name_{snapshot.graph_id}",
    ):
        try:
            backend.rename_graph(snapshot.graph_id, display_name)
        except Exception as exc:
            st.error(str(exc))
        else:
            invalidate_run_history()
            st.rerun()


def _config_path(snapshot: RunSnapshot) -> Path | None:
    """Recover the generated profile used for status, cancellation, and export."""

    value = str(snapshot.raw.get("config_path") or "")
    return Path(value) if value else None


def _requires_status_refresh(status: str) -> bool:
    """Return whether a listed run can still change without a fresh status read.

    Run listings are already authoritative for successful completion. Avoiding a
    second CLI or warehouse round trip makes completed graphs open immediately;
    detailed completion diagnostics remain available on demand, while failed and
    unfamiliar states still receive the full status contract automatically.
    """

    return status.lower() not in _SUCCESS_STATUSES


def _cache_graph_dataset(cache: object, key: str, dataset: object) -> None:
    """Store one graph while bounding the per-browser session memory footprint."""

    if not isinstance(cache, dict):
        return
    cache.pop(key, None)
    cache[key] = dataset
    while len(cache) > _RUN_GRAPH_CACHE_LIMIT:
        cache.pop(next(iter(cache)))


def _render_sharing(backend: ControlPlaneBackend, graph_id: str) -> None:
    """Show who can reach one graph, and let its owner change that.

    Access is decided by this application rather than by the warehouse, so the
    panel says so plainly: someone querying the tables directly is outside what
    these controls govern, and an operator deciding what to put in a graph
    deserves to know which of the two they are relying on.
    """

    if "share" not in backend.capabilities:
        return
    owner = backend.graph_owner(graph_id)
    viewer = backend.viewer()
    # Who else holds access is the owner's business. Shown to a grantee it
    # discloses colleagues and role names they were never granted sight of.
    owns = owner is not None and viewer.user_name == str(owner).upper()
    shares = backend.graph_shares(graph_id) if owns else []
    label = "Sharing" if owner else "Sharing · unowned"
    with st.expander(label, icon=":material/group:"):
        if owner is None:
            st.caption(
                "This graph was built before graphs recorded an owner, so everyone "
                "who can open this application can see it."
            )
        else:
            st.caption(
                f"Owned by **{owner}**. Private to its owner unless shared below. "
                "Access is enforced by this application, not by the database: "
                "anyone able to query the tables directly is not restricted by it."
            )
        if shares:
            for grantee_type, grantee in shares:
                row = st.columns([5, 1], vertical_alignment="center")
                row[0].markdown(f"{grantee_type.title()} · `{grantee}`")
                if row[1].button(
                    "",
                    icon=":material/close:",
                    key=f"unshare_{graph_id}_{grantee_type}_{grantee}",
                    help=f"Stop sharing this graph with {grantee}.",
                ):
                    backend.unshare_graph(graph_id, grantee_type, grantee)
                    st.rerun()
        elif owner is not None:
            st.caption("Not shared with anyone.")
        if owner is None or viewer.user_name != owner.upper():
            return
        with st.form(f"share_{graph_id}"):
            fields = st.columns([1, 3, 1], vertical_alignment="bottom")
            grantee_type = fields[0].selectbox(
                "Share with",
                ["User", "Role"],
                help="A Snowflake user, or every holder of a Snowflake role.",
            )
            grantee = fields[1].text_input(
                "Name",
                placeholder="colleague@example.com or ANALYSTS",
                help="The Snowflake username or role name to grant access to.",
            )
            if fields[2].form_submit_button("Share", type="primary") and grantee.strip():
                try:
                    backend.share_graph(graph_id, str(grantee_type), grantee)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.rerun()
