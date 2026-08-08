"""FlakeGraph Streamlit entry point for local, Kubernetes, and Snowflake use."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any, Protocol, cast

import streamlit as st
from flakegraph_app.backends import build_backend
from flakegraph_app.backends.base import ControlPlaneBackend
from flakegraph_app.backends.factory import active_snowflake_session
from flakegraph_app.models import ClusterSnapshot, RunSnapshot, RuntimeMode
from flakegraph_app.ui.cache_state import (
    FLEET_CACHE_GENERATION,
    RUN_HISTORY_CACHE_GENERATION,
    cache_generation,
)
from flakegraph_app.ui.navigation import render_run_navigation
from flakegraph_app.ui.shared import concise_error
from flakegraph_app.ui.theme import apply_theme

APPLICATION_ROOT = Path(__file__).resolve().parent
# Local development keeps the app inside the repository, while Snowflake stages
# the app and its required configuration files together at one deployment root.
REPOSITORY_ROOT = (
    APPLICATION_ROOT.parent
    if (APPLICATION_ROOT.parent / "configs" / "app-defaults.yaml").is_file()
    else APPLICATION_ROOT
)
SIDEBAR_LOGO = APPLICATION_ROOT / "assets" / "flakegraph-logo.png"


class _SessionState(Protocol):
    """Minimal session-state contract needed to namespace shared cache entries."""

    def get(self, key: str, default: Any = None, /) -> Any:
        """Return a stored value or its default."""
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        """Store one session-scoped value."""
        ...


def _cached_runs(
    session_cache_token: str,
    runtime_key: str,
    generation: int,
    limit: int,
    *,
    _backend: ControlPlaneBackend,
) -> list[RunSnapshot]:
    """Read durable run history, every time.

    Deliberately uncached. Run state is the one thing on the page that changes
    without the reader touching anything, and a run whose worker died looked
    alive for as long as a cached answer survived. The saving was never worth
    showing somebody a status that had stopped being true.
    """

    del session_cache_token, runtime_key, generation
    return list(_backend.list_runs(limit=limit))


def _cached_cluster(
    session_cache_token: str,
    namespace: str,
    generation: int,
    *,
    _backend: ControlPlaneBackend,
) -> ClusterSnapshot | None:
    """Cache a Kubernetes inventory only within the requesting UI session."""

    del session_cache_token, generation
    return _backend.cluster(namespace)


def _load_runs(
    backend: ControlPlaneBackend,
    runtime: RuntimeMode,
) -> tuple[list[RunSnapshot], str | None]:
    """Return recent run history and a displayable error without blocking the app."""

    try:
        runs = _cached_runs(
            _session_cache_token(cast(_SessionState, st.session_state)),
            runtime.value.lower(),
            cache_generation(RUN_HISTORY_CACHE_GENERATION),
            100,
            _backend=backend,
        )
    except Exception as exc:
        return [], concise_error(exc)
    return runs, None


def _load_cluster(
    backend: ControlPlaneBackend,
    runtime: RuntimeMode,
) -> tuple[str, ClusterSnapshot | None, str | None]:
    """Load Kubernetes inventory only when its page needs a live snapshot."""

    namespace = os.environ.get("FLAKEGRAPH_APP_KUBERNETES_NAMESPACE", "flakegraph")
    if runtime != RuntimeMode.KUBERNETES:
        return namespace, None, None

    # The selected cluster owns the namespace; fleet reads are confined to it.
    namespace = str(getattr(backend, "namespace", namespace) or namespace)
    runtime_key = runtime.value.lower()
    runtime_changed = st.session_state.get("navigation_runtime") != runtime_key
    requested_page = "fleet" if runtime_changed else st.session_state.get("active_page")
    snapshot_key = f"last_fleet_snapshot_{namespace}"
    error_key = f"last_fleet_error_{namespace}"
    if requested_page != "fleet":
        # Other pages need only the last fleet total for the sidebar label.
        # Avoiding a live kubectl query here removes a full second from graph
        # filtering, navigation actions, and ingestion form interactions.
        return (
            namespace,
            st.session_state.get(snapshot_key),
            st.session_state.get(error_key),
        )
    try:
        cluster = _cached_cluster(
            _session_cache_token(cast(_SessionState, st.session_state)),
            namespace,
            cache_generation(FLEET_CACHE_GENERATION),
            _backend=backend,
        )
    except Exception as exc:
        error = str(exc)
        st.session_state[error_key] = error
        return namespace, None, error
    st.session_state[snapshot_key] = cluster
    st.session_state[error_key] = None
    return namespace, cluster, None


@st.fragment
def _render_ingestion_fragment(
    backend: ControlPlaneBackend,
    runtime: RuntimeMode,
) -> None:
    """Keep ingestion form edits local to their workspace fragment.

    Ordinary provider and source selections no longer rebuild sidebar history or
    query a remote control plane. The ingestion module still requests a full app
    rerun after submission so the newly created graph becomes the active page.
    """

    from flakegraph_app.ui.ingestion import render_ingestion  # noqa: PLC0415

    render_ingestion(backend, runtime, REPOSITORY_ROOT)


def _render_fleet_page(
    backend: ControlPlaneBackend,
    namespace: str,
    cluster: ClusterSnapshot | None,
    cluster_error: str | None,
) -> None:
    """Load Kubernetes presentation code only when the fleet page is active."""

    from flakegraph_app.ui.fleet import render_fleet  # noqa: PLC0415

    render_fleet(backend, namespace, cluster, cluster_error)


def _render_cluster_page(backend: ControlPlaneBackend) -> None:
    """Manage registered Kubernetes clusters without leaving the application."""

    from flakegraph_app.ui.clusters import render_cluster_manager  # noqa: PLC0415

    render_cluster_manager(_app_state_root(backend))


def _app_state_root(backend: ControlPlaneBackend) -> Path:
    """Return the app-owned catalog directory, whichever backend is active."""

    state_root = getattr(backend, "state_root", None)
    if isinstance(state_root, Path):
        return state_root
    return REPOSITORY_ROOT / ".flakegraph" / "app"


def _render_run_page(backend: ControlPlaneBackend, selected_run: RunSnapshot) -> None:
    """Load graph inspection dependencies only when a run workspace is active."""

    from flakegraph_app.ui.run_workspace import render_run_workspace  # noqa: PLC0415

    render_run_workspace(backend, selected_run)


def _session_cache_token(state: _SessionState) -> str:
    """Return one unguessable cache namespace that remains stable for a session."""

    key = "_flakegraph_cache_session_token"
    current = state.get(key)
    if isinstance(current, str) and current:
        return current
    token = secrets.token_hex(16)
    state[key] = token
    return token


def main() -> None:
    """Configure navigation once and delegate each view to a focused UI module."""

    st.set_page_config(
        page_title="FlakeGraph",
        page_icon=":material/hub:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    apply_theme()
    snowflake_session = active_snowflake_session()
    default_runtime = RuntimeMode.SNOWFLAKE if snowflake_session is not None else RuntimeMode.LOCAL
    # Deployed inside Snowflake, only the Snowflake runtime can work: the local
    # and Kubernetes runtimes drive work by running CLI processes against a
    # repository checkout and a kubeconfig, none of which exist in Streamlit in
    # Snowflake. Offering them there would present choices that fail on use.
    available_runtimes = (
        [RuntimeMode.SNOWFLAKE] if snowflake_session is not None else list(RuntimeMode)
    )
    with st.sidebar:
        # Reuse the repository wordmark so the application and public README
        # always present the same product identity and supporting tagline.
        st.image(SIDEBAR_LOGO, width="stretch")
        selected_runtime = st.selectbox(
            "Runtime",
            available_runtimes,
            index=available_runtimes.index(default_runtime),
            format_func=lambda item: item.value,
            key="runtime_selector",
            help=(
                "Choose where FlakeGraph coordinates processing. Provider and graph "
                "contracts remain the same across local, Kubernetes, and Snowflake runs. "
                "Deployed in Snowflake, only the Snowflake runtime is available."
            ),
        )
        runtime = selected_runtime or default_runtime
    try:
        backend = build_backend(runtime, REPOSITORY_ROOT, snowflake_session)
    except Exception as exc:
        st.error(str(exc))
        return
    runs, run_list_error = _load_runs(backend, runtime)
    namespace, cluster, cluster_error = _load_cluster(backend, runtime)
    with st.sidebar:
        page, selected_run = render_run_navigation(
            backend,
            runtime,
            runs,
            cluster=cluster,
            cluster_error=cluster_error,
        )
        if run_list_error:
            st.warning(f"Run history is unavailable: {run_list_error}")
        st.divider()
        st.caption(
            "The runtime changes orchestration and storage. OCR, LLM, embedding, "
            "and graph contracts remain the same."
        )
    if page == "clusters":
        _render_cluster_page(backend)
    elif page == "fleet":
        _render_fleet_page(backend, namespace, cluster, cluster_error)
    elif page == "new" or selected_run is None:
        _render_ingestion_fragment(backend, runtime)
    else:
        _render_run_page(backend, selected_run)


if __name__ == "__main__":
    main()
