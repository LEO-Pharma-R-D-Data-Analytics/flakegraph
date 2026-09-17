"""FlakeGraph Streamlit entry point for local, Kubernetes, and Snowflake use."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from flakegraph_app.backends import build_backend
from flakegraph_app.backends.base import ControlPlaneBackend, app_state_root
from flakegraph_app.backends.factory import active_snowflake_session
from flakegraph_app.models import ClusterSnapshot, RunSnapshot, RuntimeMode
from flakegraph_app.ui.authentication import (
    AuthenticationNotConfigured,
    render_sign_out,
    require_sign_in,
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
_DEFAULT_RUNTIME_ENV = "FLAKEGRAPH_APP_DEFAULT_RUNTIME"
_REQUIRE_SIGN_IN_ENV = "FLAKEGRAPH_APP_REQUIRE_SIGN_IN"


def _configured_default_runtime() -> RuntimeMode:
    """Return the runtime this deployment should open on.

    A control plane installed beside a fleet is not somebody's laptop: its graphs
    live in the fleet's coordination store, and defaulting to the local runtime
    shows an operator a history of whatever that one host happened to run before
    they have chosen anything. The choice stays a choice — this only decides
    which one is already selected. Snowflake is not offered here because it is
    forced by being deployed inside Snowflake, never by configuration.
    """

    requested = os.environ.get(_DEFAULT_RUNTIME_ENV, "").strip().casefold()
    return next(
        (
            mode
            for mode in (RuntimeMode.LOCAL, RuntimeMode.KUBERNETES)
            if mode.value.casefold() == requested
        ),
        RuntimeMode.LOCAL,
    )


def _load_runs(backend: ControlPlaneBackend) -> tuple[list[RunSnapshot], str | None]:
    """Return recent run history and a displayable error without blocking the app."""

    # Deliberately uncached. Run state is the one thing on the page that changes
    # without the reader touching anything, and a run whose worker died looked
    # alive for as long as a cached answer survived.
    try:
        runs = list(backend.list_runs(limit=100))
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
        cluster = backend.cluster(namespace)
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

    render_cluster_manager(app_state_root(backend, REPOSITORY_ROOT))


def _render_run_page(backend: ControlPlaneBackend, selected_run: RunSnapshot) -> None:
    """Load graph inspection dependencies only when a run workspace is active."""

    from flakegraph_app.ui.run_workspace import render_run_workspace  # noqa: PLC0415

    render_run_workspace(backend, selected_run)


def main() -> None:
    """Configure navigation once and delegate each view to a focused UI module."""

    st.set_page_config(
        page_title="FlakeGraph",
        page_icon=":material/hub:",
        layout="wide",
        initial_sidebar_state="auto",
    )
    apply_theme()
    # Snowflake authenticates the viewer before the app is reached, so asking
    # again there would be a second sign-in for an identity we already hold.
    snowflake_session = active_snowflake_session()
    if snowflake_session is None:
        try:
            if not require_sign_in(required=_sign_in_is_required()):
                return
        except AuthenticationNotConfigured as error:
            st.error(concise_error(error))
            return
        render_sign_out()
    default_runtime = (
        RuntimeMode.SNOWFLAKE if snowflake_session is not None else _configured_default_runtime()
    )
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
    runs, run_list_error = _load_runs(backend)
    listing_warning = str(getattr(backend, "listing_warning", "") or "")
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
        elif listing_warning:
            # Reported only by runtimes whose listing can degrade to a lesser
            # source; the history is shown, with what it is missing.
            st.warning(listing_warning)
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


def _sign_in_is_required() -> bool:
    """Report whether this deployment refuses to serve an anonymous viewer.

    Off by default so a local checkout keeps working without an issuer, and set
    by the chart wherever the application is reachable by more than its
    operator.
    """

    return os.environ.get(_REQUIRE_SIGN_IN_ENV, "").strip().lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    main()
