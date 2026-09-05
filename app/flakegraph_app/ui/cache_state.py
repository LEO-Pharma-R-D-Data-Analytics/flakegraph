"""Small session-state signals for invalidating cached control-plane reads.

Streamlit reruns the application script after most interactions. The UI caches
read-only control-plane snapshots to keep those reruns fast, while these generation
counters let write operations request an immediate refresh without coupling UI
modules to the cache implementation in the application entry point.
"""

from __future__ import annotations

import streamlit as st

FLEET_CACHE_GENERATION = "fleet_cache_generation"
RUN_HISTORY_CACHE_GENERATION = "run_history_cache_generation"


def cache_generation(key: str) -> int:
    """Return the current non-negative generation for one cached data family."""

    return int(st.session_state.get(key, 0))


def invalidate_run_history() -> None:
    """Make the next application rerun fetch fresh durable run history."""

    _advance(RUN_HISTORY_CACHE_GENERATION)


def invalidate_fleet_snapshot() -> None:
    """Make the next fleet-page rerun query Kubernetes instead of reusing its cache."""

    _advance(FLEET_CACHE_GENERATION)


def _advance(key: str) -> None:
    """Increment a cache generation using Streamlit's per-browser session state."""

    st.session_state[key] = cache_generation(key) + 1


# Account metadata: the databases, schemas, warehouses, roles, stages, compute
# pools and image repositories the deployment can address. Streamlit in Snowflake
# runs with owner's rights, so these SHOW statements execute as the application
# owner and return the same answer for every viewer — which is what makes one
# process-wide cache correct here. Nothing viewer-specific may be added: a graph
# list, an owner, a share or a viewer's roles cached without the viewer in the key
# would serve one person's data to another.
_METADATA_TTL_SECONDS = 300


@st.cache_data(ttl=_METADATA_TTL_SECONDS, max_entries=128, show_spinner=False)
def account_options(runtime_key: str, method: str, argument: str, *, _backend: object) -> list[str]:
    """Return one discovery listing, re-reading it only every few minutes.

    Discovery drives selectboxes that are rebuilt on every rerun, so without this
    the ingestion page pays a Snowflake round trip per control per keystroke.
    """

    call = getattr(_backend, method, None)
    if call is None:
        return []
    try:
        return [str(item) for item in (call(argument) if argument else call())]
    except Exception:
        # Discovery is a convenience; a role that cannot list warehouses must
        # still be able to type one rather than be blocked by an error.
        return []


@st.cache_data(ttl=_METADATA_TTL_SECONDS, max_entries=8, show_spinner=False)
def account_context(runtime_key: str, *, _backend: object) -> dict[str, str]:
    """Return the session's own account, database, schema, warehouse and role."""

    call = getattr(_backend, "current_context", None)
    if call is None:
        return {}
    try:
        return {str(key): str(value) for key, value in dict(call() or {}).items()}
    except Exception:
        return {}


@st.cache_data(ttl=_METADATA_TTL_SECONDS, max_entries=8, show_spinner=False)
def graph_embedding_width(runtime_key: str, *, _backend: object) -> int | None:
    """Return the vector width the destination's graph tables already store."""

    call = getattr(_backend, "graph_embedding_width", None)
    if call is None:
        return None
    try:
        width = call()
    except Exception:
        return None
    return int(width) if width else None
