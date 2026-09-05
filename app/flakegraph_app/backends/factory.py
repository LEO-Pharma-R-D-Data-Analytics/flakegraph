"""Runtime backend selection for the Streamlit entry point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from flakegraph_app.backends.base import ControlPlaneBackend
from flakegraph_app.models import RuntimeMode

if TYPE_CHECKING:
    from snowflake.snowpark import Session


def build_backend(
    mode: RuntimeMode,
    repository_root: Path,
    snowflake_session: Any | None = None,
) -> ControlPlaneBackend:
    """Construct only the selected runtime backend.

    Local and Kubernetes adapters import the Python 3.14 processing package and
    optional clients unavailable in Snowflake's staged Python 3.11 environment.
    Runtime-local imports keep those modules outside the Snowflake import graph.
    """

    if mode == RuntimeMode.LOCAL:
        from flakegraph_app.backends.local import LocalBackend  # noqa: PLC0415

        return LocalBackend(repository_root)
    if mode == RuntimeMode.KUBERNETES:
        from flakegraph_app.backends.kubernetes import KubernetesBackend  # noqa: PLC0415

        return KubernetesBackend(repository_root)
    from flakegraph_app.backends.snowflake import SnowflakeBackend  # noqa: PLC0415

    session = snowflake_session if snowflake_session is not None else active_snowflake_session()
    if session is None:
        raise RuntimeError(
            "Snowflake mode requires an active Snowpark session. Deploy the app into "
            "Snowflake or select Local/Kubernetes mode."
        )
    return SnowflakeBackend(session)


def active_snowflake_session() -> Session | None:
    """Return Snowflake's active session without importing an app backend.

    Importing the small Snowpark context API first allows the staged entry point
    to detect its host before loading any runtime implementation. Local launches
    intentionally treat a missing Snowpark package or active session as absence.
    """

    try:
        from snowflake.snowpark.context import get_active_session  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return get_active_session()
    except Exception:
        return None
