"""Control-plane interface implemented by each supported runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from flakegraph_app.models import (
    ClusterSnapshot,
    GraphDataset,
    IngestionRequest,
    NodeWorkAssignment,
    RunSnapshot,
    SourceObject,
)
from flakegraph_app.viewer import Viewer


class ControlPlaneBackend(Protocol):
    """Operations the Streamlit UI may perform without runtime-specific branching."""

    @property
    def capabilities(self) -> frozenset[str]:
        """Return feature flags used to show only supported controls."""
        ...

    def list_source_objects(
        self,
        source: Mapping[str, str],
        *,
        limit: int = 1_000,
    ) -> Sequence[SourceObject]:
        """List selectable source objects without starting processing."""
        ...

    def viewer(self) -> Viewer:
        """Return the person looking at the application."""
        ...

    def graph_row_counts(self, graph_id: str) -> dict[str, int]:
        """Return how many rows each table still holds for one graph."""
        ...

    def delete_graph(self, graph_id: str) -> dict[str, int]:
        """Remove a graph and everything recorded about it."""
        ...

    def graph_owner(self, graph_id: str) -> str | None:
        """Return who owns one graph, where the runtime records ownership."""
        ...

    def graph_shares(self, graph_id: str) -> list[tuple[str, str]]:
        """Return the grantee type and name of every share on one graph."""
        ...

    def share_graph(self, graph_id: str, grantee_type: str, grantee: str) -> None:
        """Grant one user or role access to a graph."""
        ...

    def unshare_graph(self, graph_id: str, grantee_type: str, grantee: str) -> None:
        """Withdraw one share from a graph."""
        ...

    def graph_embedding_width(self) -> int | None:
        """Return the vector width the destination's graph tables already store.

        ``None`` means the destination does not fix a width, so any model fits.
        """

        return None

    def preflight(self, request: IngestionRequest) -> Mapping[str, object]:
        """Validate one request before it can consume provider resources."""
        ...

    def submit(self, request: IngestionRequest) -> RunSnapshot:
        """Submit one graph run and return its initial status."""
        ...

    def list_runs(self, *, limit: int = 100) -> Sequence[RunSnapshot]:
        """Return recent runs visible to the selected runtime."""
        ...

    def status(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Return the latest bounded progress view for one run."""
        ...

    def cancel(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Request cancellation and return the updated state."""
        ...

    def retry(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
        """Retry only terminally failed work while preserving completed artifacts."""
        ...

    def recover(self, run_id: str, config_path: Path | None = None) -> str:
        """Reconcile infrastructure for durable queued work and describe the action."""
        ...

    def forget(self, run_id: str, config_path: Path | None = None) -> None:
        """Remove one terminal run from app history without deleting graph data."""
        ...

    def rename_graph(self, graph_id: str, display_name: str) -> str:
        """Change a graph's display name without changing its stable identity."""
        ...

    def cluster(self, namespace: str) -> ClusterSnapshot | None:
        """Return infrastructure telemetry when this backend owns a cluster."""
        ...

    def node_assignments(
        self,
        namespace: str,
        node_name: str,
    ) -> Sequence[NodeWorkAssignment]:
        """Return bounded active task assignments for one node when supported."""
        ...

    def load_graph(self, location: str, graph_id: str | None = None) -> GraphDataset:
        """Load a completed graph for interactive inspection."""
        ...

    def load_run_graph(
        self,
        snapshot: RunSnapshot,
        config_path: Path | None = None,
    ) -> GraphDataset:
        """Materialize and load the graph produced by one completed run."""
        ...


class GraphAccessDefaults:
    """Graph ownership behaviour for runtimes that do not record it.

    A local or fleet run belongs to whoever is at the machine, so there is no
    second person to keep a graph from. Sharing is therefore a Snowflake-only
    control, and these defaults keep the other runtimes honest about that rather
    than pretending to enforce something.
    """

    def viewer(self) -> Viewer:
        """Return the person looking at the application.

        Runtimes that serve a single local operator report an unidentified
        viewer, which owns nothing and is shown only unowned graphs.
        """

        return Viewer(user_name="")

    def graph_row_counts(self, graph_id: str) -> dict[str, int]:
        """Report nothing to remove for runtimes that do not store graphs in tables."""

        del graph_id
        return {}

    def delete_graph(self, graph_id: str) -> dict[str, int]:
        """Refuse deletion where the runtime does not own the stored graph."""

        del graph_id
        raise NotImplementedError("This runtime does not delete stored graphs")

    def graph_owner(self, graph_id: str) -> str | None:
        """Return who owns one graph, where the runtime records ownership."""

        del graph_id
        return None

    def graph_shares(self, graph_id: str) -> list[tuple[str, str]]:
        """Return the grantee type and name of every share on one graph."""

        del graph_id
        return []

    def share_graph(self, graph_id: str, grantee_type: str, grantee: str) -> None:
        """Grant one user or role access to a graph."""

        raise NotImplementedError("This runtime does not share graphs")

    def unshare_graph(self, graph_id: str, grantee_type: str, grantee: str) -> None:
        """Withdraw one share from a graph."""

        raise NotImplementedError("This runtime does not share graphs")

