"""Port for publishing a partitioned graph to a configured durable destination."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from kg_processor.domain.finalization import GraphDatasetManifest


class GraphManifestPublisher(Protocol):
    """Publish one Spark manifest without exposing a concrete writer to application code."""

    def publish(
        self,
        manifest: GraphDatasetManifest,
        task_payload: Mapping[str, object],
    ) -> None:
        """Persist a manifest according to its final task's non-secret output contract."""
        ...
