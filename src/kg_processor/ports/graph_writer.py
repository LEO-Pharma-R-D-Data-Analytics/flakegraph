"""Writer port for persisting a complete graph write batch."""

from __future__ import annotations

from typing import Protocol

from kg_processor.domain.graph import GraphWriteBatch


class GraphWriter(Protocol):
    """Persists a complete graph batch to a concrete storage backend."""

    def write(self, batch: GraphWriteBatch) -> None:
        """Persist the supplied graph batch atomically for the chosen backend."""
        ...
