"""Immutable artifact-store port for distributed stage handoffs."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kg_processor.domain.distributed import ArtifactKind, ArtifactRef, StoredArtifact


class ArtifactStore(Protocol):
    """Persist content-addressed payloads independently from task orchestration."""

    def put(
        self,
        run_id: str,
        kind: ArtifactKind,
        payload: bytes,
        media_type: str,
        metadata: dict[str, Any] | None = None,
        identity_key: str | None = None,
    ) -> ArtifactRef:
        """Store bytes idempotently and return their immutable artifact reference.

        ``identity_key`` distinguishes equal bytes that represent separate source
        records with different provenance. Derived artifacts normally omit it and
        remain content-addressed.
        """
        ...

    def get(self, artifact_id: str) -> StoredArtifact:
        """Load and checksum-verify an artifact or raise when absent or corrupt."""
        ...

    def delete_run_artifacts(self, run_id: str) -> int:
        """Delete intermediate artifacts for an explicitly selected completed run."""
        ...


@runtime_checkable
class BatchArtifactReader(Protocol):
    """Optional capability for resolving several immutable artifacts efficiently.

    Distributed compaction commonly consumes all extraction windows for one
    document. Stores backed by a database plus object storage can collapse those
    metadata lookups into one query and overlap independent object reads. The
    separate capability keeps simple third-party artifact adapters valid; workers
    transparently fall back to repeated ``ArtifactStore.get`` calls.
    """

    def get_many(self, artifact_ids: list[str]) -> list[StoredArtifact]:
        """Load artifacts in caller order or raise if any item is missing or corrupt."""
        ...


@runtime_checkable
class RunArtifactReader(Protocol):
    """Optional capability for discovering artifacts by run and semantic kind.

    Finalization is a run-wide operation rather than a task with hundreds of
    thousands of explicit dependency edges. A store that supports local distributed
    finalization therefore exposes immutable stage outputs directly by their run
    ownership and kind.
    """

    def get_run_artifacts(
        self,
        run_id: str,
        kinds: set[ArtifactKind],
    ) -> list[StoredArtifact]:
        """Return matching artifacts in stable creation and identity order."""
        ...
