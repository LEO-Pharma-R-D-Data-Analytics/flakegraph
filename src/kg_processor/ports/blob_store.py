"""Immutable blob payload storage used by distributed artifact metadata."""

from __future__ import annotations

from typing import BinaryIO, Protocol, runtime_checkable


class BlobStore(Protocol):
    """Store large immutable bytes outside the transactional task database.

    PostgreSQL remains authoritative for artifact identity, checksums, task
    ownership, and publication state. Implementations provide only the scalable
    byte plane so workers and distributed data engines can read objects directly.
    """

    def put(self, key: str, payload: bytes, media_type: str) -> str:
        """Store bytes idempotently and return their canonical object URI."""
        ...

    def get(self, uri: str) -> bytes:
        """Return exact bytes for an object previously written by this store."""
        ...

    def delete(self, uri: str) -> None:
        """Delete one object idempotently during explicit run retention cleanup."""
        ...


@runtime_checkable
class BatchBlobDeleter(Protocol):
    """Optional capability for deleting many immutable payloads efficiently.

    Object stores commonly expose a bounded bulk-delete operation. Retention can
    use this capability without making it mandatory for simple third-party blob
    adapters, which continue to work through repeated idempotent ``delete`` calls.
    """

    def delete_many(self, uris: list[str]) -> None:
        """Delete every selected object or raise without suppressing store errors."""
        ...


@runtime_checkable
class BlobDownloader(Protocol):
    """Optional streaming download capability for bounded consumers."""

    def download_to(self, uri: str, destination: BinaryIO) -> None:
        """Stream an object into an open binary destination without collecting it."""
        ...
