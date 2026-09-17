"""Immutable blob payload storage used by distributed artifact metadata."""

from __future__ import annotations

from typing import BinaryIO, Protocol


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

    def download_to(self, uri: str, destination: BinaryIO) -> None:
        """Stream an object into an open binary destination without collecting it."""
        ...
