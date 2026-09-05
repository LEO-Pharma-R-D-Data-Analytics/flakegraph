"""File-source port.

Every input backend exposes local-openable files plus the original source URI.
That split lets local OCR engines read bytes from disk while Snowflake/Cortex can
still preserve stage/blob provenance for auditing and reindexing.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from kg_processor.domain.documents import InputFile


class FileSource(Protocol):
    """Lists normalized input files for the pipeline to process."""

    def list_files(self) -> list[InputFile]:
        """Discover source documents and normalize them into input-file records."""
        ...


@runtime_checkable
class IterableFileSource(Protocol):
    """Optional source capability for bounded-memory corpus discovery."""

    def iter_files(self) -> Iterator[InputFile]:
        """Yield deterministically ordered input records without retaining them all."""
        ...
