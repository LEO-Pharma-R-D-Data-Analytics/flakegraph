"""File-source port.

Every input backend exposes local-openable files plus the original source URI.
That split lets local OCR engines read bytes from disk while Snowflake/Cortex can
still preserve stage/blob provenance for auditing and reindexing.
"""

from __future__ import annotations

from typing import Protocol

from kg_processor.domain.documents import InputFile


class FileSource(Protocol):
    """Lists normalized input files for the pipeline to process."""

    def list_files(self) -> list[InputFile]:
        """Discover source documents and normalize them into input-file records."""
        ...
