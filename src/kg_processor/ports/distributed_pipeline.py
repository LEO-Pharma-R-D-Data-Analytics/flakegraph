"""Application-stage port consumed by the durable distributed worker."""

from __future__ import annotations

from typing import Protocol

from kg_processor.domain.documents import InputFile
from kg_processor.domain.extraction import EntityMention
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.stages import (
    EntityWindowShard,
    ExtractedDocumentShard,
    PreparedDocumentShard,
    RelationWindowShard,
)


class DistributedPipeline(Protocol):
    """Expose only the coarse serializable stages a distributed worker may execute."""

    def prepare_documents(self, files: list[InputFile]) -> PreparedDocumentShard:
        """OCR, normalize, and chunk an explicit source selection."""
        ...

    def extract_window_entities(
        self,
        prepared: PreparedDocumentShard,
    ) -> EntityWindowShard:
        """Extract unresolved entity mentions from one bounded chunk window."""
        ...

    def extract_window_relations(
        self,
        prepared: PreparedDocumentShard,
        document_entities: list[EntityMention],
    ) -> RelationWindowShard:
        """Extract grounded relations against a document-wide entity inventory."""
        ...

    def extract_document_context(
        self,
        prepared: PreparedDocumentShard,
    ) -> PreparedDocumentShard:
        """Extract reusable document-context mentions from bounded front matter."""
        ...

    def finalize_document_shards(
        self,
        shards: list[ExtractedDocumentShard],
        *,
        write: bool = True,
    ) -> GraphWriteBatch:
        """Resolve the complete corpus, validate it, and optionally publish it."""
        ...
