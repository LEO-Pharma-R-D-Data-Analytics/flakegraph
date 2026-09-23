# SPDX-License-Identifier: Apache-2.0
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

    def prepare_documents(
        self,
        files: list[InputFile],
        *,
        record_failures: bool = True,
    ) -> PreparedDocumentShard:
        """OCR, normalize, and chunk an explicit source selection.

        With ``record_failures`` a document that cannot be read is recorded in
        the shard's trace and skipped; without it the failure is raised, for a
        caller that still has attempts left to retry it.
        """
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
