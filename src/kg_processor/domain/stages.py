"""Serializable contracts exchanged between distributed pipeline stages.

Stage artifacts contain domain data, never provider clients, database handles, or
orchestrator state. The same models can therefore cross PostgreSQL, object storage,
or an in-process local executor without changing graph semantics.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    RelationObservation,
)
from kg_processor.domain.graph import Chunk


class PreparedDocumentShard(BaseModel):
    """Hold normalized document artifacts produced by one preparation task.

    A shard normally represents one source file. Lists remain part of the contract
    so local execution can use the identical stage with a small document batch.
    Chunk embeddings are intentionally absent until the extraction stage, allowing
    OCR workers to run without loading an embedding model.
    """

    file_ids: list[str]
    files_seen: int = Field(ge=0)
    documents_processed: int = Field(ge=0)
    document_rows: list[dict[str, Any]] = Field(default_factory=list)
    page_rows: list[dict[str, Any]] = Field(default_factory=list)
    block_rows: list[dict[str, Any]] = Field(default_factory=list)
    asset_rows: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    document_context_entities: list[EntityMention] = Field(default_factory=list)
    ocr_cache_hits: int = Field(default=0, ge=0)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class DocumentContextShard(BaseModel):
    """Carry one document's reusable context without repeated prepared content.

    Every extraction window needs focal document entities, but it does not need
    normalized pages, layout blocks, assets, or unrelated chunks. This compact
    contract prevents task fan-out from multiplying document transfer volume and
    retains the exact context trace for auditability.
    """

    file_ids: list[str]
    document_context_entities: list[EntityMention] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class ExtractionWindowShard(BaseModel):
    """Carry chunks for a bounded batch of independently extracted windows.

    The task's dependency on ``DocumentContextShard`` still enforces stage order.
    Separating bounded chunk bytes from reusable context avoids downloading an
    entire prepared document for every window in a long source file.
    """

    file_ids: list[str]
    chunks: list[Chunk]
    logical_window_count: int = Field(default=1, ge=1)


class EntityWindowShard(BaseModel):
    """Hold entity observations from one independently processed text window.

    Only chunk identities are retained because the immutable extraction-window
    artifact already owns source text. This keeps durable fan-out proportional to
    extracted observations instead of copying complete prepared-document payloads.
    """

    file_ids: list[str]
    chunk_ids: list[str]
    entities: list[EntityMention] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    logical_window_count: int = Field(default=1, ge=1)


class DocumentEntityInventoryShard(BaseModel):
    """Provide one deduplicated document-wide entity vocabulary to relation tasks.

    The inventory is the barrier between two parallel queue waves. Relation windows
    may therefore connect endpoints discovered anywhere in the same document while
    documents and windows remain independently schedulable across a worker fleet.
    """

    file_ids: list[str]
    entities: list[EntityMention] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    chunk_count: int = Field(default=0, ge=0)
    window_count: int = Field(default=0, ge=0)


class RelationWindowShard(BaseModel):
    """Hold relation observations from one document-inventory-aware window.

    Entity observations are deliberately omitted: every relation task reads the
    same immutable inventory, and document compaction persists that inventory once.
    """

    file_ids: list[str]
    chunk_ids: list[str]
    relations: list[RelationObservation] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    logical_window_count: int = Field(default=1, ge=1)


class ExtractedDocumentShard(BaseModel):
    """Hold one prepared shard and its unresolved grounded observations.

    Keeping mention identity unresolved is essential: workers may extract separate
    documents concurrently, but graph-wide resolution must see every mention before
    choosing canonical entities. Local execution nests the complete prepared shard.
    Spark execution stores only its file-level projection because the scalable
    finalizer reads run-scoped prepared artifacts directly, avoiding a second copy
    of every OCR page, block, asset, and chunk in object storage.
    """

    prepared: PreparedDocumentShard
    observations: ExtractionObservations
    trace: list[dict[str, Any]] = Field(default_factory=list)


def combine_extracted_shards(shards: list[ExtractedDocumentShard]) -> ExtractionObservations:
    """Combine worker outputs deterministically before graph-wide resolution.

    Sorting by each shard's first file id removes task completion order from graph
    identity, trace ordering, and persisted output. Duplicate mention and relation
    observations are conservatively removed by the resolution stage itself.
    """

    ordered = sorted(shards, key=_shard_order_key)
    return ExtractionObservations(
        entities=[
            entity
            for shard in ordered
            for entity in [
                *shard.prepared.document_context_entities,
                *shard.observations.entities,
            ]
        ],
        relations=[relation for shard in ordered for relation in shard.observations.relations],
        trace=[event for shard in ordered for event in shard.observations.trace],
        chunk_count=sum(shard.observations.chunk_count for shard in ordered),
        window_count=sum(shard.observations.window_count for shard in ordered),
    )


def _shard_order_key(shard: ExtractedDocumentShard) -> tuple[str, ...]:
    """Return a stable ordering key even for a defensive empty-file shard."""

    return tuple(sorted(shard.prepared.file_ids))
