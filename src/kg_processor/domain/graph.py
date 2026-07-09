"""Canonical graph rows and extraction observations.

These models are deliberately close to the persisted local/Snowflake artifacts:
stable ids and provenance fields live here so provider adapters cannot invent a
different graph contract. The pipeline first works with observations, then
promotes grounded/merged records into canonical nodes, edges, evidence, and
community rows.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    """A stable text slice that becomes the unit of LLM graph extraction."""

    id: str
    graph_id: str = ""
    file_id: str
    document_id: str = ""
    page_number: int
    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    token_count: int
    content_hash: str
    section_path: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    ocr_generation_id: str | None = None
    embedding: list[float] | None = None

    @model_validator(mode="after")
    def default_document_id(self) -> Self:
        """Use the file id as the document id for providers without a document id."""

        if not self.document_id:
            self.document_id = self.file_id
        return self


class ExtractedEntity(BaseModel):
    """One entity observation returned by an LLM for a specific chunk."""

    name: str
    type: str
    description: str
    source_chunk_id: str
    quote: str | None = None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)


class ExtractedRelation(BaseModel):
    """One relationship observation returned by an LLM for a specific chunk."""

    source_name: str
    target_name: str
    relation_type: str
    description: str
    source_chunk_id: str
    quote: str | None = None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    weight: float = Field(default=1.0, ge=0.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Structured LLM extraction output before filtering and graph assembly."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """Canonical merged entity node persisted to local artifacts or Snowflake."""

    id: str
    graph_id: str
    normalized_name: str
    name: str
    primary_type: str
    types: list[str]
    description: str
    embedding: list[float] | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    degree: int = 0
    rank: float | None = None


class GraphEdge(BaseModel):
    """Canonical relationship edge between two merged graph nodes."""

    id: str
    graph_id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    description: str
    weight: float
    source_file_id: str
    source_chunk_ids: list[str]
    embedding: list[float] | None = None


class Evidence(BaseModel):
    """Grounded text span supporting a node, edge, or community finding."""

    id: str
    graph_id: str
    subject_id: str
    subject_kind: str
    file_id: str
    chunk_id: str
    page_number: int
    start_offset: int
    end_offset: int
    quote: str


class EntitySource(BaseModel):
    """Per-file explanation of why a merged entity appears in the graph."""

    id: str
    graph_id: str
    node_id: str
    file_id: str
    per_file_description: str
    mention_count: int


class Community(BaseModel):
    """Detected group of related nodes with optional LLM-generated narrative."""

    id: str
    graph_id: str
    stable_key: str
    level: int
    title: str
    summary: str
    rating: float
    rating_explanation: str = ""
    member_node_ids: list[str]
    suggested_questions: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None


class CommunityFinding(BaseModel):
    """A claim or takeaway associated with a generated community report."""

    id: str
    graph_id: str = ""
    community_id: str
    summary: str
    explanation: str


class GraphWriteBatch(BaseModel):
    """Complete, writer-ready graph payload for one snapshot or file batch."""

    graph_id: str
    write_scope: Literal["graph_snapshot", "file_batch"] = "graph_snapshot"
    reindex_file_ids: list[str] = Field(default_factory=list)
    documents: list[dict[str, Any]]
    pages: list[dict[str, Any]]
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    assets: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[Chunk]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    evidence: list[Evidence]
    entity_sources: list[EntitySource]
    communities: list[Community]
    community_findings: list[CommunityFinding]
    run_report: dict[str, Any]
    extraction_trace: list[dict[str, Any]] = Field(default_factory=list)
    graph_metrics: dict[str, Any] = Field(default_factory=dict)
