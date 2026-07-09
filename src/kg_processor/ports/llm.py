"""LLM provider port for extraction, repair, descriptions, and communities.

LLM adapters submit prompts and normalize structured responses; they do not own
GraphRAG behavior. Provider metadata is returned with every LLM-assisted step so
traces can explain which model/prompt revision produced each graph artifact.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from kg_processor.domain.graph import Chunk, ExtractionResult


class LlmOptions(BaseModel):
    """Provider-neutral options that shape graph extraction prompts and limits."""

    model: str
    entity_types: list[str]
    relation_types: list[str] | None = None
    max_entities_per_batch: int = 100
    max_relations_per_batch: int = 100
    min_entity_confidence: float = 0.0
    min_relation_confidence: float = 0.0
    temperature: float = 0.0
    timeout_seconds: int = 120
    extraction_pass: int = 0
    previous_result: ExtractionResult | None = None


class CommunitySummaryRequest(BaseModel):
    """Inputs used by an LLM to turn a detected community into report text."""

    title_seed: str
    members: list[str]
    relations: list[str]


class CommunitySummaryResult(BaseModel):
    """Normalized community summary returned by an LLM provider."""

    title: str
    summary: str
    rating: float = Field(ge=0.0, le=10.0)
    rating_explanation: str = ""
    findings: list[tuple[str, str]]
    suggested_questions: list[str] = Field(default_factory=list)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class DescriptionMergeRequest(BaseModel):
    """Inputs used to merge several entity observations into one description."""

    entity_name: str
    entity_type: str
    descriptions: list[str]
    evidence: list[str]


class DescriptionMergeResult(BaseModel):
    """Normalized merged description plus provider trace metadata."""

    description: str
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class GraphRepairRequest(BaseModel):
    """Context sent to an LLM when the first extraction response fails validation."""

    chunks: list[Chunk]
    options: LlmOptions
    invalid_result: ExtractionResult
    validation_error: str


class LlmProvider(Protocol):
    """Provider interface for graph extraction and optional GraphRAG enrichment."""

    def extract_graph(self, chunks: list[Chunk], options: LlmOptions) -> ExtractionResult:
        """Extract entity and relation observations from a chunk batch."""
        ...

    def repair_graph_extraction(self, request: GraphRepairRequest) -> ExtractionResult:
        """Attempt to repair a provider response that failed schema validation."""
        ...

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Merge repeated entity descriptions into a canonical node description."""
        ...

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        """Generate a short report for a detected graph community."""
        ...
