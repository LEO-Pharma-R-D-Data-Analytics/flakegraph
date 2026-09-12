"""Intermediate extraction records kept separate from canonical graph rows.

The model first produces grounded mention observations.  Relations reference
those observations by local id, which makes dangling relation endpoints
impossible by construction.  Resolution later maps observations onto canonical
nodes while preserving every original mention and decision for auditability.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from kg_processor.domain.graph import Chunk


class ExtractionWindow(BaseModel):
    """Group adjacent chunks from one document into a bounded extraction task.

    Document identity is explicit so orchestration cannot accidentally infer facts
    between unrelated files batched for throughput.
    """

    id: str
    document_id: str
    chunks: list[Chunk]
    token_count: int


class EntityMention(BaseModel):
    """Represent one exact, typed entity mention accepted from source evidence.

    Mention identity remains separate from canonical graph nodes until conservative
    entity resolution has recorded how repeated surfaces should be grouped.
    """

    id: str
    name: str
    type: str
    description: str
    source_chunk_id: str
    quote: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)
    is_document_context: bool = False
    contextual_surfaces: list[str] = Field(default_factory=list)
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)


class EntityMentionBatch(BaseModel):
    """Hold grounded entity mentions from one extraction or targeted gleaning call.

    The wrapper gives provider-neutral stages a stable typed batch contract.
    """

    entities: list[EntityMention] = Field(default_factory=list)


class EntityExtractionOutcome(BaseModel):
    """Return grounded entities together with record-level provider diagnostics.

    Trace metadata explains accepted, rejected, repaired, and duplicate candidates
    without contaminating the persisted mention model.
    """

    entities: list[EntityMention] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)


class RelationObservation(BaseModel):
    """Represent one evidence-grounded predicate between accepted mention IDs.

    Endpoints cannot dangle because they reference first-pass mentions, and exact
    quote offsets preserve the assertion independently from canonical edge merging.
    """

    id: str
    source_entity_id: str
    target_entity_id: str
    source_surface: str | None = None
    target_surface: str | None = None
    relation_type: str
    description: str
    source_chunk_id: str
    quote: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_method: Literal["llm", "ontology_cue"] = "llm"
    verification_required: bool = True
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)


class RelationObservationBatch(BaseModel):
    """Hold grounded relation observations returned by one extraction-stage call.

    It separates provider-call batching from later graph-wide deduplication.
    """

    relations: list[RelationObservation] = Field(default_factory=list)


class RelationExtractionOutcome(BaseModel):
    """Return accepted relation observations and record-level rejection diagnostics.

    A malformed sibling can be traced and dropped without losing valid relations
    from the same structured response.
    """

    relations: list[RelationObservation] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)


class VerificationDecision(BaseModel):
    """Record an entailment verdict for one relation observation and exact evidence.

    Confidence and explanation remain available for filtering and operational audit.
    """

    relation_id: str
    verdict: Literal["supported", "contradicted", "insufficient"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class VerificationDecisionBatch(BaseModel):
    """Hold reconciled semantic-verification decisions for candidate relations.

    Orchestration ensures omitted IDs become conservative insufficient decisions.
    """

    decisions: list[VerificationDecision] = Field(default_factory=list)


class VerificationOutcome(BaseModel):
    """Return verifier decisions and provider metadata for extraction traces.

    The split keeps operational provenance outside the semantic decision records.
    """

    decisions: list[VerificationDecision] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)


class ResolutionCandidate(BaseModel):
    """Describe a potentially equivalent mention pair considered for resolution.

    Lexical and embedding scores make candidate selection and adjudication auditable.
    """

    left_id: str
    right_id: str
    lexical_score: float = Field(ge=0.0, le=1.0)
    embedding_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class ResolutionDecision(BaseModel):
    """Record whether two mentions denote one entity and why that decision was made.

    Both merge and non-merge outcomes are retained so graph identity is reviewable.
    """

    left_id: str
    right_id: str
    same_entity: bool
    canonical_name: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str


class ResolutionDecisionBatch(BaseModel):
    """Hold structured LLM judgments for a bounded set of uncertain identity pairs.

    Confidence policy is applied after this untrusted response is reconciled.
    """

    decisions: list[ResolutionDecision] = Field(default_factory=list)


class ExtractionObservations(BaseModel):
    """Persist window-parallel observations before graph-wide identity resolution.

    Extraction windows are independent by construction, so workers may produce
    these shards on different machines. Entity resolution remains a graph-wide
    barrier: combining shards before resolution preserves the same identity
    semantics as an in-process run over the complete corpus.
    """

    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[RelationObservation] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    chunk_count: int = Field(ge=0)
    window_count: int = Field(ge=0)
