"""Structured extraction validation and semantic grounding checks.

The JSON schema catches shape errors; the semantic validators catch hallucinated
chunk ids, invalid quotes/spans, and over-limit responses before observations can
reach merge logic. Offset normalization is deterministic so brittle provider
spans do not discard otherwise grounded quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation, ExtractionResult
from kg_processor.ports.llm import LlmOptions

EXTRACTION_SCHEMA_REVISION = "graph-extraction-current"
_MAX_DESCRIPTION_LENGTH = 2000
_MAX_RELATION_TYPE_LENGTH = 120


def extraction_json_schema() -> dict[str, Any]:
    """Return the JSON schema providers must satisfy for extraction results."""

    schema = TypeAdapter(ExtractionResult).json_schema()
    schema["additionalProperties"] = False
    return schema


def extraction_response_format() -> dict[str, Any]:
    """Return the provider-neutral structured response format for extraction."""

    return {
        "type": "json",
        "schema": extraction_json_schema(),
    }


def schema_hint(chunks: list[Chunk], options: LlmOptions) -> dict[str, object]:
    """Build a compact example payload included in extraction prompts."""

    chunk_id = chunks[0].id if chunks else "chunk_id"
    entity_type = options.entity_types[0] if options.entity_types else "CONCEPT"
    return {
        "entities": [
            {
                "name": "string",
                "type": entity_type,
                "description": "string",
                "source_chunk_id": chunk_id,
                "quote": "short exact supporting text from the chunk, or null",
                "start_offset": 0,
                "end_offset": 10,
                "confidence": 1.0,
                "aliases": [],
            }
        ],
        "relations": [
            {
                "source_name": "string",
                "target_name": "string",
                "relation_type": "string",
                "description": "string",
                "source_chunk_id": chunk_id,
                "quote": "short exact supporting text from the chunk, or null",
                "start_offset": 0,
                "end_offset": 10,
                "weight": 1.0,
                "confidence": 1.0,
            }
        ],
    }


def validate_extraction_result(
    result: ExtractionResult,
    chunks: list[Chunk],
    options: LlmOptions,
    pass_index: int,
    *,
    drop_ungrounded_observations: bool = False,
) -> ExtractionResult:
    """Validate and normalize extraction observations against chunk evidence."""

    # Pydantic validates shape before this function is called; these checks
    # enforce the semantic GraphRAG contract that every observation is grounded
    # in the exact chunk batch sent to the provider.
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    valid_chunk_ids = set(chunks_by_id)
    if not valid_chunk_ids and (result.entities or result.relations):
        raise ValueError("LLM returned graph observations for an empty chunk batch")
    entities, entity_stats = _validate_entities(
        result.entities,
        chunks_by_id,
        options,
        drop_ungrounded_observations,
    )
    relations, relation_stats = _validate_relations(
        result.relations,
        chunks_by_id,
        options,
        drop_ungrounded_observations,
    )
    stats = entity_stats + relation_stats
    metadata = dict(result.provider_metadata)
    metadata.update(
        {
            "schema_revision": EXTRACTION_SCHEMA_REVISION,
            "extraction_pass": pass_index,
            "batch_chunk_ids": sorted(valid_chunk_ids),
            "evidence_offsets_normalized": stats.evidence_normalizations,
            "dropped_entities": stats.dropped_entities,
            "dropped_relations": stats.dropped_relations,
            "dropped_observations_by_reason": stats.dropped_observations_by_reason,
        }
    )
    return ExtractionResult(
        entities=entities,
        relations=relations,
        provider_metadata=metadata,
    )


def merge_extraction_results(results: list[ExtractionResult]) -> ExtractionResult:
    """Concatenate extraction batches while preserving provider metadata."""

    merged = ExtractionResult()
    metadata: dict[str, Any] = {
        "schema_revision": EXTRACTION_SCHEMA_REVISION,
        "batches": [],
    }
    for result in results:
        merged.entities.extend(result.entities)
        merged.relations.extend(result.relations)
        metadata["batches"].append(result.provider_metadata)
    merged.provider_metadata = metadata
    return merged


def dedupe_extraction_result(result: ExtractionResult) -> ExtractionResult:
    """Remove duplicate entity and relation observations deterministically."""

    seen_entities: set[tuple[str, str, str]] = set()
    entities: list[ExtractedEntity] = []
    for entity in result.entities:
        entity_key = (_normalize(entity.name), entity.type, entity.source_chunk_id)
        if entity_key in seen_entities:
            continue
        seen_entities.add(entity_key)
        entities.append(entity)

    seen_relations: set[tuple[str, str, str, str]] = set()
    relations: list[ExtractedRelation] = []
    for relation in result.relations:
        relation_key = (
            _normalize(relation.source_name),
            _normalize(relation.target_name),
            relation.relation_type.strip().lower(),
            relation.source_chunk_id,
        )
        if relation_key in seen_relations:
            continue
        seen_relations.add(relation_key)
        relations.append(relation)

    return ExtractionResult(
        entities=entities,
        relations=relations,
        provider_metadata=result.provider_metadata,
    )


def _validate_entities(
    entities: list[ExtractedEntity],
    chunks_by_id: dict[str, Chunk],
    options: LlmOptions,
    drop_ungrounded_observations: bool,
) -> tuple[list[ExtractedEntity], _ValidationStats]:
    if len(entities) > options.max_entities_per_batch:
        raise ValueError(
            "LLM returned too many entities for one batch: "
            f"{len(entities)} > {options.max_entities_per_batch}"
        )
    valid_types = set(options.entity_types)
    validated: list[ExtractedEntity] = []
    stats = _ValidationStats()
    for entity in entities:
        chunk = chunks_by_id.get(entity.source_chunk_id)
        if chunk is None:
            raise ValueError(f"Entity cites invalid source_chunk_id: {entity.source_chunk_id}")
        if valid_types and entity.type not in valid_types:
            raise ValueError(f"Entity type is outside the configured taxonomy: {entity.type}")
        if not entity.name.strip():
            raise ValueError("Entity name must not be blank")
        if len(entity.description) > _MAX_DESCRIPTION_LENGTH:
            raise ValueError(f"Entity description is too long: {entity.name}")
        try:
            evidence = _normalize_observation_evidence(
                chunk,
                entity.quote,
                entity.start_offset,
                entity.end_offset,
                f"Entity {entity.name}",
            )
        except ValueError as exc:
            if _drop_ungrounded_observation(drop_ungrounded_observations, exc):
                stats = stats.drop_entity("ungrounded_quote")
                continue
            raise
        if evidence.normalized:
            stats = stats.normalize_evidence()
        validated.append(
            entity.model_copy(
                update={
                    "quote": evidence.quote,
                    "start_offset": evidence.start_offset,
                    "end_offset": evidence.end_offset,
                }
            )
        )
    return validated, stats


def _validate_relations(
    relations: list[ExtractedRelation],
    chunks_by_id: dict[str, Chunk],
    options: LlmOptions,
    drop_ungrounded_observations: bool,
) -> tuple[list[ExtractedRelation], _ValidationStats]:
    if len(relations) > options.max_relations_per_batch:
        raise ValueError(
            "LLM returned too many relations for one batch: "
            f"{len(relations)} > {options.max_relations_per_batch}"
        )
    valid_relation_types = set(options.relation_types or [])
    validated: list[ExtractedRelation] = []
    stats = _ValidationStats()
    for relation in relations:
        chunk = chunks_by_id.get(relation.source_chunk_id)
        if chunk is None:
            raise ValueError(f"Relation cites invalid source_chunk_id: {relation.source_chunk_id}")
        if valid_relation_types and relation.relation_type not in valid_relation_types:
            raise ValueError(
                f"Relation type is outside the configured taxonomy: {relation.relation_type}"
            )
        if not relation.source_name.strip() or not relation.target_name.strip():
            raise ValueError("Relation endpoints must not be blank")
        if len(relation.relation_type) > _MAX_RELATION_TYPE_LENGTH:
            raise ValueError(f"Relation type is too long: {relation.relation_type}")
        if len(relation.description) > _MAX_DESCRIPTION_LENGTH:
            raise ValueError(f"Relation description is too long: {relation.relation_type}")
        try:
            evidence = _normalize_observation_evidence(
                chunk,
                relation.quote,
                relation.start_offset,
                relation.end_offset,
                f"Relation {relation.source_name} -> {relation.target_name}",
            )
        except ValueError as exc:
            if _drop_ungrounded_observation(drop_ungrounded_observations, exc):
                stats = stats.drop_relation("ungrounded_quote")
                continue
            raise
        if evidence.normalized:
            stats = stats.normalize_evidence()
        validated.append(
            relation.model_copy(
                update={
                    "quote": evidence.quote,
                    "start_offset": evidence.start_offset,
                    "end_offset": evidence.end_offset,
                }
            )
        )
    return validated, stats


@dataclass(frozen=True)
class _ValidationStats:
    """Immutable counters collected while validating extraction observations."""

    evidence_normalizations: int = 0
    dropped_entities: int = 0
    dropped_relations: int = 0
    dropped_observations_by_reason: dict[str, int] | None = None

    def normalize_evidence(self) -> _ValidationStats:
        """Return stats with one additional evidence normalization recorded."""

        return _ValidationStats(
            evidence_normalizations=self.evidence_normalizations + 1,
            dropped_entities=self.dropped_entities,
            dropped_relations=self.dropped_relations,
            dropped_observations_by_reason=dict(self.dropped_observations_by_reason or {}),
        )

    def drop_entity(self, reason: str) -> _ValidationStats:
        """Return stats with one dropped entity recorded for the reason."""

        return self._drop(reason, entity=True)

    def drop_relation(self, reason: str) -> _ValidationStats:
        """Return stats with one dropped relation recorded for the reason."""

        return self._drop(reason, entity=False)

    def _drop(self, reason: str, *, entity: bool) -> _ValidationStats:
        reasons = dict(self.dropped_observations_by_reason or {})
        reasons[reason] = reasons.get(reason, 0) + 1
        return _ValidationStats(
            evidence_normalizations=self.evidence_normalizations,
            dropped_entities=self.dropped_entities + (1 if entity else 0),
            dropped_relations=self.dropped_relations + (0 if entity else 1),
            dropped_observations_by_reason=reasons,
        )

    def __add__(self, other: _ValidationStats) -> _ValidationStats:
        reasons = dict(self.dropped_observations_by_reason or {})
        for reason, count in (other.dropped_observations_by_reason or {}).items():
            reasons[reason] = reasons.get(reason, 0) + count
        return _ValidationStats(
            evidence_normalizations=(
                self.evidence_normalizations + other.evidence_normalizations
            ),
            dropped_entities=self.dropped_entities + other.dropped_entities,
            dropped_relations=self.dropped_relations + other.dropped_relations,
            dropped_observations_by_reason=reasons,
        )


def _drop_ungrounded_observation(enabled: bool, exc: ValueError) -> bool:
    # Dropping is only allowed after a repair attempt and only for quotes that
    # still cannot be found in the source chunk. Taxonomy, limit, blank-field,
    # and invalid-chunk errors remain hard failures because they indicate a
    # broken provider contract rather than an over-eager observation.
    return enabled and "quote is not present in source chunk" in str(exc)


@dataclass(frozen=True)
class _NormalizedEvidence:
    """Normalized quote/span evidence returned by semantic validators."""

    quote: str | None
    start_offset: int | None
    end_offset: int | None
    normalized: bool = False


def _normalize_observation_evidence(
    chunk: Chunk,
    quote: str | None,
    start_offset: int | None,
    end_offset: int | None,
    label: str,
) -> _NormalizedEvidence:
    if quote is not None and not quote.strip():
        raise ValueError(f"{label} quote must not be blank")
    normalized_quote = quote.strip() if quote is not None else None
    if (start_offset is None) != (end_offset is None):
        if normalized_quote is not None:
            # Providers often know the exact quote but return one missing offset;
            # source-searching the quote keeps grounded evidence instead of
            # rejecting a recoverable span formatting issue.
            return _evidence_from_quote(chunk, normalized_quote, label, normalized=True)
        raise ValueError(f"{label} must provide both start_offset and end_offset")
    if start_offset is not None and end_offset is not None:
        if start_offset < 0:
            raise ValueError(f"{label} start_offset must be non-negative")
        if start_offset >= end_offset or end_offset > len(chunk.content):
            if normalized_quote is not None:
                return _evidence_from_quote(chunk, normalized_quote, label, normalized=True)
            if start_offset >= end_offset:
                raise ValueError(f"{label} start_offset must be smaller than end_offset")
            raise ValueError(f"{label} end_offset exceeds source chunk length")
        if normalized_quote is not None:
            span = chunk.content[start_offset:end_offset]
            if span.strip().lower() != normalized_quote.lower():
                return _evidence_from_quote(chunk, normalized_quote, label, normalized=True)
        return _NormalizedEvidence(
            quote=normalized_quote,
            start_offset=start_offset,
            end_offset=end_offset,
        )
    if normalized_quote is not None:
        return _evidence_from_quote(chunk, normalized_quote, label, normalized=True)
    return _NormalizedEvidence(quote=None, start_offset=None, end_offset=None)


def _evidence_from_quote(
    chunk: Chunk,
    quote: str,
    label: str,
    *,
    normalized: bool,
) -> _NormalizedEvidence:
    local_start = chunk.content.lower().find(quote.lower())
    if local_start < 0:
        raise ValueError(f"{label} quote is not present in source chunk")
    return _NormalizedEvidence(
        quote=quote,
        start_offset=local_start,
        end_offset=local_start + len(quote),
        normalized=normalized,
    )


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())
