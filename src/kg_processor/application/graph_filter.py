"""Deterministic grounding and filtering before graph merge.

Filters are intentionally free and explainable: static blocklists, confidence
thresholds, source-chunk grounding, and endpoint grounding run before any graph
rows are created so dropped observations can be traced without mutating state.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from kg_processor.application.graph_merge import normalize_entity_name
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation

_SIGNIFICANT_WORD_MIN_LEN = 2
_DEFAULT_BLOCKLIST = [
    "chapter",
    "section",
    "introduction",
    "conclusion",
    "paragraph",
    "document",
    "page",
]


@dataclass(frozen=True)
class FilterDecision:
    """Traceable decision explaining why an observation was kept or dropped."""

    kind: Literal["entity", "relation"]
    action: Literal["kept", "dropped"]
    reason: str
    source_chunk_id: str
    name: str | None = None
    source_name: str | None = None
    target_name: str | None = None
    relation_type: str | None = None
    confidence: float | None = None

    def to_trace_event(self) -> dict[str, object]:
        """Serialize the decision into the extraction trace format."""

        event: dict[str, object] = {
            "stage": "filter_decision",
            "kind": self.kind,
            "action": self.action,
            "reason": self.reason,
            "source_chunk_id": self.source_chunk_id,
        }
        _set_if_present(event, "name", self.name)
        _set_if_present(event, "source_name", self.source_name)
        _set_if_present(event, "target_name", self.target_name)
        _set_if_present(event, "relation_type", self.relation_type)
        _set_if_present(event, "confidence", self.confidence)
        return event


@dataclass(frozen=True)
class EntityFilterResult:
    """Filtered entity observations plus their explainable decisions."""

    kept: list[ExtractedEntity]
    decisions: list[FilterDecision]

    def dropped_reason_counts(self) -> dict[str, int]:
        """Return counts for dropped entity reasons."""

        return _reason_counts(self.decisions)


@dataclass(frozen=True)
class RelationFilterResult:
    """Filtered relation observations plus their explainable decisions."""

    kept: list[ExtractedRelation]
    decisions: list[FilterDecision]

    def dropped_reason_counts(self) -> dict[str, int]:
        """Return counts for dropped relation reasons."""

        return _reason_counts(self.decisions)


def filter_entities(
    entities: list[ExtractedEntity],
    chunks_by_id: dict[str, Chunk],
    min_confidence: float = 0.0,
    min_name_length: int = 2,
    blocklist: list[str] | None = None,
) -> list[ExtractedEntity]:
    """Return only entity observations that pass grounding and quality filters."""

    return filter_entities_with_decisions(
        entities,
        chunks_by_id,
        min_confidence,
        min_name_length,
        blocklist,
    ).kept


def filter_entities_with_decisions(
    entities: list[ExtractedEntity],
    chunks_by_id: dict[str, Chunk],
    min_confidence: float = 0.0,
    min_name_length: int = 2,
    blocklist: list[str] | None = None,
) -> EntityFilterResult:
    """Filter entity observations while preserving reviewable decisions."""

    kept: list[ExtractedEntity] = []
    decisions: list[FilterDecision] = []
    corpus_by_file = {chunk_id: chunk.content.lower() for chunk_id, chunk in chunks_by_id.items()}
    normalized_blocklist = {
        normalize_entity_name(value) for value in (blocklist or _DEFAULT_BLOCKLIST)
    }
    for entity in entities:
        if entity.confidence < min_confidence:
            decisions.append(_entity_decision(entity, "dropped", "low_confidence"))
            continue
        if len(entity.name.strip()) < min_name_length:
            decisions.append(_entity_decision(entity, "dropped", "name_too_short"))
            continue
        normalized = normalize_entity_name(entity.name)
        if normalized in normalized_blocklist:
            decisions.append(_entity_decision(entity, "dropped", "blocklisted_entity"))
            continue
        source_text = corpus_by_file.get(entity.source_chunk_id)
        if source_text is None:
            decisions.append(_entity_decision(entity, "dropped", "missing_source_chunk"))
            continue
        if _grounded(entity.name, source_text):
            kept.append(entity)
            decisions.append(_entity_decision(entity, "kept", "grounded"))
        else:
            decisions.append(_entity_decision(entity, "dropped", "ungrounded_name"))
    return EntityFilterResult(kept=kept, decisions=decisions)


def filter_relations(
    relations: list[ExtractedRelation],
    entities: list[ExtractedEntity],
    chunks_by_id: dict[str, Chunk] | None = None,
    min_confidence: float = 0.0,
    require_endpoint_grounding: bool = True,
) -> list[ExtractedRelation]:
    """Return only relation observations that pass endpoint and quality filters."""

    return filter_relations_with_decisions(
        relations,
        entities,
        chunks_by_id,
        min_confidence,
        require_endpoint_grounding,
    ).kept


def filter_relations_with_decisions(
    relations: list[ExtractedRelation],
    entities: list[ExtractedEntity],
    chunks_by_id: dict[str, Chunk] | None = None,
    min_confidence: float = 0.0,
    require_endpoint_grounding: bool = True,
) -> RelationFilterResult:
    """Filter relation observations while preserving reviewable decisions."""

    known = {normalize_entity_name(entity.name) for entity in entities}
    kept: list[ExtractedRelation] = []
    decisions: list[FilterDecision] = []
    for relation in relations:
        if relation.confidence < min_confidence:
            decisions.append(_relation_decision(relation, "dropped", "low_confidence"))
            continue
        if normalize_entity_name(relation.source_name) not in known:
            decisions.append(_relation_decision(relation, "dropped", "source_entity_missing"))
            continue
        if normalize_entity_name(relation.target_name) not in known:
            decisions.append(_relation_decision(relation, "dropped", "target_entity_missing"))
            continue
        if require_endpoint_grounding and chunks_by_id is not None:
            source_text = chunks_by_id.get(relation.source_chunk_id)
            if source_text is None:
                decisions.append(_relation_decision(relation, "dropped", "missing_source_chunk"))
                continue
            if not _grounded(relation.source_name, source_text.content.lower()):
                decisions.append(
                    _relation_decision(relation, "dropped", "source_endpoint_ungrounded")
                )
                continue
            if not _grounded(relation.target_name, source_text.content.lower()):
                decisions.append(
                    _relation_decision(relation, "dropped", "target_endpoint_ungrounded")
                )
                continue
        kept.append(relation)
        decisions.append(_relation_decision(relation, "kept", "grounded"))
    return RelationFilterResult(kept=kept, decisions=decisions)


def _grounded(name: str, source_text: str) -> bool:
    needle = name.lower()
    if needle in source_text:
        return True
    words = [word for word in needle.split() if len(word) > _SIGNIFICANT_WORD_MIN_LEN]
    return bool(words) and all(word in source_text for word in words)


def _entity_decision(
    entity: ExtractedEntity,
    action: Literal["kept", "dropped"],
    reason: str,
) -> FilterDecision:
    return FilterDecision(
        kind="entity",
        action=action,
        reason=reason,
        source_chunk_id=entity.source_chunk_id,
        name=entity.name,
        confidence=entity.confidence,
    )


def _relation_decision(
    relation: ExtractedRelation,
    action: Literal["kept", "dropped"],
    reason: str,
) -> FilterDecision:
    return FilterDecision(
        kind="relation",
        action=action,
        reason=reason,
        source_chunk_id=relation.source_chunk_id,
        source_name=relation.source_name,
        target_name=relation.target_name,
        relation_type=relation.relation_type,
        confidence=relation.confidence,
    )


def _reason_counts(decisions: list[FilterDecision]) -> dict[str, int]:
    counter = Counter(decision.reason for decision in decisions if decision.action == "dropped")
    return dict(sorted(counter.items()))


def _set_if_present(event: dict[str, object], key: str, value: object) -> None:
    if value is not None:
        event[key] = value
