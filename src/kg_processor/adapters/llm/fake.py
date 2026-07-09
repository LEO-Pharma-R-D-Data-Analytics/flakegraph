"""Deterministic LLM provider for tests and dry-run graph shape checks."""

from __future__ import annotations

import re

from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation, ExtractionResult
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    GraphRepairRequest,
    LlmOptions,
)

# Restrict multi-token fake entities to horizontal whitespace. Using ``\s``
# would let a line break join the last capitalized term on one line with the
# first capitalized term on the next line, which makes local smoke artifacts
# harder to review even though the provider is only deterministic test tooling.
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.-]*(?:[ \t]+[A-Z][A-Za-z0-9&.-]*){0,3}\b")
_STOP_NAMES = {"The", "This", "That", "A", "An", "It", "In", "On", "For", "And"}
_MIN_ENTITY_NAME_LEN = 3
_PERSON_MIN_TOKENS = 2
_ORG_MARKERS = ("Inc", "Corp", "Ltd", "LLC")


class FakeLlmProvider:
    """Deterministic provider for tests and local dry-runs."""

    def extract_graph(self, chunks: list[Chunk], options: LlmOptions) -> ExtractionResult:
        """Extract predictable title-cased entity observations from chunk text."""

        entities: list[ExtractedEntity] = []
        relations: list[ExtractedRelation] = []
        remaining_entities = options.max_entities_per_batch
        remaining_relations = options.max_relations_per_batch
        for chunk in chunks:
            # The fake provider is deterministic test tooling, but it still
            # honors the same batch limits passed to real LLM adapters. That
            # keeps smoke tests representative and lets larger fixture corpora
            # exercise OCR/file handling without exploding graph size.
            names = _ordered_unique(_extract_names(chunk.content))[:remaining_entities]
            for name in names:
                entities.append(
                    ExtractedEntity(
                        name=name,
                        type=_infer_type(name, options.entity_types),
                        description=f"{name} is mentioned in the source text.",
                        source_chunk_id=chunk.id,
                        confidence=1.0,
                    )
                )
            remaining_entities -= len(names)
            for source, target in list(zip(names, names[1:], strict=False))[:remaining_relations]:
                relations.append(
                    ExtractedRelation(
                        source_name=source,
                        target_name=target,
                        relation_type="mentions",
                        description=f"{source} is mentioned near {target}.",
                        source_chunk_id=chunk.id,
                        weight=1.0,
                    )
                )
            remaining_relations -= min(max(len(names) - 1, 0), remaining_relations)
            if remaining_entities <= 0:
                break
        return ExtractionResult(
            entities=entities,
            relations=relations,
            provider_metadata={
                "provider": "fake_llm",
                "model": options.model,
                "max_entities_per_batch": options.max_entities_per_batch,
                "max_relations_per_batch": options.max_relations_per_batch,
            },
        )

    def repair_graph_extraction(self, request: GraphRepairRequest) -> ExtractionResult:
        """Return a deterministic replacement extraction for repair tests."""

        result = self.extract_graph(request.chunks, request.options)
        result.provider_metadata = {
            **result.provider_metadata,
            "repair_prompt_name": "fake_repair",
            "repair_validation_error": request.validation_error,
        }
        return result

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Choose the longest provided description as a deterministic merge."""

        description = max(request.descriptions, key=len, default="")
        return DescriptionMergeResult(
            description=description,
            provider_metadata={
                "provider": "fake_llm",
                "model": "fake-description-merge",
            },
        )

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        """Return a deterministic community summary for local smoke tests."""

        members = ", ".join(request.members[:5])
        title = request.title_seed or (request.members[0] if request.members else "Community")
        summary = f"Community around {members}." if members else "Empty community."
        return CommunitySummaryResult(
            title=title[:160],
            summary=summary,
            rating=5.0,
            rating_explanation="The deterministic test provider assigns a neutral score.",
            findings=[("Connected entities", summary)],
            suggested_questions=[f"What connects {title}?"[:160]],
            provider_metadata={
                "provider": "fake_llm",
                "model": "fake-community-summary",
            },
        )


def _extract_names(text: str) -> list[str]:
    names = []
    for match in _ENTITY_RE.findall(text):
        cleaned = match.strip()
        if cleaned in _STOP_NAMES or len(cleaned) < _MIN_ENTITY_NAME_LEN:
            continue
        names.append(cleaned)
    return names


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _infer_type(name: str, allowed: list[str]) -> str:
    if "ORGANIZATION" in allowed and any(token in name for token in _ORG_MARKERS):
        return "ORGANIZATION"
    if "PERSON" in allowed and len(name.split()) >= _PERSON_MIN_TOKENS:
        return "PERSON"
    return "CONCEPT" if "CONCEPT" in allowed else allowed[0]
