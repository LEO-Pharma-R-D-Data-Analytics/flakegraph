"""Strict request and response contracts for two-pass graph extraction tasks.

Natural-language instructions are intentionally owned by :mod:`kg_processor.prompts`
and loaded through the prompt registry. This module owns only typed payloads, dynamic
JSON schemas, token bounds, and provider-neutral request construction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from kg_processor.application.prompt_registry import extraction_prompt, prompt_metadata
from kg_processor.application.structured_output import strict_json_schema
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionWindow,
    RelationObservation,
    ResolutionCandidate,
)
from kg_processor.domain.ontology import OntologyProfile
from kg_processor.ports.llm import StructuredCompletionRequest


class EntityCandidate(BaseModel):
    """Represent an untrusted entity candidate before deterministic grounding.

    Strict shape validation happens here, while source spans, aliases, ontology
    membership, and duplicate identity are validated by the entity-stage adapter.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    # The request carries every ontology type definition, and a description field
    # that states no purpose of its own invites the model to repeat the one it was
    # just given. The resulting text describes the category rather than the
    # entity, which reaches the explorer, the resolution embedding, and the
    # community report as if it were a finding about this entity.
    description: str = Field(
        description=(
            "One sentence about this specific entity, drawn from what the source "
            "says about it. Never the definition of its ontology type."
        )
    )
    source_chunk_id: str
    quote: str
    confidence: float = Field(ge=0.0, le=1.0)
    aliases: list[str]


class EntityCandidateBatch(BaseModel):
    """Define the complete strict provider payload for one entity extraction pass.

    Extra fields are forbidden so schema drift or explanatory model prose fails
    at the transport boundary rather than leaking into domain records.
    """

    model_config = ConfigDict(extra="forbid")
    entities: list[EntityCandidate]


def document_context_extraction_request(
    window: ExtractionWindow,
    ontology: OntologyProfile,
    *,
    model: str,
    timeout_seconds: int,
    max_entities: int,
    seed: int,
) -> StructuredCompletionRequest:
    """Build a strict request that identifies reusable document-context entities.

    The caller supplies only front-matter chunks and an ontology containing types
    explicitly marked for document context. Keeping this separate from ordinary
    entity extraction prevents cited works and background concepts from being
    promoted into document-wide discourse anchors.
    """

    payload = {
        "document_id": window.document_id,
        "window_id": window.id,
        "document_context_types": [
            item.model_dump(mode="json") for item in ontology.entity_types if item.document_context
        ],
        "front_matter_chunks": [_chunk_payload(chunk) for chunk in window.chunks],
    }
    schema = EntityCandidateBatch.model_json_schema()
    item_schema = _array_item_schema(schema, "entities")
    item_schema["properties"]["type"] = {
        "type": "string",
        "enum": [item.name for item in ontology.entity_types if item.document_context],
    }
    item_schema["properties"]["source_chunk_id"] = {
        "type": "string",
        "enum": [chunk.id for chunk in window.chunks],
    }
    schema["properties"]["entities"]["maxItems"] = max_entities
    return _request(
        task_name="document_context_extraction",
        model=model,
        payload=payload,
        schema=schema,
        timeout_seconds=timeout_seconds,
        max_tokens=2048,
        seed=seed,
    )


class RelationCandidate(BaseModel):
    """Represent an untrusted relation between previously accepted mention IDs.

    Later validation checks ontology signatures, direction, exact evidence, and
    duplicates before producing a grounded relation observation.
    """

    model_config = ConfigDict(extra="forbid")

    source_entity_id: str
    target_entity_id: str
    source_surface: str
    target_surface: str
    relation_type: str
    description: str = Field(
        description=(
            "One sentence stating what the source says holds between these two "
            "entities. Never the definition of the relation type."
        )
    )
    source_chunk_id: str
    quote: str
    confidence: float = Field(ge=0.0, le=1.0)


class RelationCandidateBatch(BaseModel):
    """Define the strict top-level payload returned by relation extraction.

    Keeping this contract separate from domain observations allows malformed
    siblings to be rejected without weakening persisted graph invariants.
    """

    model_config = ConfigDict(extra="forbid")
    relations: list[RelationCandidate]


class VerificationCandidate(BaseModel):
    """Represent one strict entailment decision before ID reconciliation.

    The verifier may only judge supplied relation IDs; orchestration later fills
    omitted decisions conservatively as insufficient evidence.
    """

    model_config = ConfigDict(extra="forbid")

    relation_id: str
    verdict: Literal["supported", "contradicted", "insufficient"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(
        description=(
            "One sentence giving the reason for this verdict on this relation. "
            "Never the definition of the relation type."
        )
    )


class VerificationCandidateBatch(BaseModel):
    """Define the complete strict payload for semantic relation verification.

    Request construction constrains item count and relation IDs so the model
    cannot introduce or silently replace graph triples during verification.
    """

    model_config = ConfigDict(extra="forbid")
    decisions: list[VerificationCandidate]


class ResolutionDecisionCandidate(BaseModel):
    """Represent one model judgment for a bounded uncertain identity pair.

    This remains an untrusted candidate until pair reconciliation and the
    configured merge-confidence threshold are applied.
    """

    model_config = ConfigDict(extra="forbid")

    left_id: str
    right_id: str
    same_entity: bool
    canonical_name: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class ResolutionDecisionCandidateBatch(BaseModel):
    """Define a strict batch of uncertain entity identity decisions.

    The request schema fixes participating IDs and expected record count, while
    resolution orchestration supplies conservative behavior for malformed output.
    """

    model_config = ConfigDict(extra="forbid")
    decisions: list[ResolutionDecisionCandidate]


def entity_extraction_request(
    window: ExtractionWindow,
    ontology: OntologyProfile,
    *,
    model: str,
    timeout_seconds: int,
    max_entities: int,
    seed: int,
    previous_entities: list[EntityMention] | None,
) -> StructuredCompletionRequest:
    """Build a strict first-pass request for grounded entities in one window.

    The ontology vocabulary, exact chunks, previous gleaning results, output
    limits, and provenance fingerprint are all encoded into one provider-neutral
    structured-completion request.
    """

    payload = {
        "document_id": window.document_id,
        "window_id": window.id,
        "entity_types": ontology.entity_type_names(),
        "entity_type_definitions": [item.model_dump(mode="json") for item in ontology.entity_types],
        "chunks": [_chunk_payload(chunk) for chunk in window.chunks],
        "previous_entities": [
            item.model_dump(mode="json", exclude={"start_offset", "end_offset"})
            for item in (previous_entities or [])
        ],
    }
    schema = EntityCandidateBatch.model_json_schema()
    item_schema = _array_item_schema(schema, "entities")
    item_schema["properties"]["type"] = {"type": "string", "enum": ontology.entity_type_names()}
    item_schema["properties"]["source_chunk_id"] = {
        "type": "string",
        "enum": [chunk.id for chunk in window.chunks],
    }
    item_schema["properties"]["name"]["maxLength"] = 160
    item_schema["properties"]["description"]["maxLength"] = 500
    item_schema["properties"]["quote"]["maxLength"] = 1200
    schema["properties"]["entities"]["maxItems"] = max_entities
    return _request(
        task_name="entity_extraction",
        model=model,
        payload=payload,
        schema=schema,
        timeout_seconds=timeout_seconds,
        max_tokens=8192,
        seed=seed,
    )


def relation_extraction_request(
    window: ExtractionWindow,
    entities: list[EntityMention],
    ontology: OntologyProfile,
    *,
    model: str,
    timeout_seconds: int,
    max_relations: int,
    seed: int,
    previous_relations: list[RelationObservation] | None,
) -> StructuredCompletionRequest:
    """Build a relation request constrained to accepted IDs and ontology labels.

    Endpoint and chunk enums make dangling or cross-window references invalid at
    schema generation time, before semantic grounding examines each record.
    """

    entity_ids = [entity.id for entity in entities]
    payload = {
        "document_id": window.document_id,
        "window_id": window.id,
        "relation_types": ontology.relation_type_names(),
        "relation_type_definitions": [
            item.model_dump(mode="json") for item in ontology.relation_types
        ],
        "chunks": [_chunk_payload(chunk) for chunk in window.chunks],
        "entities": [entity.model_dump(mode="json") for entity in entities],
        "previous_relations": [
            item.model_dump(mode="json", exclude={"start_offset", "end_offset"})
            for item in (previous_relations or [])
        ],
    }
    schema = RelationCandidateBatch.model_json_schema()
    item_schema = _array_item_schema(schema, "relations")
    item_schema["properties"]["source_entity_id"] = {"type": "string", "enum": entity_ids}
    item_schema["properties"]["target_entity_id"] = {"type": "string", "enum": entity_ids}
    item_schema["properties"]["source_surface"]["maxLength"] = 160
    item_schema["properties"]["target_surface"]["maxLength"] = 160
    if ontology.mode == "closed":
        item_schema["properties"]["relation_type"] = {
            "type": "string",
            "enum": ontology.relation_type_names(),
        }
    else:
        item_schema["properties"]["relation_type"]["maxLength"] = 80
    item_schema["properties"]["source_chunk_id"] = {
        "type": "string",
        "enum": [chunk.id for chunk in window.chunks],
    }
    item_schema["properties"]["description"]["maxLength"] = 500
    item_schema["properties"]["quote"]["maxLength"] = 1200
    schema["properties"]["relations"]["maxItems"] = max_relations
    return _request(
        task_name="relation_extraction",
        model=model,
        payload=payload,
        schema=schema,
        timeout_seconds=timeout_seconds,
        # Forty grounded records with quotes and descriptions can exceed 8K
        # tokens in dense scholarly windows. Providers apply their own lower
        # capability ceiling; local vLLM can complete the larger valid object.
        max_tokens=16384,
        seed=seed,
    )


def relation_verification_request(
    window: ExtractionWindow,
    entities: list[EntityMention],
    relations: list[RelationObservation],
    ontology: OntologyProfile,
    *,
    model: str,
    timeout_seconds: int,
    seed: int,
) -> StructuredCompletionRequest:
    """Build an entailment request that can only judge supplied relation IDs.

    Exact item bounds require one decision per candidate, preventing the verifier
    from adding new facts or silently changing relation identity.
    """

    payload = {
        "window_id": window.id,
        "relation_type_definitions": [
            item.model_dump(mode="json", exclude={"evidence_cues"})
            for item in ontology.relation_types
        ],
        "entities": [item.model_dump(mode="json") for item in entities],
        "relations": [item.model_dump(mode="json") for item in relations],
    }
    schema = VerificationCandidateBatch.model_json_schema()
    item_schema = _array_item_schema(schema, "decisions")
    item_schema["properties"]["relation_id"] = {
        "type": "string",
        "enum": [relation.id for relation in relations],
    }
    item_schema["properties"]["explanation"]["maxLength"] = 240
    schema["properties"]["decisions"]["minItems"] = len(relations)
    schema["properties"]["decisions"]["maxItems"] = len(relations)
    return _request(
        task_name="relation_verification",
        model=model,
        payload=payload,
        schema=schema,
        timeout_seconds=timeout_seconds,
        # Verification output grows linearly because the strict schema requires
        # exactly one decision per relation. Scale the budget to the request while
        # retaining the provider-neutral upper bound used by extraction adapters.
        max_tokens=min(8192, max(1024, 256 + len(relations) * 128)),
        seed=seed,
    )


def entity_resolution_request(
    mentions: list[EntityMention],
    candidates: list[ResolutionCandidate],
    *,
    model: str,
    timeout_seconds: int,
    seed: int,
) -> StructuredCompletionRequest:
    """Build a bounded adjudication request for uncertain identity pairs.

    Only mentions referenced by candidates are included, and output token budget
    scales with batch size to keep local inference predictable.
    """

    mention_ids = {candidate.left_id for candidate in candidates} | {
        candidate.right_id for candidate in candidates
    }
    payload = {
        "mentions": [
            mention.model_dump(mode="json") for mention in mentions if mention.id in mention_ids
        ],
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
    }
    schema = ResolutionDecisionCandidateBatch.model_json_schema()
    item_schema = _array_item_schema(schema, "decisions")
    ids = sorted(mention_ids)
    item_schema["properties"]["left_id"] = {"type": "string", "enum": ids}
    item_schema["properties"]["right_id"] = {"type": "string", "enum": ids}
    schema["properties"]["decisions"]["minItems"] = len(candidates)
    schema["properties"]["decisions"]["maxItems"] = len(candidates)
    # Resolution records are compact and the candidate batch is bounded. A
    # proportional budget prevents small local models from spending minutes on
    # runaway output while still leaving room for a concise reason per pair.
    max_tokens = min(4096, max(512, 256 + len(candidates) * 160))
    return _request(
        task_name="entity_resolution",
        model=model,
        payload=payload,
        schema=schema,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        seed=seed,
    )


def extraction_contract_fingerprint() -> str:
    """Fingerprint the contracts that decide what a provider may return.

    Cached extraction is only reusable while the request that produced it still
    describes the same output. A hand-maintained revision label is a second thing
    to remember on every contract change, and forgetting it serves the previous
    model's output for a schema it was never shown.
    """

    payload = "|".join(
        json.dumps(model.model_json_schema(), sort_keys=True, separators=(",", ":"))
        for model in (
            EntityCandidateBatch,
            RelationCandidateBatch,
            VerificationCandidateBatch,
            ResolutionDecisionCandidateBatch,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _request(
    *,
    task_name: str,
    model: str,
    payload: dict[str, Any],
    schema: dict[str, Any],
    timeout_seconds: int,
    max_tokens: int,
    seed: int,
) -> StructuredCompletionRequest:
    """Build a deterministic request and attach cache-safe prompt provenance.

    Temperature is fixed to zero and the prompt registry serializes machine input
    deterministically. Prompt metadata identifies the exact registered instruction;
    request data remains available through ordinary task traces and cache keys.
    """

    prompt = extraction_prompt(task_name, payload)
    metadata = prompt_metadata(prompt)
    request_checksum = hashlib.sha256(f"{prompt.system}\n{prompt.user}".encode()).hexdigest()
    return StructuredCompletionRequest(
        task_name=task_name,
        model=model,
        system=prompt.system,
        user=prompt.user,
        json_schema=strict_json_schema(schema),
        timeout_seconds=timeout_seconds,
        temperature=0.0,
        max_tokens=max_tokens,
        seed=seed,
        prompt_metadata={
            **metadata,
            "prompt_sha256": request_checksum,
        },
    )


def _chunk_payload(chunk: Any) -> dict[str, Any]:
    """Project a chunk into the minimal source context exposed to extraction models."""

    return {
        "id": str(chunk.id),
        "page_number": int(chunk.page_number),
        "section_path": list(chunk.section_path),
        "content": str(chunk.content),
    }


def _array_item_schema(schema: dict[str, Any], property_name: str) -> dict[str, Any]:
    """Resolve a Pydantic array item schema through an optional local reference.

    Callers mutate the resolved definition to add request-specific enums and
    bounds before the schema is closed for strict provider submission.
    """

    properties = schema.get("properties", {})
    property_schema = properties[property_name]
    item_schema = property_schema["items"]
    reference = item_schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        return cast(dict[str, Any], schema["$defs"][reference.rsplit("/", maxsplit=1)[1]])
    return cast(dict[str, Any], item_schema)
