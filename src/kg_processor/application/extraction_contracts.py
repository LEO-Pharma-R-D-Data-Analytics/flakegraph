# SPDX-License-Identifier: Apache-2.0
"""Strict request and response contracts for two-pass graph extraction tasks.

Natural-language instructions are intentionally owned by :mod:`kg_processor.prompts`
and loaded through the prompt registry. This module owns only typed payloads, dynamic
JSON schemas, token bounds, and provider-neutral request construction.
"""

from __future__ import annotations

import json
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from kg_processor.application.content_kinds import DATASET_KINDS
from kg_processor.application.prompt_registry import extraction_prompt, prompt_metadata
from kg_processor.application.structured_output import strict_json_schema
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionWindow,
    RelationObservation,
    ResolutionCandidate,
)
from kg_processor.domain.ids import sha256_hex
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


def dataset_profile_request(
    window: ExtractionWindow,
    ontology: OntologyProfile,
    *,
    file_name: str,
    model: str,
    timeout_seconds: int,
    seed: int,
) -> StructuredCompletionRequest:
    """Build a request that says what a dataset is, from its name and opening rows.

    The kind decides how much of the dataset the extraction passes read; the
    summary describes the file in the graph whichever kind it is.
    """

    payload = {
        "file_name": file_name,
        "entity_types": [
            {"name": item.name, "description": item.description} for item in ontology.entity_types
        ],
        "opening": [_chunk_payload(chunk) for chunk in window.chunks],
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": list(DATASET_KINDS)},
            "summary": {"type": "string", "maxLength": _MAX_DATASET_SUMMARY_CHARACTERS},
        },
        "required": ["kind", "summary"],
        "additionalProperties": False,
    }
    return _request(
        task_name="dataset_profile",
        model=model,
        payload=payload,
        schema=schema,
        timeout_seconds=timeout_seconds,
        max_tokens=1024,
        seed=seed,
    )


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


class QuantityCandidate(BaseModel):
    """One value a relation states, as the model reported it, before grounding."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        description="The value exactly as written, such as '≤ 0.5 %' or '98.5 - 101.5'."
    )
    kind: str = Field(
        description="What the value is: limit, result, target, typical, or another word."
    )
    unit: str = Field(description="The unit as written, or empty.")
    comparator: str = Field(description="A comparator such as '≤' or 'NMT' when stated, or empty.")


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
    quantities: list[QuantityCandidate] = Field(
        default_factory=list,
        description="Values this relation states, for relation types that carry quantities.",
    )


class EndpointCandidate(BaseModel):
    """A named thing a relation needs as an endpoint that the window's entities lack.

    The relation pass may add one - "Ida" is responsible for "the LAF bench
    scales" when only Ida was listed - and it is then grounded like any entity.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    type: str
    source_chunk_id: str
    quote: str


class RelationCandidateBatch(BaseModel):
    """Define the strict top-level payload returned by relation extraction.

    Keeping this contract separate from domain observations allows malformed
    siblings to be rejected without weakening persisted graph invariants.
    """

    model_config = ConfigDict(extra="forbid")
    relations: list[RelationCandidate]
    new_entities: list[EndpointCandidate] = Field(default_factory=list)


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

    # Ontology first, window last: identical across every window of a run, so
    # the engine's prefix cache covers it instead of re-reading it each time.
    payload = {
        "entity_type_definitions": [item.model_dump(mode="json") for item in ontology.entity_types],
        "entity_types": ontology.entity_type_names(),
        "document_id": window.document_id,
        "window_id": window.id,
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
    new_endpoints: bool = False,
) -> StructuredCompletionRequest:
    """Build a relation request constrained to accepted IDs and ontology labels.

    Endpoint and chunk enums make dangling or cross-window references invalid at
    schema generation time, before semantic grounding examines each record.
    """

    endpoint_ids = (
        [f"{NEW_ENDPOINT_PREFIX}{index}" for index in range(1, MAX_NEW_ENDPOINTS + 1)]
        if new_endpoints
        else []
    )
    entity_ids = [entity.id for entity in entities] + endpoint_ids
    payload = {
        "relation_type_definitions": [
            item.model_dump(mode="json") for item in ontology.relation_types
        ],
        "relation_types": ontology.relation_type_names(),
        "document_id": window.document_id,
        "window_id": window.id,
        "chunks": [_chunk_payload(chunk) for chunk in window.chunks],
        "entities": [_relation_entity_payload(entity) for entity in entities],
        "new_entities_allowed": new_endpoints,
        "relations_by_entity_type": _relations_by_entity_type(
            ontology, sorted({entity.type for entity in entities})
        ),
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
    if new_endpoints:
        endpoint_schema = _array_item_schema(schema, "new_entities")
        endpoint_schema["properties"]["id"] = {"type": "string", "enum": endpoint_ids}
        endpoint_schema["properties"]["type"] = {
            "type": "string",
            "enum": ontology.entity_type_names(),
        }
        endpoint_schema["properties"]["source_chunk_id"] = {
            "type": "string",
            "enum": [chunk.id for chunk in window.chunks],
        }
        endpoint_schema["properties"]["name"]["maxLength"] = 160
        endpoint_schema["properties"]["quote"]["maxLength"] = 600
        schema["properties"]["new_entities"]["maxItems"] = MAX_NEW_ENDPOINTS
        schema.setdefault("required", [])
        if "new_entities" not in schema["required"]:
            schema["required"].append("new_entities")
    else:
        # Off, the response has no endpoint list to fill.
        schema["properties"].pop("new_entities")
        schema.get("$defs", {}).pop("EndpointCandidate", None)
        if "new_entities" in schema.get("required", []):
            schema["required"].remove("new_entities")
    # Only an ontology that gives values a home asks for them; every other
    # profile keeps the relation contract it always had.
    if ontology.carries_quantities():
        item_schema["properties"]["quantities"]["maxItems"] = _MAX_RELATION_QUANTITIES
    else:
        item_schema["properties"].pop("quantities")
        if "quantities" in item_schema.get("required", []):
            item_schema["required"].remove("quantities")
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


def _relations_by_entity_type(
    ontology: OntologyProfile,
    entity_types: list[str],
) -> dict[str, dict[str, list[str]]]:
    """For each entity type in a window, the relations it may start and end.

    The definitions carry the same rules relation by relation; this reads them
    the way the model chooses - from the two entities it wants to relate.
    """

    return {
        entity_type: {
            "may_start": [
                item.name
                for item in ontology.relation_types
                if not item.source_types or entity_type in item.source_types
            ],
            "may_end": [
                item.name
                for item in ontology.relation_types
                if not item.target_types or entity_type in item.target_types
            ],
        }
        for entity_type in entity_types
    }


# Endpoints a relation response may add for things the window's entities lack,
# named new-1, new-2, ... so a relation can point at them.
NEW_ENDPOINT_PREFIX = "new-"
MAX_NEW_ENDPOINTS = 6
# A dataset's summary: one or two sentences.
_MAX_DATASET_SUMMARY_CHARACTERS = 600
# Values one relation may state, at most.
_MAX_RELATION_QUANTITIES = 6
# The fewest units that state one relation, at most.
_MAX_GROUNDING_UNITS = 6


def relation_grounding_request(
    units: list[tuple[int, str]],
    relations: list[tuple[int, str, str, str]],
    *,
    model: str,
    timeout_seconds: int,
    seed: int,
) -> StructuredCompletionRequest:
    """Build a request that may only point at numbered evidence units.

    The answer names, per relation, whether the units state it and which ones;
    it cannot write evidence text, so everything kept is exact source text.
    """

    ids = [f"r{index}" for index, *_ in relations]
    payload = {
        "units": [{"n": number, "text": text} for number, text in units],
        "relations": [
            {"id": f"r{index}", "source": source, "relation_type": label, "target": target}
            for index, source, label, target in relations
        ],
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "supported": {"type": "boolean"},
                        "lines": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "maxItems": _MAX_GROUNDING_UNITS,
                        },
                    },
                    "required": ["id", "supported", "lines"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }
    return _request(
        task_name="relation_grounding",
        model=model,
        payload=payload,
        schema=schema,
        timeout_seconds=timeout_seconds,
        max_tokens=min(4096, 256 + 48 * len(ids)),
        seed=seed,
    )


def entity_grounding_request(
    units: list[tuple[int, str]],
    entities: list[tuple[int, str, str, str]],
    *,
    model: str,
    timeout_seconds: int,
    seed: int,
) -> StructuredCompletionRequest:
    """Build a request that may only point at the numbered units naming each entity."""

    ids = [f"e{index}" for index, *_ in entities]
    payload = {
        "units": [{"n": number, "text": text} for number, text in units],
        "entities": [
            {"id": f"e{index}", "name": name, "type": entity_type, "quote": quote[:400]}
            for index, name, entity_type, quote in entities
        ],
    }
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "supported": {"type": "boolean"},
                        "lines": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "maxItems": _MAX_GROUNDING_UNITS,
                        },
                    },
                    "required": ["id", "supported", "lines"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }
    return _request(
        task_name="entity_grounding",
        model=model,
        payload=payload,
        schema=schema,
        timeout_seconds=timeout_seconds,
        max_tokens=min(4096, 256 + 48 * len(ids)),
        seed=seed,
    )


def relation_relabel_request(
    items: list[dict[str, Any]],
    *,
    model: str,
    timeout_seconds: int,
    seed: int,
) -> StructuredCompletionRequest:
    """Build a request that may only choose among a statement's allowed relations.

    Each statement carries numbered options - the relations the ontology allows
    between its two entities, in either direction. The answer is an option
    number, or -1 when none states what the text says.
    """

    ids = [str(item["id"]) for item in items]
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "option": {"type": "integer"},
                    },
                    "required": ["id", "option"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }
    return _request(
        task_name="relation_relabel",
        model=model,
        payload={"statements": items},
        schema=schema,
        timeout_seconds=timeout_seconds,
        max_tokens=min(4096, 256 + 32 * len(ids)),
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
    return sha256_hex(payload)


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
    request_checksum = sha256_hex(f"{prompt.system}\n{prompt.user}")
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


# What the relation pass needs of an entity: which record to point at, and
# enough to recognise it in the text. Its quote and description are already in
# `chunks` verbatim - sending them again cost about a fifth of the prompt and
# bought nothing, measured at 1.36x throughput when dropped.
_RELATION_ENTITY_FIELDS = ("id", "name", "type", "source_chunk_id", "aliases")


def _relation_entity_payload(entity: Any) -> dict[str, Any]:
    """Project one accepted mention into what the relation pass has to know."""

    record = entity.model_dump(mode="json")
    projected = {key: record[key] for key in _RELATION_ENTITY_FIELDS if key in record}
    if not projected.get("aliases"):
        projected.pop("aliases", None)
    # Only the document's own subject is marked; every other entity would carry
    # the flag for nothing.
    if record.get("is_document_subject"):
        projected["is_document_subject"] = True
    return projected


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
