"""LLM-backed implementations of the extraction-stage ports.

These adapters translate strict provider payloads into grounded domain records.
Shape errors fail the individual call; semantic errors reject only the affected
record and are counted in the returned trace.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ValidationError

from kg_processor.application.entity_names import identity_surface_match, validate_identity_aliases
from kg_processor.application.extraction_contracts import (
    EntityCandidate,
    RelationCandidate,
    VerificationCandidate,
    document_context_extraction_request,
    entity_extraction_request,
    relation_extraction_request,
    relation_verification_request,
)
from kg_processor.application.extraction_grounding import (
    ground_evidence,
    surface_occurs,
    surface_spans,
)
from kg_processor.application.relation_grounding import relation_evidence_supported
from kg_processor.domain.extraction import (
    EntityExtractionOutcome,
    EntityMention,
    ExtractionWindow,
    RelationExtractionOutcome,
    RelationObservation,
    VerificationDecision,
    VerificationOutcome,
)
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import (
    OntologyProfile,
    RelationTypeDefinition,
    normalize_ontology_label,
)
from kg_processor.ports.llm import StructuredCompletionProvider, StructuredCompletionRequest

_MIN_RELATION_ENTITIES = 2


class LlmEntityExtractor:
    """Extract and independently ground entity mentions through an LLM port.

    Provider records are treated as untrusted candidates. Ontology labels,
    aliases, chunk identity, evidence spans, duplicates, and output limits are
    enforced before a mention can enter resolution or graph assembly.
    """

    def __init__(self, llm: StructuredCompletionProvider, seed: int = 17) -> None:
        """Store the structured provider and deterministic seed used by all entity requests."""

        self.llm = llm
        self.seed = seed

    def extract(
        self,
        window: ExtractionWindow,
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_entities: int,
        previous_entities: list[EntityMention] | None = None,
    ) -> EntityExtractionOutcome:
        """Return only ontology-valid candidates anchored to exact source text.

        Invalid siblings are counted and discarded independently, allowing one
        local model mistake to remain local while preserving valid observations
        and complete provider provenance in the trace.
        """

        request = entity_extraction_request(
            window,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            max_entities=max_entities,
            seed=self.seed,
            previous_entities=previous_entities,
        )
        return self._extract_from_request(
            window,
            ontology,
            request,
            max_entities=max_entities,
            previous_entities=previous_entities,
            stage="entity_extraction",
            document_context=False,
        )

    def extract_document_context_entities(
        self,
        window: ExtractionWindow,
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_entities: int,
    ) -> EntityExtractionOutcome:
        """Identify reusable focal entities from bounded document front matter.

        Context mentions are ordinary grounded entities with an additional discourse
        role. Stable contextual surfaces let later independent windows resolve
        explicit phrases such as ``this paper`` without making those pronouns global
        identity aliases during graph-wide entity resolution.
        """

        request = document_context_extraction_request(
            window,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            max_entities=max_entities,
            seed=self.seed,
        )
        return self._extract_from_request(
            window,
            ontology,
            request,
            max_entities=max_entities,
            previous_entities=None,
            stage="document_context_extraction",
            document_context=True,
        )

    def _extract_from_request(
        self,
        window: ExtractionWindow,
        ontology: OntologyProfile,
        request: StructuredCompletionRequest,
        *,
        max_entities: int,
        previous_entities: list[EntityMention] | None,
        stage: str,
        document_context: bool,
    ) -> EntityExtractionOutcome:
        """Ground one strict entity response under ordinary or document context semantics."""

        completion = self.llm.complete_structured(request)
        chunks_by_id = {chunk.id: chunk for chunk in window.chunks}
        valid_types = set(ontology.entity_type_names())
        previous_keys = {
            (_normalize(item.name), item.type, item.source_chunk_id)
            for item in (previous_entities or [])
        }
        seen = set(previous_keys)
        # Read once per response rather than per candidate: the profile is fixed
        # for the whole request and is not hashable, so it cannot be cached.
        definitions = _ontology_definition_keys(ontology)
        reasons: Counter[str] = Counter()
        raw_entities, candidates = _validated_records(
            completion.payload,
            "entities",
            EntityCandidate,
            reasons,
        )
        entities: list[EntityMention] = []
        for candidate in candidates[:max_entities]:
            chunk = chunks_by_id.get(candidate.source_chunk_id)
            if chunk is None:
                reasons["invalid_chunk"] += 1
                continue
            if candidate.type not in valid_types:
                reasons["invalid_entity_type"] += 1
                continue
            name = candidate.name.strip()
            if not name:
                reasons["blank_name"] += 1
                continue
            if not document_context and _is_contextual_surface_name(
                name,
                candidate.type,
                ontology,
            ):
                # Contextual surfaces are discourse pointers, not identities. The
                # matching document-context mention remains available to relation
                # extraction, where the same surface can ground a real endpoint.
                reasons["contextual_surface_as_entity"] += 1
                continue
            key = (_normalize(name), candidate.type, chunk.id)
            if key in seen:
                reasons["duplicate"] += 1
                continue
            alias_validation = validate_identity_aliases(
                name,
                candidate.aliases,
                chunk.content,
            )
            if alias_validation.rejected_count:
                reasons["untrusted_alias"] += alias_validation.rejected_count
            if document_context and not _document_context_surface_match(
                name,
                candidate.quote,
                alias_validation.aliases,
            ):
                reasons["document_context_name_mismatch"] += 1
                continue
            grounded = ground_evidence(
                chunk.content,
                candidate.quote,
                ([[candidate.quote]] if document_context else [[name, *alias_validation.aliases]]),
            )
            if grounded is None:
                reasons["ungrounded_quote"] += 1
                continue
            seen.add(key)
            if grounded.repair:
                reasons[f"repaired_{grounded.repair}"] += 1
            description = _entity_description(candidate.description, definitions)
            if not description and candidate.description.strip():
                reasons["type_definition_as_description"] += 1
            entities.append(
                EntityMention(
                    id=stable_id(
                        "entity_mention",
                        window.id,
                        chunk.id,
                        candidate.type,
                        _normalize(name),
                        grounded.start_offset,
                    ),
                    name=name,
                    type=candidate.type,
                    description=description,
                    source_chunk_id=chunk.id,
                    quote=grounded.quote,
                    confidence=candidate.confidence,
                    aliases=_dedupe_aliases(alias_validation.aliases, name),
                    is_document_context=document_context,
                    contextual_surfaces=(
                        next(
                            (
                                list(definition.contextual_surfaces)
                                for definition in ontology.entity_types
                                if definition.name == candidate.type
                            ),
                            [],
                        )
                        if document_context
                        else []
                    ),
                    start_offset=grounded.start_offset,
                    end_offset=grounded.end_offset,
                )
            )
        return EntityExtractionOutcome(
            entities=entities,
            trace={
                "stage": stage,
                "window_id": window.id,
                "input_records": len(raw_entities),
                "accepted_records": len(entities),
                "record_actions": dict(sorted(reasons.items())),
                **completion.provider_metadata,
            },
        )


def _is_contextual_surface_name(
    name: str,
    entity_type: str,
    ontology: OntologyProfile,
) -> bool:
    """Identify a discourse placeholder emitted as an ordinary entity name.

    Document-context surfaces intentionally let relations in independent windows
    refer back to a focal paper, method, or model through phrases such as ``we``
    and ``our method``. Accepting the same phrases as ordinary entity candidates
    creates competing canonical nodes and diverts those relations away from the
    grounded focal entity. Matching is exact after whitespace and case
    normalization so legitimate names that merely contain a common word remain
    unaffected.
    """

    definition = next(
        (item for item in ontology.entity_types if item.name == entity_type),
        None,
    )
    if definition is None or not definition.document_context:
        return False
    normalized_name = _normalize(name)
    return normalized_name in {_normalize(surface) for surface in definition.contextual_surfaces}


class LlmRelationExtractor:
    """Extract grounded relations constrained to previously accepted entity IDs.

    The adapter validates ontology domain/range rules, repairs unambiguous
    direction mistakes, requires relation cues and endpoint evidence, and rejects
    self-loops unless the ontology explicitly permits them.
    """

    def __init__(self, llm: StructuredCompletionProvider, seed: int = 17) -> None:
        """Store the structured provider and deterministic seed used by relation requests.

        No provider work occurs during construction.
        """

        self.llm = llm
        self.seed = seed

    def extract(  # noqa: PLR0912, PLR0915 - branches record distinct rejection reasons.
        self,
        window: ExtractionWindow,
        entities: list[EntityMention],
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_relations: int,
        previous_relations: list[RelationObservation] | None = None,
    ) -> RelationExtractionOutcome:
        """Ground, type-check, orient, and deduplicate relation candidates.

        Fewer than two accepted entities short-circuits the provider call. For
        larger windows, each malformed or unsupported record is rejected without
        invalidating valid sibling relations from the same response.
        """

        if len(entities) < _MIN_RELATION_ENTITIES:
            return RelationExtractionOutcome(
                trace={
                    "stage": "relation_extraction",
                    "window_id": window.id,
                    "input_records": 0,
                    "accepted_records": 0,
                    "skipped": "fewer_than_two_entities",
                }
            )
        request = relation_extraction_request(
            window,
            entities,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            max_relations=max_relations,
            seed=self.seed,
            previous_relations=previous_relations,
        )
        completion = self.llm.complete_structured(request)
        entities_by_id = {entity.id: entity for entity in entities}
        chunks_by_id = {chunk.id: chunk for chunk in window.chunks}
        previous_keys = {
            (
                item.source_entity_id,
                item.target_entity_id,
                item.relation_type,
                item.source_chunk_id,
            )
            for item in (previous_relations or [])
        }
        seen = set(previous_keys)
        reasons: Counter[str] = Counter()
        raw_relations, candidates = _validated_records(
            completion.payload,
            "relations",
            RelationCandidate,
            reasons,
        )
        relations: list[RelationObservation] = []
        for candidate in candidates[:max_relations]:
            source = entities_by_id.get(candidate.source_entity_id)
            target = entities_by_id.get(candidate.target_entity_id)
            chunk = chunks_by_id.get(candidate.source_chunk_id)
            if source is None or target is None:
                reasons["invalid_endpoint"] += 1
                continue
            definition = _relation_definition(candidate.relation_type, ontology, reasons)
            if definition is None:
                continue
            if source.id == target.id and not definition.allow_self_loop:
                reasons["self_loop"] += 1
                continue
            source, target, direction_repaired = _orient_relation(source, target, definition)
            if not _types_allowed(source.type, target.type, definition):
                reasons["domain_or_range_violation"] += 1
                continue
            if direction_repaired:
                reasons["repaired_direction"] += 1
            if chunk is None:
                reasons["invalid_chunk"] += 1
                continue
            endpoint_surfaces = _validated_endpoint_surfaces(candidate, reasons)
            if endpoint_surfaces is None:
                continue
            source_surface, target_surface = endpoint_surfaces
            if direction_repaired:
                # Endpoint surfaces belong to the candidate's original direction.
                # Keep them aligned with the repaired entity IDs so evidence never
                # attributes one local mention to the opposite canonical endpoint.
                source_surface, target_surface = target_surface, source_surface
            key = (source.id, target.id, definition.name, chunk.id)
            if key in seen:
                reasons["duplicate"] += 1
                continue
            required_surface_groups, implicit_document_source = _relation_required_surface_groups(
                chunk.content,
                source,
                source_surface,
                target,
                target_surface,
            )
            grounded = ground_evidence(
                chunk.content,
                candidate.quote,
                required_surface_groups,
            )
            if grounded is None:
                reasons["ungrounded_quote"] += 1
                continue
            if implicit_document_source:
                reasons["implicit_document_source"] += 1
            repaired_source_surface = _grounded_endpoint_surface(
                grounded.quote,
                source,
                source_surface,
            )
            repaired_target_surface = _grounded_endpoint_surface(
                grounded.quote,
                target,
                target_surface,
            )
            if repaired_source_surface is None and implicit_document_source:
                # A provenance-only source has no local mention by definition.
                # Persisting its canonical name identifies the endpoint clearly
                # for semantic verification and downstream evidence inspection.
                repaired_source_surface = source.name
            if repaired_source_surface is None or repaired_target_surface is None:
                reasons["unresolved_grounded_surface"] += 1
                continue
            if repaired_source_surface != source_surface:
                reasons["repaired_source_surface"] += 1
            if repaired_target_surface != target_surface:
                reasons["repaired_target_surface"] += 1
            source_surface = repaired_source_surface
            target_surface = repaired_target_surface
            if not relation_evidence_supported(
                grounded.quote,
                [source_surface],
                [target_surface],
                definition,
            ):
                # Ontology cues are useful diagnostics, but natural-language
                # predicates have unbounded surface forms. Exact endpoint and
                # evidence grounding plus the independent semantic verifier are
                # the authoritative entailment checks; an incomplete regex list
                # must not silently erase an otherwise valid candidate.
                reasons["cue_not_found"] += 1
            seen.add(key)
            if grounded.repair:
                reasons[f"repaired_{grounded.repair}"] += 1
            relations.append(
                RelationObservation(
                    id=stable_id(
                        "relation_observation",
                        window.id,
                        source.id,
                        target.id,
                        definition.name,
                        chunk.id,
                        grounded.start_offset,
                    ),
                    source_entity_id=source.id,
                    target_entity_id=target.id,
                    source_surface=source_surface,
                    target_surface=target_surface,
                    relation_type=definition.name,
                    description=candidate.description.strip(),
                    source_chunk_id=chunk.id,
                    quote=grounded.quote,
                    confidence=candidate.confidence,
                    start_offset=grounded.start_offset,
                    end_offset=grounded.end_offset,
                )
            )
        return RelationExtractionOutcome(
            relations=relations,
            trace={
                "stage": "relation_extraction",
                "window_id": window.id,
                "input_records": len(raw_relations),
                "accepted_records": len(relations),
                "record_actions": dict(sorted(reasons.items())),
                **completion.provider_metadata,
            },
        )


def _relation_definition(
    relation_type: str,
    ontology: OntologyProfile,
    reasons: Counter[str],
) -> RelationTypeDefinition | None:
    """Resolve a declared predicate or create a safe open-vocabulary definition.

    Closed profiles reject unknown labels. Open and hybrid profiles retain an
    otherwise grounded predicate with no domain/range restrictions, allowing the
    declared vocabulary to guide extraction without silently erasing useful typing.
    """

    definition = ontology.relation(relation_type)
    if definition is not None:
        return definition
    if ontology.mode == "closed":
        reasons["invalid_relation_type"] += 1
        return None
    normalized_type = normalize_ontology_label(relation_type).upper()
    if not normalized_type:
        reasons["blank_relation_type"] += 1
        return None
    reasons["open_relation_type"] += 1
    return RelationTypeDefinition(
        name=normalized_type,
        description=("Source-grounded relation discovered outside the declared vocabulary."),
    )


def _validated_endpoint_surfaces(
    candidate: RelationCandidate,
    reasons: Counter[str],
) -> tuple[str, str] | None:
    """Validate exact local endpoint mentions supplied with a relation candidate.

    Local surfaces let a canonical inventory entry participate through a grounded
    shortened form in another sentence. Blank or identical surfaces cannot prove
    two distinct endpoints and are rejected before evidence grounding.
    """

    source_surface = candidate.source_surface.strip()
    target_surface = candidate.target_surface.strip()
    if not source_surface or not target_surface:
        reasons["blank_endpoint_surface"] += 1
        return None
    if _normalize(source_surface) == _normalize(target_surface):
        reasons["identical_endpoint_surface"] += 1
        return None
    return source_surface, target_surface


def _relation_grounding_surfaces(entity: EntityMention, proposed_surface: str) -> list[str]:
    """Return exact identity surfaces eligible to repair relation evidence.

    Structured models occasionally assign the correct inventory ID but normalize
    or abbreviate its local surface differently from noisy OCR. The proposed
    surface remains preferred. Canonical names, identity-validated aliases, and
    ontology-approved discourse surfaces provide conservative fallbacks; the
    resulting quote is still an exact source span and still passes independent
    directed-predicate verification before publication.
    """

    identity_surfaces = [entity.name, *entity.aliases]
    surfaces = [proposed_surface, *identity_surfaces]
    if entity.is_document_context:
        contextual_keys = {_normalize(surface) for surface in entity.contextual_surfaces}
        proposed_is_approved = (
            any(identity_surface_match(proposed_surface, surface) for surface in identity_surfaces)
            or _normalize(proposed_surface) in contextual_keys
        )
        surfaces = [
            *([proposed_surface] if proposed_is_approved else []),
            *identity_surfaces,
            *entity.contextual_surfaces,
        ]
    return list(dict.fromkeys(surface.strip() for surface in surfaces if surface.strip()))


def _grounded_endpoint_surface(
    quote: str,
    entity: EntityMention,
    proposed_surface: str,
) -> str | None:
    """Return the exact quote substring that grounded one endpoint identity.

    Evidence repair may rely on a canonical name, trusted alias, OCR-tolerant
    spelling, or approved document-context surface when the provider's proposed
    label is absent or normalized. Persisting the actual matched substring keeps
    semantic verification aligned with deterministic grounding. Surface priority
    follows the same ordered inventory used to ground the quote; source offsets
    break ties deterministically within one surface.
    """

    for surface in _relation_grounding_surfaces(entity, proposed_surface):
        spans = surface_spans(quote, [surface])
        if spans:
            start, end = spans[0]
            return quote[start:end]
    return None


def _relation_required_surface_groups(
    content: str,
    source: EntityMention,
    source_surface: str,
    target: EntityMention,
    target_surface: str,
) -> tuple[list[list[str]], bool]:
    """Build exact endpoint requirements for one directed relation candidate.

    Ordinary endpoints must occur literally in the bounded evidence span. A focal
    source paper is the sole exception: document provenance already identifies
    the paper whose body is being processed, so an authorial technical statement
    need not repeat its title or a first-person discourse pointer in every
    sentence. The exception applies when no one-to-three-sentence span can ground
    distinct source and target mentions. This includes a task or concept nested
    inside the paper title, where requiring two non-overlapping text spans would
    reject a valid title claim even though provenance independently supplies the
    paper endpoint. The exception remains source-only and never relaxes exact
    grounding for the target. Independent semantic verification must still
    distinguish the paper's own claims from cited or historical background.

    The returned flag makes use of this provenance rule explicit in extraction
    traces. If the paper title or a configured contextual surface does occur, the
    normal two-endpoint grounding contract is retained.
    """

    source_surfaces = _relation_grounding_surfaces(source, source_surface)
    target_surfaces = _relation_grounding_surfaces(target, target_surface)
    explicit_endpoint_span = ground_evidence(
        content,
        "",
        [source_surfaces, target_surfaces],
    )
    implicit_document_source = (
        source.is_document_context and source.type == "PAPER" and explicit_endpoint_span is None
    )
    if implicit_document_source:
        return [target_surfaces], True
    return [source_surfaces, target_surfaces], False


class LlmRelationVerifier:
    """Apply an independent entailment judgment to grounded relation candidates.

    Verification judges the exact directed predicate, not mere co-occurrence.
    Omitted or malformed decisions become insufficient evidence rather than
    silently approving plausible-sounding triples.
    """

    def __init__(self, llm: StructuredCompletionProvider, seed: int = 17) -> None:
        """Store the structured provider and deterministic seed used by verification requests.

        No provider work occurs during construction.
        """

        self.llm = llm
        self.seed = seed

    def verify(
        self,
        window: ExtractionWindow,
        entities: list[EntityMention],
        relations: list[RelationObservation],
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
    ) -> VerificationOutcome:
        """Return one reconciled entailment decision for every candidate relation.

        Empty input avoids an unnecessary provider call. Missing response IDs are
        filled with zero-confidence insufficient decisions so downstream filtering
        remains conservative and total.
        """

        if not relations:
            return VerificationOutcome(
                trace={
                    "stage": "relation_verification",
                    "window_id": window.id,
                    "input_records": 0,
                    "accepted_records": 0,
                }
            )
        request = relation_verification_request(
            window,
            entities,
            relations,
            ontology,
            model=model,
            timeout_seconds=timeout_seconds,
            seed=self.seed,
        )
        completion = self.llm.complete_structured(request)
        reasons: Counter[str] = Counter()
        _raw_decisions, candidates = _validated_records(
            completion.payload,
            "decisions",
            VerificationCandidate,
            reasons,
        )
        candidates_by_id = {candidate.relation_id: candidate for candidate in candidates}
        decisions: list[VerificationDecision] = []
        for relation in relations:
            candidate = candidates_by_id.get(relation.id)
            if candidate is None:
                decisions.append(
                    VerificationDecision(
                        relation_id=relation.id,
                        verdict="insufficient",
                        confidence=0.0,
                        explanation="The verifier omitted this relation.",
                    )
                )
                continue
            decisions.append(VerificationDecision.model_validate(candidate.model_dump()))
        return VerificationOutcome(
            decisions=decisions,
            trace={
                "stage": "relation_verification",
                "window_id": window.id,
                "input_records": len(relations),
                "supported_records": sum(decision.verdict == "supported" for decision in decisions),
                "record_actions": dict(sorted(reasons.items())),
                **completion.provider_metadata,
            },
        )


def _record_list(payload: dict[str, Any], key: str) -> list[object]:
    """Read one required top-level record array before sibling validation.

    A wrong top-level shape indicates a broken structured response and fails the
    call; item-level mistakes are handled separately by ``_validated_records``.
    """

    records = payload.get(key)
    if not isinstance(records, list):
        raise ValueError(f"Structured response field '{key}' must be an array")
    return records


def _validated_records[RecordT: BaseModel](
    payload: dict[str, Any],
    key: str,
    model_type: type[RecordT],
    reasons: Counter[str],
) -> tuple[list[object], list[RecordT]]:
    """Validate sibling records independently so one model error remains local.

    Both raw and accepted records are returned so traces can report complete input
    counts while invalid-schema reasons remain visible in run traces.
    """

    raw_records = _record_list(payload, key)
    records: list[RecordT] = []
    for raw_record in raw_records:
        try:
            records.append(model_type.model_validate(raw_record))
        except ValidationError:
            # JSON-schema constrained local models can still violate a numeric
            # bound. The trace records the drop while valid siblings survive.
            reasons["invalid_schema"] += 1
    return raw_records, records


def _types_allowed(
    source_type: str,
    target_type: str,
    definition: RelationTypeDefinition,
) -> bool:
    """Check whether endpoint types satisfy an ontology relation signature.

    An empty source or target type list means the ontology intentionally leaves
    that side unrestricted.
    """

    source_allowed = not definition.source_types or source_type in definition.source_types
    target_allowed = not definition.target_types or target_type in definition.target_types
    return source_allowed and target_allowed


def _orient_relation(
    source: EntityMention,
    target: EntityMention,
    definition: RelationTypeDefinition,
) -> tuple[EntityMention, EntityMention, bool]:
    """Orient endpoints according to ontology types when reversal is unambiguous.

    The boolean records whether a repair occurred; incompatible pairs retain their
    original order and are rejected by the caller's subsequent type check.
    """

    if _types_allowed(source.type, target.type, definition):
        return source, target, False
    if _types_allowed(target.type, source.type, definition):
        return target, source, True
    return source, target, False


def _dedupe_aliases(aliases: list[str], name: str) -> list[str]:
    """Return accepted aliases distinct from the canonical mention name.

    Original provider ordering is preserved deterministically.
    """

    seen = {_normalize(name)}
    result: list[str] = []
    for alias in aliases:
        cleaned = alias.strip()
        key = _normalize(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _document_context_surface_match(name: str, quote: str, aliases: list[str]) -> bool:
    """Validate a focal identity against its exact front-matter evidence quote.

    A model may quote a short phrase containing the canonical name rather than
    returning the name alone, and a trusted source-grounded alias may be the
    surface printed by the paper. Exact surface occurrence and conservative
    spelling identity support those cases. Compact equality remains as a narrow
    OCR repair for isolated spaces such as ``M ETHOD``; semantic similarity is
    intentionally absent so a related title cannot become a document-wide actor.
    """

    compact_name = "".join(character for character in name.casefold() if character.isalnum())
    compact_quote = "".join(character for character in quote.casefold() if character.isalnum())
    return bool(compact_name) and (
        compact_name == compact_quote
        or surface_occurs(quote, [name, *aliases])
        or identity_surface_match(name, quote)
    )


def _entity_description(value: str, definitions: frozenset[str]) -> str:
    """Return a description of this entity, dropping one of a category.

    The extraction request carries every ontology type definition, so a model can
    answer the description field with a definition it was just given. That text
    says nothing about the entity, and it says it identically for every entity it
    is repeated for: it inflates resolution similarity between distinct entities,
    is read as a finding in the explorer, and survives description merging, which
    counts distinct descriptions and so never sees more than one. Recording it as
    absent is the honest reading of what the model returned.

    Every definition is compared, not only the one belonging to this entity's own
    type: a model that reaches for the wrong definition has still described a
    category rather than the entity, and does so — a CONCEPT given the LOCATION
    definition was observed in a real run.
    """

    description = value.strip()
    if not description:
        return ""
    if _definition_key(description) in definitions:
        return ""
    return description


def _ontology_definition_keys(ontology: OntologyProfile) -> frozenset[str]:
    """Return every ontology definition in the form used to recognize one."""

    return frozenset(
        key
        for key in (_definition_key(item.description) for item in ontology.entity_types)
        if key
    )


def _definition_key(value: str) -> str:
    """Reduce text to what distinguishes a definition from a description.

    A definition repeated back with a dropped full stop, a doubled space, or a
    non-breaking space is the same category text and carries no more about the
    entity than the exact copy does.
    """

    return " ".join(value.casefold().split()).rstrip(".。 ")


def _normalize(value: str) -> str:
    """Create a case- and whitespace-insensitive key for local record deduplication.

    Punctuation remains significant at this stage.
    """

    return " ".join(value.casefold().split())
