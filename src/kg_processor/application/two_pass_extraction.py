"""Provider-neutral orchestration for high-precision two-pass extraction."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from kg_processor.application.bibliography import is_reference_text
from kg_processor.application.entity_resolution import resolve_entity_mentions
from kg_processor.application.extraction_grounding import (
    build_surface_text_index,
    indexed_surface_occurs,
    sentence_spans,
    surface_occurs,
    surface_spans,
)
from kg_processor.application.extraction_results import dedupe_extraction_result
from kg_processor.application.extraction_windows import build_extraction_windows
from kg_processor.application.llm_extractors import (
    LlmEntityExtractor,
    LlmRelationExtractor,
    LlmRelationVerifier,
)
from kg_processor.application.progress import error_metadata
from kg_processor.application.prompt_registry import TWO_PASS_PROMPT_REVISION
from kg_processor.application.redaction import redact_sensitive_text
from kg_processor.application.relation_candidates import discover_cue_relation_candidates
from kg_processor.application.window_errors import is_systemic_provider_error
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.extraction import (
    EntityMention,
    ExtractionObservations,
    ExtractionWindow,
    RelationObservation,
    VerificationDecision,
)
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation, ExtractionResult
from kg_processor.domain.ids import stable_id
from kg_processor.domain.ontology import OntologyProfile, normalize_ontology_label
from kg_processor.ports.embeddings import EmbeddingProvider, EmbedOptions
from kg_processor.ports.extraction import EntityExtractor, RelationExtractor, RelationVerifier
from kg_processor.ports.llm import LlmProvider

WindowProgressCallback = Callable[[int, int, int, int], None]
ResolutionProgressCallback = Callable[[int, int, int, int], None]
_MIN_RELATION_ENTITIES = 2
_MIN_COMPOUND_OCCURRENCES = 2
_RELATION_VERIFICATION_BATCH_SIZE = 32


@dataclass(frozen=True)
class _WindowResult:
    """Bundle grounded observations and trace events produced for one window.

    Keeping window output self-contained allows concurrent execution while a
    later deterministic merge restores source order.
    """

    entities: list[EntityMention]
    relations: list[RelationObservation]
    trace: list[dict[str, object]]


def extract_graph_two_pass(
    chunks: list[Chunk],
    llm: LlmProvider,
    embeddings: EmbeddingProvider,
    embed_options: EmbedOptions,
    graph_settings: GraphSettings,
    ontology: OntologyProfile,
    ontology_checksum: str,
    ontology_source: str,
    model: str,
    timeout_seconds: int,
    window_progress: WindowProgressCallback | None = None,
    resolution_progress: ResolutionProgressCallback | None = None,
    entity_extractor: EntityExtractor | None = None,
    relation_extractor: RelationExtractor | None = None,
    relation_verifier: RelationVerifier | None = None,
    document_context_entities: list[EntityMention] | None = None,
    document_context_trace: list[dict[str, Any]] | None = None,
) -> ExtractionResult:
    """Run the complete grounded two-pass extraction and identity-resolution flow.

    Document-scoped windows extract entities first, relations may reference only
    those accepted IDs, optional verification removes unsupported predicates, and
    conservative resolution maps mentions onto canonical names. The result is
    translated to the established extraction contract for downstream assembly.
    """

    observations = extract_graph_observations(
        chunks,
        llm,
        graph_settings.extraction_window_tokens,
        graph_settings.max_chunks_per_llm_call,
        graph_settings,
        ontology,
        model,
        timeout_seconds,
        window_progress,
        entity_extractor=entity_extractor,
        relation_extractor=relation_extractor,
        relation_verifier=relation_verifier,
        document_context_entities=document_context_entities,
    )
    if document_context_trace:
        observations = observations.model_copy(
            update={"trace": [*document_context_trace, *observations.trace]}
        )
    return resolve_extraction_observations(
        observations,
        llm,
        embeddings,
        embed_options,
        graph_settings,
        ontology,
        ontology_checksum,
        ontology_source,
        model,
        timeout_seconds,
        resolution_progress,
    )


def extract_graph_observations(
    chunks: list[Chunk],
    llm: LlmProvider,
    extraction_window_tokens: int,
    max_chunks_per_llm_call: int,
    graph_settings: GraphSettings,
    ontology: OntologyProfile,
    model: str,
    timeout_seconds: int,
    window_progress: WindowProgressCallback | None = None,
    entity_extractor: EntityExtractor | None = None,
    relation_extractor: RelationExtractor | None = None,
    relation_verifier: RelationVerifier | None = None,
    document_context_entities: list[EntityMention] | None = None,
) -> ExtractionObservations:
    """Extract document-wide entities before independently extracting relations.

    Every entity window completes before relation windows begin for its in-process
    shard. This true two-phase boundary lets a relation cite an endpoint discovered
    in another window without serializing model calls inside either phase. Durable
    workers expose the same two phases as separate queue waves.
    """

    entity_observations = extract_entity_observations(
        chunks,
        llm,
        extraction_window_tokens,
        max_chunks_per_llm_call,
        graph_settings,
        ontology,
        model,
        timeout_seconds,
        entity_extractor=entity_extractor,
        document_context_entities=document_context_entities,
        window_progress=window_progress,
    )
    relation_observations = extract_relation_observations(
        chunks,
        entity_observations.entities,
        llm,
        extraction_window_tokens,
        max_chunks_per_llm_call,
        graph_settings,
        ontology,
        model,
        timeout_seconds,
        window_progress=window_progress,
        relation_extractor=relation_extractor,
        relation_verifier=relation_verifier,
    )
    return ExtractionObservations(
        entities=entity_observations.entities,
        relations=relation_observations.relations,
        trace=[*entity_observations.trace, *relation_observations.trace],
        chunk_count=len(chunks),
        window_count=entity_observations.window_count,
    )


def extract_entity_observations(
    chunks: list[Chunk],
    llm: LlmProvider,
    extraction_window_tokens: int,
    max_chunks_per_llm_call: int,
    graph_settings: GraphSettings,
    ontology: OntologyProfile,
    model: str,
    timeout_seconds: int,
    window_progress: WindowProgressCallback | None = None,
    entity_extractor: EntityExtractor | None = None,
    document_context_entities: list[EntityMention] | None = None,
) -> ExtractionObservations:
    """Extract and deduplicate entity mentions without issuing relation calls.

    This function is the first parallel phase shared by local and distributed
    execution. Its output is compact enough to form one immutable per-document
    inventory before the second queue wave starts.
    """

    windows = build_extraction_windows(
        chunks,
        extraction_window_tokens,
        max_chunks_per_llm_call,
    )
    entity_stage = entity_extractor or LlmEntityExtractor(llm, graph_settings.deterministic_seed)
    document_by_chunk = {chunk.id: chunk.document_id for chunk in chunks}
    document_ids = {chunk.document_id for chunk in chunks}
    subjects_by_document: dict[str, list[EntityMention]] = {}
    unassigned_subjects: list[EntityMention] = []
    for subject in document_context_entities or []:
        document_id = document_by_chunk.get(subject.source_chunk_id)
        if document_id is not None:
            subjects_by_document.setdefault(document_id, []).append(subject)
        else:
            unassigned_subjects.append(subject)
    if len(document_ids) == 1 and unassigned_subjects:
        # Distributed extraction sends one bounded window at a time. Its focal
        # entity is grounded in a front-matter chunk that usually belongs to a
        # different window, so a shard-local chunk lookup cannot recover the
        # document id. A single-document shard is unambiguous and can safely
        # inherit that document's reusable context. Multi-document callers keep
        # the strict lookup above to prevent context from leaking across sources.
        only_document_id = next(iter(document_ids))
        subjects_by_document.setdefault(only_document_id, []).extend(unassigned_subjects)
    results = _run_window_stage(
        windows,
        graph_settings,
        lambda window: _extract_window_entities(
            window,
            graph_settings,
            ontology,
            model,
            timeout_seconds,
            entity_stage,
            subjects_by_document.get(window.document_id, []),
        ),
        window_progress,
    )
    return ExtractionObservations(
        entities=_dedupe_mentions([item for result in results for item in result.entities]),
        trace=[event for result in results for event in result.trace],
        chunk_count=len(chunks),
        window_count=len(windows),
    )


def extract_relation_observations(
    chunks: list[Chunk],
    document_entities: list[EntityMention],
    llm: LlmProvider,
    extraction_window_tokens: int,
    max_chunks_per_llm_call: int,
    graph_settings: GraphSettings,
    ontology: OntologyProfile,
    model: str,
    timeout_seconds: int,
    window_progress: WindowProgressCallback | None = None,
    relation_extractor: RelationExtractor | None = None,
    relation_verifier: RelationVerifier | None = None,
) -> ExtractionObservations:
    """Extract relations using the complete entity inventory for each document.

    Relation windows remain mutually independent and can be work-stolen across a
    fleet. The document-wide inventory only broadens valid endpoint choices; source
    grounding and verification continue to constrain every emitted observation.
    """

    windows = build_extraction_windows(
        chunks,
        extraction_window_tokens,
        max_chunks_per_llm_call,
    )
    relation_stage = relation_extractor or LlmRelationExtractor(
        llm, graph_settings.deterministic_seed
    )
    verifier_stage = relation_verifier or LlmRelationVerifier(
        llm, graph_settings.deterministic_seed
    )
    document_by_chunk = {chunk.id: chunk.document_id for chunk in chunks}
    document_ids = {chunk.document_id for chunk in chunks}
    entities_by_document: dict[str, list[EntityMention]] = {}
    unassigned_entities: list[EntityMention] = []
    for entity in document_entities:
        document_id = document_by_chunk.get(entity.source_chunk_id)
        if document_id is None:
            unassigned_entities.append(entity)
        else:
            entities_by_document.setdefault(document_id, []).append(entity)
    if len(document_ids) == 1 and unassigned_entities:
        entities_by_document.setdefault(next(iter(document_ids)), []).extend(unassigned_entities)
    results = _run_window_stage(
        windows,
        graph_settings,
        lambda window: _extract_window_relations(
            window,
            graph_settings,
            ontology,
            model,
            timeout_seconds,
            relation_stage,
            verifier_stage,
            entities_by_document.get(window.document_id, []),
        ),
        window_progress,
    )
    return ExtractionObservations(
        relations=_dedupe_relations([item for result in results for item in result.relations]),
        trace=[event for result in results for event in result.trace],
        chunk_count=len(chunks),
        window_count=len(windows),
    )


def resolve_extraction_observations(
    observations: ExtractionObservations,
    llm: LlmProvider,
    embeddings: EmbeddingProvider,
    embed_options: EmbedOptions,
    graph_settings: GraphSettings,
    ontology: OntologyProfile,
    ontology_checksum: str,
    ontology_source: str,
    model: str,
    timeout_seconds: int,
    resolution_progress: ResolutionProgressCallback | None = None,
) -> ExtractionResult:
    """Resolve combined observation shards and produce the canonical extraction.

    Callers must combine all shards for one graph before invoking this barrier.
    Resolution decisions and provider traces are retained exactly as they are for
    local runs, making distributed execution auditable and semantically equivalent.
    """

    mentions = _dedupe_mentions(observations.entities)
    relations = _dedupe_relations(observations.relations)
    resolution = resolve_entity_mentions(
        mentions,
        enabled=graph_settings.entity_resolution_enabled,
        embeddings=embeddings,
        embed_options=embed_options,
        llm=llm,
        model=model,
        timeout_seconds=timeout_seconds,
        lexical_auto_merge=graph_settings.resolution_lexical_auto_merge,
        embedding_auto_merge=graph_settings.resolution_embedding_auto_merge,
        candidate_threshold=graph_settings.resolution_candidate_threshold,
        embedding_lexical_floor=graph_settings.resolution_embedding_lexical_floor,
        max_candidates_per_mention=graph_settings.resolution_max_candidates_per_mention,
        adjudication_batch_size=graph_settings.resolution_adjudication_batch_size,
        adjudication_parallelism=graph_settings.resolution_parallelism,
        llm_merge_min_confidence=graph_settings.resolution_llm_merge_min_confidence,
        batch_progress=resolution_progress,
        seed=graph_settings.deterministic_seed,
    )
    extraction = _to_extraction_result(mentions, relations, resolution.canonical_name_by_mention_id)
    extraction.provider_metadata = {
        "schema_revision": TWO_PASS_PROMPT_REVISION,
        "strategy": "two_pass",
        "ontology_name": ontology.name,
        "ontology_source": ontology_source,
        "ontology_checksum": ontology_checksum,
        "chunk_count": observations.chunk_count,
        "batch_count": observations.window_count,
        "window_count": observations.window_count,
        "entity_mentions": len(mentions),
        "relation_observations": len(relations),
        "resolution_decisions": [
            decision.model_dump(mode="json") for decision in resolution.decisions
        ],
        "trace": [
            *observations.trace,
            resolution.trace,
        ],
    }
    return dedupe_extraction_result(extraction)


def _run_window_stage(  # noqa: PLR0912
    windows: list[ExtractionWindow],
    settings: GraphSettings,
    operation: Callable[[ExtractionWindow], _WindowResult],
    progress: WindowProgressCallback | None,
) -> list[_WindowResult]:
    """Run one extraction phase concurrently while preserving source order.

    Progress callbacks follow actual completions, but indexed result collection
    ensures provider latency cannot change deduplication or persisted trace order.
    """

    if settings.extraction_parallelism == 1 or len(windows) <= 1:
        results: list[_WindowResult] = []
        failures: list[Exception] = []
        for window in windows:
            try:
                result = operation(window)
            except Exception as exc:
                if is_systemic_provider_error(exc):
                    raise
                failures.append(exc)
                result = _failed_window_result(window, exc)
            results.append(result)
            if progress:
                progress(len(results), len(windows), len(result.entities), len(result.relations))
        if windows and len(failures) == len(windows):
            raise RuntimeError(_all_windows_failed_message(windows, failures)) from failures[0]
        return results

    results_by_index: dict[int, _WindowResult] = {}
    parallel_failures: list[Exception] = []
    max_workers = min(settings.extraction_parallelism, len(windows))
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            executor.submit(operation, window): index for index, window in enumerate(windows)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                if is_systemic_provider_error(exc):
                    for pending in futures:
                        pending.cancel()
                    raise
                parallel_failures.append(exc)
                result = _failed_window_result(windows[index], exc)
            results_by_index[index] = result
            if progress:
                progress(completed, len(windows), len(result.entities), len(result.relations))
    finally:
        executor.shutdown(wait=True)
    if windows and len(parallel_failures) == len(windows):
        raise RuntimeError(
            _all_windows_failed_message(windows, parallel_failures)
        ) from parallel_failures[0]
    return [results_by_index[index] for index in range(len(windows))]


_WINDOW_FAILURE_DETAIL_LIMIT = 300


def _all_windows_failed_message(
    windows: Sequence[ExtractionWindow],
    failures: Sequence[Exception],
) -> str:
    """Name the first cause in the message, not only in the exception chain.

    A worker records this message and discards the chain, so a bare "all
    extraction windows failed" reaches the operator with nothing to act on: the
    provider answered, the pipeline rejected every record, and which of those
    happened is exactly what is missing. Carry a bounded, redacted summary of the
    first failure so the reason survives the trip through the task store.
    """

    first = failures[0]
    detail = redact_sensitive_text(str(first)).strip().replace("\n", " ")
    if len(detail) > _WINDOW_FAILURE_DETAIL_LIMIT:
        detail = detail[:_WINDOW_FAILURE_DETAIL_LIMIT] + "..."
    # A transport error says only that something refused a connection. Which
    # endpoint refused it is the entire question, and the exception carries it.
    target = _failed_request_target(first)
    location = f" while calling {target}" if target else ""
    return (
        f"all {len(windows)} extraction windows failed; "
        f"first cause{location}: {type(first).__name__}: {detail or '<no message>'}"
    )


def _failed_request_target(exc: BaseException) -> str:
    """Return the scheme, host, and path an HTTP failure was aimed at.

    Credentials and query strings are dropped: this ends up in a task record
    that operators read casually, and the useful part is the destination.
    """

    request = getattr(exc, "request", None)
    url = getattr(request, "url", None)
    if url is None:
        return ""
    host = getattr(url, "host", "") or ""
    port = getattr(url, "port", None)
    authority = f"{host}:{port}" if port else host
    return f"{getattr(url, 'scheme', '')}://{authority}{getattr(url, 'path', '')}"


def _failed_window_result(window: ExtractionWindow, exc: Exception) -> _WindowResult:
    """Contain one provider failure while retaining a redacted audit event."""

    return _WindowResult(
        entities=[],
        relations=[],
        trace=[
            {
                "stage": "window_error",
                "window_id": window.id,
                "document_id": window.document_id,
                "error": error_metadata(exc),
            }
        ],
    )


def _extract_window_entities(
    window: ExtractionWindow,
    settings: GraphSettings,
    ontology: OntologyProfile,
    model: str,
    timeout_seconds: int,
    entity_extractor: EntityExtractor,
    document_context_entities: list[EntityMention],
) -> _WindowResult:
    """Run entity extraction and completeness audits for one bounded window.

    Gleaning audits every substantive chunk instead of assuming that one accepted
    record makes a chunk complete. It stops on no output or saturation.
    """

    trace: list[dict[str, object]] = []

    if is_reference_only_window(window):
        return _WindowResult(
            entities=document_context_entities,
            relations=[],
            trace=[
                {
                    "stage": "reference_filter",
                    "window_id": window.id,
                    "skipped": True,
                    "reason": "bibliography_only",
                }
            ],
        )

    entity_outcome = entity_extractor.extract(
        window,
        ontology,
        model=model,
        timeout_seconds=timeout_seconds,
        max_entities=settings.max_entities_per_batch,
    )
    local_entities = entity_outcome.entities
    trace.append(entity_outcome.trace)
    for pass_index in range(1, settings.gleaning_max_passes + 1):
        audit_window = _entity_audit_window(
            window,
            settings.gleaning_min_uncovered_tokens,
            pass_index,
        )
        if audit_window is None:
            break
        try:
            entity_gleaned = entity_extractor.extract(
                audit_window,
                ontology,
                model=model,
                timeout_seconds=timeout_seconds,
                max_entities=settings.max_entities_per_batch,
                previous_entities=local_entities,
            )
        except Exception as exc:
            if is_systemic_provider_error(exc):
                raise
            trace.append(
                {
                    "stage": "entity_gleaning_error",
                    "window_id": window.id,
                    "gleaning_pass": pass_index,
                    "error": error_metadata(exc),
                }
            )
            break
        trace.append({**entity_gleaned.trace, "gleaning_pass": pass_index})
        if not entity_gleaned.entities:
            break
        local_entities = _dedupe_mentions([*local_entities, *entity_gleaned.entities])
        if len(entity_gleaned.entities) < settings.gleaning_saturation_threshold:
            break

    cue_entity_audit = _cue_entity_audit_window(window, local_entities, ontology)
    if cue_entity_audit is not None:
        try:
            entity_audited = entity_extractor.extract(
                cue_entity_audit,
                ontology,
                model=model,
                timeout_seconds=timeout_seconds,
                max_entities=settings.max_entities_per_batch,
                previous_entities=local_entities,
            )
        except Exception as exc:
            if is_systemic_provider_error(exc):
                raise
            trace.append(
                {
                    "stage": "cue_entity_audit_error",
                    "window_id": window.id,
                    "error": error_metadata(exc),
                }
            )
        else:
            trace.append({**entity_audited.trace, "cue_entity_audit": True})
            local_entities = _dedupe_mentions([*local_entities, *entity_audited.entities])

    entities = _dedupe_mentions([*document_context_entities, *local_entities])
    return _WindowResult(entities=entities, relations=[], trace=trace)


def _extract_window_relations(  # noqa: PLR0912
    window: ExtractionWindow,
    settings: GraphSettings,
    ontology: OntologyProfile,
    model: str,
    timeout_seconds: int,
    relation_extractor: RelationExtractor,
    verifier: RelationVerifier,
    entities: list[EntityMention],
) -> _WindowResult:
    """Extract and verify relations against locally groundable document identities."""

    if is_reference_only_window(window):
        return _WindowResult(entities=[], relations=[], trace=[])
    trace: list[dict[str, object]] = []

    # Compact repeated observations before selecting identities whose name, alias,
    # or approved context surface occurs in this window. An abbreviation learned
    # elsewhere remains usable, while impossible-to-ground remote endpoints never
    # consume prompt space or endpoint-pair enumeration.
    relation_entities = _relation_inventory_for_window(entities, window)
    trace.append(
        {
            "stage": "relation_inventory",
            "window_id": window.id,
            "observation_records": len(entities),
            "identity_records": len(relation_entities),
            "locally_grounded_records": len(relation_entities),
        }
    )

    relation_outcome = relation_extractor.extract(
        window,
        relation_entities,
        ontology,
        model=model,
        timeout_seconds=timeout_seconds,
        max_relations=settings.max_relations_per_batch,
    )
    relations = relation_outcome.relations
    trace.append(relation_outcome.trace)
    for pass_index in range(1, settings.gleaning_max_passes + 1):
        audit_window = _relation_audit_window(
            window,
            entities,
            settings.gleaning_min_uncovered_tokens,
            pass_index,
        )
        if audit_window is None:
            break
        audit_entities = _relation_completion_inventory(
            audit_window,
            _relation_inventory_for_window(relation_entities, audit_window),
            ontology,
            relations,
        )
        try:
            relation_gleaned = relation_extractor.extract(
                audit_window,
                audit_entities,
                ontology,
                model=model,
                timeout_seconds=timeout_seconds,
                max_relations=settings.max_relations_per_batch,
                previous_relations=relations,
            )
        except Exception as exc:
            if is_systemic_provider_error(exc):
                raise
            trace.append(
                {
                    "stage": "relation_gleaning_error",
                    "window_id": window.id,
                    "gleaning_pass": pass_index,
                    "error": error_metadata(exc),
                }
            )
            break
        trace.append({**relation_gleaned.trace, "gleaning_pass": pass_index})
        if not relation_gleaned.relations:
            break
        relations = _dedupe_relations([*relations, *relation_gleaned.relations])
        if len(relation_gleaned.relations) < settings.gleaning_saturation_threshold:
            break

    cue_candidates = discover_cue_relation_candidates(window, relation_entities, ontology)
    relations_before_cue_audit = len(relations)
    relations = _dedupe_relations([*relations, *cue_candidates])
    trace.append(
        {
            "stage": "cue_candidate_generation",
            "window_id": window.id,
            "candidate_records": len(cue_candidates),
            "additional_records": len(relations) - relations_before_cue_audit,
            "direct_evidence_records": sum(
                not relation.verification_required for relation in cue_candidates
            ),
        }
    )

    relations_requiring_verification = [
        relation for relation in relations if relation.verification_required
    ]
    if settings.verify_relations and relations_requiring_verification:
        decisions: dict[str, VerificationDecision] = {}
        for batch_index, start in enumerate(
            range(0, len(relations_requiring_verification), _RELATION_VERIFICATION_BATCH_SIZE),
            start=1,
        ):
            batch = relations_requiring_verification[
                start : start + _RELATION_VERIFICATION_BATCH_SIZE
            ]
            try:
                verification = verifier.verify(
                    window,
                    relation_entities,
                    batch,
                    ontology,
                    model=model,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                if is_systemic_provider_error(exc):
                    raise
                trace.append(
                    {
                        "stage": "relation_verification_error",
                        "window_id": window.id,
                        "verification_batch": batch_index,
                        "verification_batch_size": len(batch),
                        "error": error_metadata(exc),
                    }
                )
                continue
            trace.append(
                {
                    **verification.trace,
                    "verification_batch": batch_index,
                    "verification_batch_size": len(batch),
                }
            )
            decisions.update(
                {decision.relation_id: decision for decision in verification.decisions}
            )
        verified: list[RelationObservation] = []
        for relation in relations:
            if not relation.verification_required:
                verified.append(relation)
                continue
            decision = decisions.get(relation.id)
            if (
                decision is not None
                and decision.verdict == "supported"
                and decision.confidence >= settings.verification_min_confidence
            ):
                verified.append(
                    relation.model_copy(
                        update={"confidence": min(relation.confidence, decision.confidence)}
                    )
                )
        relations = verified
    return _WindowResult(entities=[], relations=relations, trace=trace)


def is_reference_only_window(window: ExtractionWindow) -> bool:
    """Identify bibliography-only windows without interpreting source claims.

    PDF chunks can begin mid-bibliography after a heading fell into the previous
    window. Requiring several bracketed or author-style numbered entries avoids
    classifying ordinary prose with one citation as references. Skipping these
    windows prevents citation metadata from becoming entities or unsupported
    ``BUILDS_ON`` relations and avoids unnecessary provider calls.
    """

    return is_reference_text("\n".join(chunk.content for chunk in window.chunks))


def _entity_audit_window(
    window: ExtractionWindow,
    minimum_tokens: int,
    pass_index: int,
) -> ExtractionWindow | None:
    """Build a gleaning subwindow containing every substantive source chunk.

    A chunk that yielded one entity may still contain several missed entities, so
    record presence is not a valid completeness signal. Very small chunks remain
    excluded because they rarely justify another provider call and can encourage
    duplicate low-context observations.
    """

    chunks = [chunk for chunk in window.chunks if chunk.token_count >= minimum_tokens]
    return _subwindow(window, chunks, "entity_glean", pass_index)


def _cue_entity_audit_window(
    window: ExtractionWindow,
    entities: list[EntityMention],
    ontology: OntologyProfile,
) -> ExtractionWindow | None:
    """Focus one entity audit on relation sentences with incomplete endpoints.

    Relation extraction cannot recover a fact whose compound subject was shortened
    to a different known entity. This audit selects only cue-bearing sentences with
    fewer than two grounded entities or a grounded surface that appears to continue
    into a longer lowercase noun phrase. Original chunk IDs are retained so any new
    mention still points to valid source provenance.
    """

    selected_by_chunk: dict[str, list[str]] = {}
    window_text = "\n".join(chunk.content for chunk in window.chunks)
    for chunk in window.chunks:
        for sentence in sentence_spans(chunk.content):
            cue_starts = [
                match.start()
                for definition in ontology.relation_types
                for cue in definition.evidence_cues
                for match in re.finditer(cue, sentence.quote, flags=re.IGNORECASE)
            ]
            if not cue_starts:
                continue
            grounded_entities = [
                entity
                for entity in entities
                if any(
                    surface_spans(sentence.quote, [surface])
                    for surface in [entity.name, *entity.aliases]
                )
            ]
            occurrences = [
                (start, end)
                for entity in grounded_entities
                for surface in [entity.name, *entity.aliases]
                for start, end in surface_spans(sentence.quote, [surface])
            ]
            incomplete_endpoints = (
                len({entity.id for entity in grounded_entities}) < _MIN_RELATION_ENTITIES
            )
            partial_endpoint = any(
                _surface_continues_as_repeated_noun_phrase(
                    sentence.quote,
                    start,
                    end,
                    cue_starts,
                    window_text,
                )
                for start, end in occurrences
            )
            if incomplete_endpoints or partial_endpoint:
                selected_by_chunk.setdefault(chunk.id, []).append(sentence.quote)

    # Keep the original chunk content and offsets. A prior targeted-sentence copy
    # retained the source chunk ID while resetting text coordinates, which made
    # otherwise valid quotes point at unrelated source spans after publication.
    chunks = [chunk for chunk in window.chunks if selected_by_chunk.get(chunk.id)]
    if not chunks:
        return None
    return ExtractionWindow(
        id=stable_id("cue_entity_audit_window", window.id),
        document_id=window.document_id,
        chunks=chunks,
        token_count=sum(chunk.token_count for chunk in chunks),
    )


def _surface_continues_as_repeated_noun_phrase(
    sentence: str,
    surface_start: int,
    surface_end: int,
    cue_starts: list[int],
    window_text: str,
) -> bool:
    """Detect a recurring compound that extends a known relation subject surface."""

    following = re.match(r"\s+([a-z][\w-]*)", sentence[surface_end:])
    if following is None:
        return False
    token = following.group(1).casefold()
    grammatical_words = {
        "are",
        "as",
        "authored",
        "became",
        "combines",
        "developed",
        "documented",
        "emphasizes",
        "founded",
        "governed",
        "had",
        "has",
        "have",
        "in",
        "includes",
        "influenced",
        "is",
        "of",
        "occurred",
        "originated",
        "popularized",
        "practiced",
        "preceded",
        "presented",
        "recognized",
        "regulated",
        "taught",
        "uses",
        "was",
        "were",
    }
    next_start = surface_end + following.start(1)
    if token in grammatical_words or not any(next_start < cue for cue in cue_starts):
        return False
    compound = sentence[surface_start:surface_end] + following.group(0)
    repeated_compound = (
        len(
            re.findall(
                rf"(?<!\w){re.escape(compound)}(?!\w)",
                window_text,
                flags=re.IGNORECASE,
            )
        )
        >= _MIN_COMPOUND_OCCURRENCES
    )
    repeated_head_pattern = (
        rf"(?<!\w)(?:a|an|the)\s+{re.escape(token)}(?!\w)"
        + r"|['\u2019]s\s+"
        + re.escape(token)
        + r"(?!\w)"
    )
    repeated_head = re.search(
        repeated_head_pattern,
        window_text,
        flags=re.IGNORECASE,
    )
    return repeated_compound or repeated_head is not None


def _relation_audit_window(
    window: ExtractionWindow,
    entities: list[EntityMention],
    minimum_tokens: int,
    pass_index: int,
) -> ExtractionWindow | None:
    """Build a relation audit from chunks containing two grounded entity mentions.

    Existing relations do not make a chunk complete: one sentence can state many
    independent facts. Requiring two text-grounded entities avoids provider calls
    for chunks that cannot produce a valid non-self relation, while occurrence
    matching includes entities first recognized in an overlapping neighbor chunk.
    """

    chunks = [
        chunk
        for chunk in window.chunks
        if chunk.token_count >= minimum_tokens
        and len(_entities_grounded_in_text(entities, chunk.content)) >= _MIN_RELATION_ENTITIES
    ]
    return _subwindow(window, chunks, "relation_glean", pass_index)


def _entities_grounded_in_window(
    entities: list[EntityMention], window: ExtractionWindow
) -> list[EntityMention]:
    """Return inventory entries whose names or aliases occur in an audit window.

    Selecting by source provenance alone loses endpoints that were first extracted
    from an overlapping chunk but are also stated in the audited text. Exact surface
    occurrence keeps the broader inventory grounded without exposing unrelated
    document entities to the relation model.
    """

    return [
        entity
        for entity in entities
        if entity.is_document_context
        or any(
            surface_occurs(chunk.content, [entity.name, *entity.aliases]) for chunk in window.chunks
        )
    ]


def _relation_inventory_for_window(
    entities: list[EntityMention], window: ExtractionWindow
) -> list[EntityMention]:
    """Return the compact set of identities groundable in one relation window.

    Entity extraction deliberately retains repeated source observations for
    provenance and later identity resolution. Relation generation needs identity
    choices rather than every source span, so exposing all repeated mentions adds
    tokens and lets the provider select among multiple IDs with the same name.

    Exact normalized name and ontology type are a conservative identity key. The
    current window's mention supplies the endpoint ID when one exists, preserving
    repeated evidence from different chunks; otherwise the first stable mention is
    used. Aliases, document-context surfaces, and the context flag are unioned
    across observations. An identity established in a different window remains
    available when its name, alias, or approved context surface occurs here.
    Identities with no local surface are omitted because grounding cannot accept
    them; sending them only adds prompt tokens and quadratic endpoint pairs.
    """

    grouped: dict[tuple[str, str], list[EntityMention]] = {}
    order: list[tuple[str, str]] = []
    for entity in entities:
        key = (_normalize(entity.name), entity.type)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(entity)

    window_chunk_ids = {chunk.id for chunk in window.chunks}
    compacted: dict[tuple[str, str], EntityMention] = {}
    for key in order:
        observations = grouped[key]
        representative = next(
            (entity for entity in observations if entity.source_chunk_id in window_chunk_ids),
            observations[0],
        )
        aliases = list(dict.fromkeys(alias for entity in observations for alias in entity.aliases))
        contextual_surfaces = list(
            dict.fromkeys(
                surface for entity in observations for surface in entity.contextual_surfaces
            )
        )
        compacted[key] = representative.model_copy(
            update={
                "aliases": aliases,
                "is_document_context": any(entity.is_document_context for entity in observations),
                "contextual_surfaces": contextual_surfaces,
            }
        )

    inventory = [compacted[key] for key in order]
    return _entities_grounded_in_window(inventory, window)


def _relation_completion_inventory(
    window: ExtractionWindow,
    entities: list[EntityMention],
    ontology: OntologyProfile,
    previous_relations: list[RelationObservation],
) -> list[EntityMention]:
    """Focus a gleaning pass on uncovered cue-compatible endpoint pairs.

    The initial relation pass considers every locally groundable pair. A second
    broad request tends to repeat salient facts, so this completion inventory keeps
    only endpoints participating in a still-uncovered typed pair for which the
    source chunk contains one of that predicate's configured evidence cues. The
    LLM still receives the original chunk and must emit exact endpoint surfaces;
    ordinary grounding and semantic verification remain unchanged.

    Definitions without lexical cues remain eligible in every chunk. Otherwise,
    the presence of an unrelated cue-backed pair could exclude all endpoints for
    layout-defined relations such as publication authorship or affiliation. If no
    guided pair can be formed, the complete local inventory is returned. This
    preserves semantic coverage while keeping common completion requests smaller.
    """

    covered_targets: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    covered_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for relation in previous_relations:
        covered_targets[
            (relation.source_chunk_id, relation.relation_type, relation.source_entity_id)
        ].add(relation.target_entity_id)
        covered_sources[
            (relation.source_chunk_id, relation.relation_type, relation.target_entity_id)
        ].add(relation.source_entity_id)
    selected_ids: set[str] = set()
    for chunk in window.chunks:
        local_entities = _entities_grounded_in_text(entities, chunk.content)
        cue_definitions = [
            definition
            for definition in ontology.relation_types
            if not definition.evidence_cues
            or any(
                re.search(cue, chunk.content, flags=re.IGNORECASE)
                for cue in definition.evidence_cues
            )
        ]
        for definition in cue_definitions:
            sources = [
                entity
                for entity in local_entities
                if not definition.source_types or entity.type in definition.source_types
            ]
            targets = [
                entity
                for entity in local_entities
                if not definition.target_types or entity.type in definition.target_types
            ]
            source_ids = {entity.id for entity in sources}
            target_ids = {entity.id for entity in targets}
            for source in sources:
                excluded_self = not definition.allow_self_loop and source.id in target_ids
                candidate_count = len(target_ids) - int(excluded_self)
                blocked_count = sum(
                    target_id in target_ids
                    and (definition.allow_self_loop or target_id != source.id)
                    for target_id in covered_targets[(chunk.id, definition.name, source.id)]
                )
                if candidate_count > blocked_count:
                    selected_ids.add(source.id)
            for target in targets:
                excluded_self = not definition.allow_self_loop and target.id in source_ids
                candidate_count = len(source_ids) - int(excluded_self)
                blocked_count = sum(
                    source_id in source_ids
                    and (definition.allow_self_loop or source_id != target.id)
                    for source_id in covered_sources[(chunk.id, definition.name, target.id)]
                )
                if candidate_count > blocked_count:
                    selected_ids.add(target.id)

    selected = [entity for entity in entities if entity.id in selected_ids]
    return selected if len(selected) >= _MIN_RELATION_ENTITIES else entities


def _entities_grounded_in_text(entities: list[EntityMention], text: str) -> list[EntityMention]:
    """Return mentions grounded by identity or document-context surfaces in text."""

    index = build_surface_text_index(text)
    return [
        entity
        for entity in entities
        if indexed_surface_occurs(index, [entity.name, *entity.aliases])
        or (
            entity.is_document_context and indexed_surface_occurs(index, entity.contextual_surfaces)
        )
    ]


def _subwindow(
    parent: ExtractionWindow,
    chunks: list[Chunk],
    stage: str,
    pass_index: int,
) -> ExtractionWindow | None:
    """Create a stable child window for one targeted gleaning pass.

    Empty chunk selections return ``None`` so orchestration can terminate without
    constructing or tracing a meaningless provider request.
    """

    if not chunks:
        return None
    return ExtractionWindow(
        id=stable_id("extraction_subwindow", parent.id, stage, pass_index),
        document_id=parent.document_id,
        chunks=chunks,
        token_count=sum(chunk.token_count for chunk in chunks),
    )


def _to_extraction_result(
    mentions: list[EntityMention],
    relations: list[RelationObservation],
    canonical_names: dict[str, str],
) -> ExtractionResult:
    """Translate grounded mention observations into the established graph contract.

    Resolution-selected names are applied to entities and relation endpoints while
    exact quotes, offsets, confidence, and non-canonical spellings remain available
    for downstream evidence and alias assembly.
    """

    mentions_by_id = {mention.id: mention for mention in mentions}
    entities = [
        ExtractedEntity(
            name=canonical_names[mention.id],
            type=mention.type,
            description=mention.description,
            source_chunk_id=mention.source_chunk_id,
            quote=mention.quote,
            start_offset=mention.start_offset,
            end_offset=mention.end_offset,
            confidence=mention.confidence,
            aliases=_canonical_aliases(mention, canonical_names[mention.id]),
        )
        for mention in mentions
    ]
    extracted_relations: list[ExtractedRelation] = []
    for relation in relations:
        source = mentions_by_id.get(relation.source_entity_id)
        target = mentions_by_id.get(relation.target_entity_id)
        if source is None or target is None:
            continue
        source_name = canonical_names[source.id]
        target_name = canonical_names[target.id]
        if source.id == target.id:
            continue
        extracted_relations.append(
            ExtractedRelation(
                source_name=source_name,
                target_name=target_name,
                source_surface=relation.source_surface,
                target_surface=relation.target_surface,
                source_type=source.type,
                target_type=target.type,
                relation_type=relation.relation_type,
                description=relation.description,
                source_chunk_id=relation.source_chunk_id,
                quote=relation.quote,
                start_offset=relation.start_offset,
                end_offset=relation.end_offset,
                weight=max(0.1, relation.confidence),
                confidence=relation.confidence,
            )
        )
    return ExtractionResult(entities=entities, relations=extracted_relations)


def _canonical_aliases(mention: EntityMention, canonical_name: str) -> list[str]:
    """Combine accepted aliases with a non-canonical mention spelling.

    Canonical-equivalent duplicates are removed in deterministic order.
    """

    values = [*mention.aliases]
    if _normalize(mention.name) != _normalize(canonical_name):
        values.append(mention.name)
    seen = {_normalize(canonical_name)}
    result: list[str] = []
    for value in values:
        key = _normalize(value)
        if value.strip() and key not in seen:
            seen.add(key)
            result.append(value.strip())
    return result


def _dedupe_mentions(mentions: list[EntityMention]) -> list[EntityMention]:
    """Remove repeated grounded mentions without collapsing distinct source spans.

    The first stable observation is retained.
    """

    seen: set[tuple[str, str, str, int | None]] = set()
    result: list[EntityMention] = []
    for mention in mentions:
        key = (
            _normalize(mention.name),
            mention.type,
            mention.source_chunk_id,
            mention.start_offset,
        )
        if key not in seen:
            seen.add(key)
            result.append(mention)
    return result


def _dedupe_relations(relations: list[RelationObservation]) -> list[RelationObservation]:
    """Remove repeated directed relation observations within the same source chunk.

    Cross-chunk support remains independent evidence.
    """

    indexes: dict[tuple[str, str, str, str], int] = {}
    result: list[RelationObservation] = []
    for relation in relations:
        key = (
            relation.source_entity_id,
            relation.target_entity_id,
            relation.relation_type,
            relation.source_chunk_id,
        )
        if key not in indexes:
            indexes[key] = len(result)
            result.append(relation)
            continue
        existing_index = indexes[key]
        existing = result[existing_index]
        if existing.verification_required and not relation.verification_required:
            # Deterministic cue support must survive when an LLM independently
            # emitted the same triple first.
            result[existing_index] = existing.model_copy(
                update={
                    "evidence_method": "ontology_cue",
                    "verification_required": False,
                    "confidence": max(existing.confidence, relation.confidence),
                }
            )
    return result


def _normalize(value: str) -> str:
    """Create a stable case-, space-, and hyphen-insensitive key for names and aliases.

    Grouping and deduplication share the ontology label rule so one spelling
    variant is never treated as two identities here and one identity there.
    """

    return normalize_ontology_label(value)
