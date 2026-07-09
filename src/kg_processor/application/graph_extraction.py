"""Provider-neutral GraphRAG extraction orchestration.

This module owns batching, repair, semantic validation, and bounded gleaning.
LLM adapters only execute prompts; the application layer decides which
observations are accepted and records pass-level metadata for traceability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kg_processor.application.extraction_schema import (
    EXTRACTION_SCHEMA_REVISION,
    dedupe_extraction_result,
    validate_extraction_result,
)
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation, ExtractionResult
from kg_processor.domain.ids import stable_id
from kg_processor.ports.llm import GraphRepairRequest, LlmOptions, LlmProvider


@dataclass(frozen=True)
class ExtractionBatchCheckpoint:
    """Cache lookup result for one extraction chunk batch."""

    cache_id: str
    result: ExtractionResult | None = None


BatchCheckpointGetter = Callable[[list[Chunk]], ExtractionBatchCheckpoint]
BatchCheckpointWriter = Callable[[list[Chunk], ExtractionResult], str]


def extract_graph_from_chunks(
    chunks: list[Chunk],
    llm: LlmProvider,
    graph_settings: GraphSettings,
    model: str,
    timeout_seconds: int,
    batch_checkpoint_get: BatchCheckpointGetter | None = None,
    batch_checkpoint_put: BatchCheckpointWriter | None = None,
) -> ExtractionResult:
    """Extract, repair, validate, glean, cache, and merge chunk-batch results."""

    # Keep initial extraction and gleaning in the same code path so provider
    # adapters do not need to know whether a call is first-pass or recall-pass.
    batch_results: list[ExtractionResult] = []
    for batch_index, batch in enumerate(_batches(chunks, graph_settings.max_chunks_per_llm_call)):
        checkpoint = batch_checkpoint_get(batch) if batch_checkpoint_get is not None else None
        if checkpoint is not None and checkpoint.result is not None:
            batch_results.append(
                _with_checkpoint_metadata(
                    checkpoint.result,
                    cache_id=checkpoint.cache_id,
                    cache_hit=True,
                )
            )
            continue
        batch_result = _run_extraction_pass(
            batch,
            llm,
            graph_settings,
            model,
            timeout_seconds,
            pass_index=0,
            previous_result=None,
        )
        pass_results = [batch_result]
        current = batch_result
        # Gleaning is bounded and recall-oriented: low first-pass yield is a
        # reason to try another pass, while duplicate/no-new output is the stop.
        for pass_index in range(1, graph_settings.gleaning_max_passes + 1):
            gleaned = _run_extraction_pass(
                batch,
                llm,
                graph_settings,
                model,
                timeout_seconds,
                pass_index=pass_index,
                previous_result=current,
            )
            new_records = _new_records(gleaned, current)
            if not new_records.entities and not new_records.relations:
                break
            pass_results.append(new_records)
            current = dedupe_extraction_result(_merge_records([current, new_records]))
            if _record_count(new_records) < graph_settings.gleaning_saturation_threshold:
                break
        batch_result = _merge_batch_passes(pass_results, batch, batch_index)
        if batch_checkpoint_put is not None:
            cache_id = batch_checkpoint_put(batch, batch_result)
            batch_result = _with_checkpoint_metadata(
                batch_result,
                cache_id=cache_id,
                cache_hit=False,
            )
        batch_results.append(batch_result)
    merged = dedupe_extraction_result(_merge_batches(batch_results, chunk_count=len(chunks)))
    return merged


def _batches(values: list[Chunk], size: int) -> list[list[Chunk]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _run_extraction_pass(
    batch: list[Chunk],
    llm: LlmProvider,
    graph_settings: GraphSettings,
    model: str,
    timeout_seconds: int,
    pass_index: int,
    previous_result: ExtractionResult | None,
) -> ExtractionResult:
    options = LlmOptions(
        model=model,
        entity_types=graph_settings.entity_types,
        relation_types=graph_settings.relation_types,
        max_entities_per_batch=graph_settings.max_entities_per_batch,
        max_relations_per_batch=graph_settings.max_relations_per_batch,
        min_entity_confidence=graph_settings.min_entity_confidence,
        min_relation_confidence=graph_settings.min_relation_confidence,
        timeout_seconds=timeout_seconds,
        extraction_pass=pass_index,
        previous_result=previous_result,
    )
    result = llm.extract_graph(batch, options)
    return dedupe_extraction_result(
        _validate_or_repair_extraction_result(result, batch, llm, options, pass_index)
    )


def _validate_or_repair_extraction_result(
    result: ExtractionResult,
    batch: list[Chunk],
    llm: LlmProvider,
    options: LlmOptions,
    pass_index: int,
) -> ExtractionResult:
    try:
        validated = validate_extraction_result(result, batch, options, pass_index)
    except ValueError as exc:
        repaired = llm.repair_graph_extraction(
            GraphRepairRequest(
                chunks=batch,
                options=options,
                invalid_result=result,
                validation_error=str(exc),
            )
        )
        try:
            validated = validate_extraction_result(repaired, batch, options, pass_index)
            repair_failed_error = None
        except ValueError as repair_exc:
            # The second validation pass is still strict except for one case:
            # ungrounded repaired quotes are dropped and counted so a single bad
            # relation does not discard an otherwise usable batch.
            validated = validate_extraction_result(
                repaired,
                batch,
                options,
                pass_index,
                drop_ungrounded_observations=True,
            )
            repair_failed_error = str(repair_exc)
        metadata = dict(validated.provider_metadata)
        metadata.update(
            {
                "validation_repair_attempts": 1,
                "validation_repair_error": str(exc),
                "validation_repair_failed_error": repair_failed_error,
                "pass_index": pass_index,
                "pre_repair_provider_metadata": result.provider_metadata,
            }
        )
        return ExtractionResult(
            entities=validated.entities,
            relations=validated.relations,
            provider_metadata=metadata,
        )
    metadata = dict(validated.provider_metadata)
    metadata["validation_repair_attempts"] = 0
    metadata["pass_index"] = pass_index
    return ExtractionResult(
        entities=validated.entities,
        relations=validated.relations,
        provider_metadata=metadata,
    )


def _merge_batch_passes(
    results: list[ExtractionResult],
    batch: list[Chunk],
    batch_index: int,
) -> ExtractionResult:
    merged = _merge_records(results)
    chunk_ids = [chunk.id for chunk in batch]
    merged.provider_metadata = {
        "schema_revision": EXTRACTION_SCHEMA_REVISION,
        "batch_id": stable_id("extraction_batch", *chunk_ids),
        "batch_index": batch_index,
        "chunk_ids": chunk_ids,
        "passes": [result.provider_metadata for result in results],
    }
    return merged


def _merge_batches(results: list[ExtractionResult], chunk_count: int) -> ExtractionResult:
    merged = _merge_records(results)
    merged.provider_metadata = {
        "schema_revision": EXTRACTION_SCHEMA_REVISION,
        "chunk_count": chunk_count,
        "batch_count": len(results),
        "batches": [result.provider_metadata for result in results],
    }
    return merged


def _with_checkpoint_metadata(
    result: ExtractionResult,
    *,
    cache_id: str,
    cache_hit: bool,
) -> ExtractionResult:
    metadata = dict(result.provider_metadata)
    metadata.update({"checkpoint_cache_id": cache_id, "checkpoint_cache_hit": cache_hit})
    return ExtractionResult(
        entities=result.entities,
        relations=result.relations,
        provider_metadata=metadata,
    )


def _merge_records(results: list[ExtractionResult]) -> ExtractionResult:
    merged = ExtractionResult()
    for result in results:
        merged.entities.extend(result.entities)
        merged.relations.extend(result.relations)
    return merged


def _record_count(result: ExtractionResult) -> int:
    return len(result.entities) + len(result.relations)


def _new_records(candidate: ExtractionResult, existing: ExtractionResult) -> ExtractionResult:
    existing_entity_keys = {_entity_key(entity) for entity in existing.entities}
    existing_relation_keys = {_relation_key(relation) for relation in existing.relations}
    return ExtractionResult(
        entities=[
            entity
            for entity in candidate.entities
            if _entity_key(entity) not in existing_entity_keys
        ],
        relations=[
            relation
            for relation in candidate.relations
            if _relation_key(relation) not in existing_relation_keys
        ],
        provider_metadata=candidate.provider_metadata,
    )


def _entity_key(entity: ExtractedEntity) -> tuple[str, str, str]:
    return (_normalize(entity.name), entity.type, entity.source_chunk_id)


def _relation_key(relation: ExtractedRelation) -> tuple[str, str, str, str]:
    return (
        _normalize(relation.source_name),
        _normalize(relation.target_name),
        relation.relation_type.strip().lower(),
        relation.source_chunk_id,
    )


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())
