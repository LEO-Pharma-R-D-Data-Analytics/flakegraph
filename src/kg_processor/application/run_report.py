"""Run-report and graph-metric assembly.

The report is deterministic except for timestamps and summarizes provider,
cache, filter, merge, and quality signals in one object that both local and
Snowflake writers persist.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from kg_processor.application.chunking import compute_ordered_chunk_hash
from kg_processor.application.graph_filter import EntityFilterResult, RelationFilterResult
from kg_processor.application.graph_merge import GraphAssemblyResult
from kg_processor.application.graph_quality import GraphQualityResult
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.graph import (
    Chunk,
    Community,
    CommunityFinding,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from kg_processor.domain.ids import stable_id


@dataclass(frozen=True)
class RunProviders:
    """Provider names included in run reports and graph metrics."""

    ocr: str
    llm: str
    embedding: str
    writer: str
    cache: str


@dataclass(frozen=True)
class RunReportRequest:
    """Complete input needed to build run report and graph metric artifacts."""

    job_id: str
    graph_id: str
    write_scope: Literal["graph_snapshot", "file_batch"]
    file_ids: list[str]
    files_seen: int
    documents_processed: int
    block_rows: list[dict[str, Any]]
    asset_rows: list[dict[str, Any]]
    chunks: list[Chunk]
    extraction: ExtractionResult
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    entity_filter: EntityFilterResult
    relation_filter: RelationFilterResult
    assembly: GraphAssemblyResult
    descriptions_merged: int
    communities: list[Community]
    findings: list[CommunityFinding]
    ocr_cache_hits: int
    extraction_cache_hit: bool
    providers: RunProviders
    embedding_dimension: int
    graph_settings: GraphSettings
    quality_result: GraphQualityResult


@dataclass(frozen=True)
class RunReportArtifacts:
    """Run-report and graph-metric payloads persisted by graph writers."""

    run_report: dict[str, Any]
    graph_metrics: dict[str, Any]


def build_run_report_artifacts(request: RunReportRequest) -> RunReportArtifacts:
    """Build deterministic run report and graph metric payloads for a batch."""

    ordered_chunk_hash = compute_ordered_chunk_hash(request.chunks)
    run_id = stable_id(
        "run",
        request.graph_id,
        request.job_id,
        request.write_scope,
        request.file_ids,
        ordered_chunk_hash,
    )
    return RunReportArtifacts(
        run_report=_run_report(request, run_id, ordered_chunk_hash),
        graph_metrics=_graph_metrics(request),
    )


def _run_report(
    request: RunReportRequest,
    run_id: str,
    ordered_chunk_hash: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "job_id": request.job_id,
        "graph_id": request.graph_id,
        "write_scope": request.write_scope,
        "file_ids": request.file_ids,
        "started_at": datetime.now(UTC).isoformat(),
        "files_seen": request.files_seen,
        "files_processed": request.documents_processed,
        "ordered_chunk_hash": ordered_chunk_hash,
        "chunks_created": len(request.chunks),
        "blocks_created": len(request.block_rows),
        "assets_created": len(request.asset_rows),
        "entities_extracted": len(request.extraction.entities),
        "entities_after_filter": len(request.entities),
        "relations_extracted": len(request.extraction.relations),
        "relations_after_filter": len(request.relations),
        "nodes_created": len(request.assembly.nodes),
        "edges_created": len(request.assembly.edges),
        "evidence_created": len(request.assembly.evidence),
        "descriptions_merged": request.descriptions_merged,
        "communities_created": len(request.communities),
    }


def _graph_metrics(request: RunReportRequest) -> dict[str, Any]:
    return {
        "counts": {
            "files": request.files_seen,
            "chunks": len(request.chunks),
            "blocks": len(request.block_rows),
            "assets": len(request.asset_rows),
            "entities_extracted": len(request.extraction.entities),
            "entities_after_filter": len(request.entities),
            "relations_extracted": len(request.extraction.relations),
            "relations_after_filter": len(request.relations),
            "nodes": len(request.assembly.nodes),
            "edges": len(request.assembly.edges),
            "evidence": len(request.assembly.evidence),
            "communities": len(request.communities),
            "community_findings": len(request.findings),
            "descriptions_merged": request.descriptions_merged,
        },
        "dedupe": {
            "entity_reduction": len(request.extraction.entities) - len(request.assembly.nodes),
            "relation_reduction": (len(request.extraction.relations) - len(request.assembly.edges)),
        },
        "providers": {
            "ocr": request.providers.ocr,
            "llm": request.providers.llm,
            "embedding": request.providers.embedding,
            "writer": request.providers.writer,
            "cache": request.providers.cache,
        },
        "embedding": {
            "dimension": request.embedding_dimension,
        },
        "filtering": {
            "min_entity_confidence": request.graph_settings.min_entity_confidence,
            "min_relation_confidence": request.graph_settings.min_relation_confidence,
            "min_entity_name_length": request.graph_settings.min_entity_name_length,
            "require_relation_endpoint_grounding": (
                request.graph_settings.require_relation_endpoint_grounding
            ),
            "dropped_entities_by_reason": request.entity_filter.dropped_reason_counts(),
            "dropped_relations_by_reason": request.relation_filter.dropped_reason_counts(),
        },
        "extraction": _extraction_metrics(request.extraction),
        "merge": {
            "decision_actions": request.assembly.decision_action_counts(),
            "decision_reasons": request.assembly.decision_reason_counts(),
        },
        "cache": {
            "ocr_hits": request.ocr_cache_hits,
            "ocr_misses": request.files_seen - request.ocr_cache_hits,
            "extraction_hit": request.extraction_cache_hit,
        },
        "quality": request.quality_result.model_dump(mode="json"),
    }


def _extraction_metrics(extraction: ExtractionResult) -> dict[str, Any]:
    metadata = extraction.provider_metadata
    passes = list(_pass_metadata(metadata))
    return {
        "chunk_count": _int_value(metadata.get("chunk_count")),
        "batch_count": _int_value(metadata.get("batch_count")) or len(_batch_metadata(metadata)),
        "pass_count": len(passes),
        "llm_repair_attempts": sum(_int_value(item.get("repair_attempts")) for item in passes),
        "validation_repair_attempts": sum(
            _int_value(item.get("validation_repair_attempts")) for item in passes
        ),
        "evidence_offsets_normalized": sum(
            _int_value(item.get("evidence_offsets_normalized")) for item in passes
        ),
        "dropped_entities": sum(_int_value(item.get("dropped_entities")) for item in passes),
        "dropped_relations": sum(_int_value(item.get("dropped_relations")) for item in passes),
        "dropped_observations_by_reason": _reason_counts(passes),
    }


def _batch_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    batches = metadata.get("batches")
    if not isinstance(batches, list):
        return []
    return [item for item in batches if isinstance(item, dict)]


def _pass_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    passes: list[dict[str, Any]] = []
    for batch in _batch_metadata(metadata):
        raw_passes = batch.get("passes")
        if not isinstance(raw_passes, list):
            continue
        passes.extend(item for item in raw_passes if isinstance(item, dict))
    if passes:
        return passes
    return [metadata] if metadata else []


def _reason_counts(passes: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in passes:
        reasons = item.get("dropped_observations_by_reason")
        if not isinstance(reasons, dict):
            continue
        for reason, count in reasons.items():
            if isinstance(reason, str):
                counts[reason] = counts.get(reason, 0) + _int_value(count)
    return counts


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0
