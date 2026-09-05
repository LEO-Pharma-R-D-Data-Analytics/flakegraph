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
    """Collect provider identities included in run reports and graph metrics.

    Keeping this group typed prevents report construction from mixing provider
    display names or omitting one execution boundary.
    """

    ocr: str
    llm: str
    embedding: str
    writer: str
    cache: str


@dataclass(frozen=True)
class RunReportRequest:
    """Bundle every deterministic input needed for run-report and metric artifacts.

    The request separates report assembly from pipeline orchestration and makes the
    provenance surface straightforward to unit-test without invoking providers.
    """

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
    # Optional so report assembly stays usable by callers that do not meter.
    consumption: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunReportArtifacts:
    """Hold the operational run summary and detailed graph metrics for persistence.

    Writers receive both payloads unchanged, ensuring local and Snowflake outputs
    expose the same review information.
    """

    run_report: dict[str, Any]
    graph_metrics: dict[str, Any]


def build_run_report_artifacts(request: RunReportRequest) -> RunReportArtifacts:
    """Build run-summary and graph-metric payloads from one completed batch.

    Stable run identity is derived from graph/job scope, files, and ordered chunk
    content; only the human-facing start timestamp is intentionally non-deterministic.
    """

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
    """Build the concise operational summary returned by the worker command.

    Counts cover every major transformation so users can understand what survived
    extraction, filtering, merge, enrichment, and persistence at a glance.
    """

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
        "edge_observations_created": len(request.assembly.edge_observations),
        "evidence_created": len(request.assembly.evidence),
        "descriptions_merged": request.descriptions_merged,
        "communities_created": len(request.communities),
    }


def _graph_metrics(request: RunReportRequest) -> dict[str, Any]:
    """Build detailed extraction, filter, merge, cache, provider, and quality metrics.

    This payload favors reviewability and diagnosis over terminal brevity and is
    persisted alongside graph artifacts for later comparison.
    """

    metrics: dict[str, Any] = {
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
            "edge_observations": len(request.assembly.edge_observations),
            "evidence": len(request.assembly.evidence),
            "communities": len(request.communities),
            "community_findings": len(request.findings),
            "descriptions_merged": request.descriptions_merged,
        },
        "dedupe": {
            "entity_reduction": len(request.entities) - len(request.assembly.nodes),
            "relation_reduction": len(request.relations) - len(request.assembly.edges),
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
    if request.consumption is not None:
        # Consumption rides with the metrics so every writer that already
        # persists them — local artifacts and KG_GRAPH_METRICS alike — carries
        # spend without each one learning about it separately.
        metrics["consumption"] = request.consumption
    return metrics


def _extraction_metrics(extraction: ExtractionResult) -> dict[str, Any]:
    """Summarize extraction-stage traces under the stable run-report contract."""

    return _two_pass_extraction_metrics(extraction.provider_metadata)


def _two_pass_extraction_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    """Summarize modular two-pass stage traces and record-level actions.

    Entity, relation, and verifier calls are counted separately, while rejection
    and repair reasons are aggregated across document windows and gleaning passes.
    """

    raw_trace = metadata.get("trace", [])
    trace = [item for item in raw_trace if isinstance(item, dict)]
    entity_events = [item for item in trace if item.get("stage") == "entity_extraction"]
    relation_events = [item for item in trace if item.get("stage") == "relation_extraction"]
    verification_events = [item for item in trace if item.get("stage") == "relation_verification"]
    context_events = [item for item in trace if item.get("stage") == "document_context_extraction"]
    record_actions: dict[str, int] = {}
    # Document-context extraction grounds entities through the same checks and
    # raises the same reasons. Counting only the other two stages reports a
    # cleaner run than actually happened.
    for event in [*entity_events, *context_events, *relation_events]:
        actions = event.get("record_actions", {})
        if not isinstance(actions, dict):
            continue
        for reason, count in actions.items():
            if isinstance(reason, str):
                record_actions[reason] = record_actions.get(reason, 0) + _int_value(count)
    return {
        "strategy": "two_pass",
        "chunk_count": _int_value(metadata.get("chunk_count")),
        "batch_count": _int_value(metadata.get("batch_count")),
        "window_count": _int_value(metadata.get("window_count")),
        "entity_calls": len(entity_events),
        "relation_calls": len(relation_events),
        "verification_calls": len(verification_events),
        "entity_mentions": _int_value(metadata.get("entity_mentions")),
        "relation_observations": _int_value(metadata.get("relation_observations")),
        "record_actions": dict(sorted(record_actions.items())),
    }


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0
