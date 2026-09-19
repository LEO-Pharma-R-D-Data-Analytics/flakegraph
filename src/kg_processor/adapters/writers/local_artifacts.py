"""Atomic local artifact writer.

Local artifacts are not debug leftovers; they mirror the Snowflake logical table
contract and are written as all-or-nothing snapshots so failed runs do not leave
half-updated review output.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from kg_processor.application.redaction import redact_sensitive_data
from kg_processor.domain.graph import (
    Chunk,
    Community,
    CommunityFinding,
    EdgeObservation,
    EntitySource,
    Evidence,
    GraphEdge,
    GraphNode,
    GraphWriteBatch,
)

_BLOCK_COLUMNS = ["id", "graph_id", "file_id", "page_number", "kind", "text", "bbox", "metadata"]
_ASSET_COLUMNS = ["id", "graph_id", "file_id", "kind", "page_number", "uri", "metadata"]
_DOCUMENT_COLUMNS = [
    "id",
    "graph_id",
    "file_id",
    "checksum",
    "source_uri",
    "mime_type",
    "size_bytes",
    "ocr_provider",
]
_PAGE_COLUMNS = [
    "id",
    "graph_id",
    "file_id",
    "page_number",
    "markdown",
    "raw_text",
    "detected_language",
]
_CHUNK_COLUMNS = list(Chunk.model_fields)
_NODE_COLUMNS = list(GraphNode.model_fields)
_EDGE_COLUMNS = list(GraphEdge.model_fields)
_EDGE_OBSERVATION_COLUMNS = list(EdgeObservation.model_fields)
_EVIDENCE_COLUMNS = list(Evidence.model_fields)
_ENTITY_SOURCE_COLUMNS = list(EntitySource.model_fields)
_COMMUNITY_COLUMNS = list(Community.model_fields)
_COMMUNITY_FINDING_COLUMNS = list(CommunityFinding.model_fields)
_ARTIFACT_MANIFEST = ".flakegraph-artifacts.json"
_ARTIFACT_FORMAT = "flakegraph-local-artifacts-v1"
_SNAPSHOT_FILES = frozenset(
    {
        "documents.parquet",
        "pages.parquet",
        "blocks.parquet",
        "assets.parquet",
        "chunks.parquet",
        "nodes.parquet",
        "edges.parquet",
        "edge_observations.parquet",
        "evidence.parquet",
        "entity_sources.parquet",
        "communities.parquet",
        "community_findings.parquet",
        "run_report.json",
        "graph_metrics.json",
        "extraction_trace.jsonl",
    }
)


class LocalArtifactsWriter:
    """Writes Snowflake-shaped graph artifacts to a local directory atomically."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def write(self, batch: GraphWriteBatch) -> None:
        """Write a snapshot without taking ownership of an unrelated directory.

        Empty destinations are safe to claim. Non-empty destinations must either
        contain this writer's ownership manifest or the complete legacy artifact
        contract, which lets upgrades replace old FlakeGraph snapshots without
        allowing a typo in ``output_path`` to delete arbitrary user data.
        """

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        _assert_replaceable_output(self.output_path)
        temp_path = self.output_path.parent / (f".{self.output_path.name}.tmp-{uuid.uuid4().hex}")
        try:
            temp_path.mkdir(parents=False, exist_ok=False)
            _write_snapshot(temp_path, batch)
            _replace_snapshot(temp_path, self.output_path)
        finally:
            if temp_path.exists():
                _remove_path(temp_path)


def _write_snapshot(output_path: Path, batch: GraphWriteBatch) -> None:
    """Materialize every table and diagnostic file in a temporary snapshot.

    The caller publishes the directory only after this function succeeds, which
    prevents readers from observing a mixture of old and partially written
    artifacts.
    """

    _write_parquet(output_path / "documents.parquet", batch.documents, columns=_DOCUMENT_COLUMNS)
    _write_parquet(output_path / "pages.parquet", batch.pages, columns=_PAGE_COLUMNS)
    _write_parquet(output_path / "blocks.parquet", batch.blocks, columns=_BLOCK_COLUMNS)
    _write_parquet(output_path / "assets.parquet", batch.assets, columns=_ASSET_COLUMNS)
    # Always pass the domain columns. Pandas cannot infer a schema from an empty
    # row list, but an edge-free graph is still a valid artifact set that must
    # satisfy the same local/Snowflake contract as a populated graph.
    _write_parquet(
        output_path / "chunks.parquet",
        _dump_models(batch.chunks),
        columns=_CHUNK_COLUMNS,
    )
    _write_parquet(
        output_path / "nodes.parquet",
        _dump_models(batch.nodes),
        columns=_NODE_COLUMNS,
    )
    _write_parquet(
        output_path / "edges.parquet",
        _dump_models(batch.edges),
        columns=_EDGE_COLUMNS,
    )
    _write_parquet(
        output_path / "edge_observations.parquet",
        _dump_models(batch.edge_observations),
        columns=_EDGE_OBSERVATION_COLUMNS,
    )
    _write_parquet(
        output_path / "evidence.parquet",
        _dump_models(batch.evidence),
        columns=_EVIDENCE_COLUMNS,
    )
    _write_parquet(
        output_path / "entity_sources.parquet",
        _dump_models(batch.entity_sources),
        columns=_ENTITY_SOURCE_COLUMNS,
    )
    _write_parquet(
        output_path / "communities.parquet",
        _dump_models(batch.communities),
        columns=_COMMUNITY_COLUMNS,
    )
    _write_parquet(
        output_path / "community_findings.parquet",
        _dump_models(batch.community_findings),
        columns=_COMMUNITY_FINDING_COLUMNS,
    )
    # Run reports and traces are review artifacts. Redacting here protects both
    # normal pipeline output and any tests or custom callers that construct a
    # GraphWriteBatch directly.
    (output_path / "run_report.json").write_text(
        json.dumps(redact_sensitive_data(batch.run_report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "graph_metrics.json").write_text(
        json.dumps(redact_sensitive_data(batch.graph_metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_path / "extraction_trace.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in redact_sensitive_data(batch.extraction_trace)
        ),
        encoding="utf-8",
    )
    (output_path / _ARTIFACT_MANIFEST).write_text(
        json.dumps({"format": _ARTIFACT_FORMAT}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _assert_replaceable_output(output_path: Path) -> None:
    """Refuse replacement unless an existing destination is empty or app-owned."""

    if not output_path.exists():
        return
    if not output_path.is_dir():
        raise ValueError(f"Refusing to replace non-directory output path: {output_path}")
    children = {child.name for child in output_path.iterdir()}
    if not children:
        return
    manifest_path = output_path / _ARTIFACT_MANIFEST
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            manifest = None
        if isinstance(manifest, dict) and manifest.get("format") == _ARTIFACT_FORMAT:
            return
    if _SNAPSHOT_FILES.issubset(children):
        return
    raise ValueError(f"Refusing to replace unrelated non-empty output directory: {output_path}")


def _replace_snapshot(temp_path: Path, output_path: Path) -> None:
    # The backup path is only kept during the rename window. If replacement
    # fails, the previous complete snapshot is restored before the exception
    # escapes.
    backup_path = output_path.parent / f".{output_path.name}.previous-{uuid.uuid4().hex}"
    backup_created = False
    if output_path.exists():
        output_path.replace(backup_path)
        backup_created = True
    try:
        temp_path.replace(output_path)
    except Exception:
        if backup_created and backup_path.exists() and not output_path.exists():
            backup_path.replace(output_path)
        raise
    finally:
        if backup_path.exists():
            _remove_path(backup_path)


def _dump_models(models: list[Any]) -> list[dict[str, Any]]:
    return [model.model_dump(mode="json") for model in models]


def _write_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> None:
    normalized_rows = [
        {key: _normalize_parquet_value(value) for key, value in row.items()} for row in rows
    ]
    frame = pd.DataFrame(normalized_rows, columns=columns)
    frame.to_parquet(path, index=False)


def _normalize_parquet_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
