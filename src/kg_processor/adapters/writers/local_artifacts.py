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
from kg_processor.domain.graph import GraphWriteBatch

_BLOCK_COLUMNS = ["id", "graph_id", "file_id", "page_number", "kind", "text", "bbox", "metadata"]
_ASSET_COLUMNS = ["id", "graph_id", "file_id", "kind", "page_number", "uri", "metadata"]


class LocalArtifactsWriter:
    """Writes Snowflake-shaped graph artifacts to a local directory atomically."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

    def write(self, batch: GraphWriteBatch) -> None:
        """Write a complete snapshot and replace the previous directory on success."""

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_path.parent / (f".{self.output_path.name}.tmp-{uuid.uuid4().hex}")
        try:
            temp_path.mkdir(parents=False, exist_ok=False)
            _write_snapshot(temp_path, batch)
            _replace_snapshot(temp_path, self.output_path)
        finally:
            if temp_path.exists():
                _remove_path(temp_path)


def _write_snapshot(output_path: Path, batch: GraphWriteBatch) -> None:
    _write_parquet(output_path / "documents.parquet", batch.documents)
    _write_parquet(output_path / "pages.parquet", batch.pages)
    _write_parquet(output_path / "blocks.parquet", batch.blocks, columns=_BLOCK_COLUMNS)
    _write_parquet(output_path / "assets.parquet", batch.assets, columns=_ASSET_COLUMNS)
    _write_parquet(output_path / "chunks.parquet", _dump_models(batch.chunks))
    _write_parquet(output_path / "nodes.parquet", _dump_models(batch.nodes))
    _write_parquet(output_path / "edges.parquet", _dump_models(batch.edges))
    _write_parquet(output_path / "evidence.parquet", _dump_models(batch.evidence))
    _write_parquet(
        output_path / "entity_sources.parquet",
        _dump_models(batch.entity_sources),
    )
    _write_parquet(output_path / "communities.parquet", _dump_models(batch.communities))
    _write_parquet(
        output_path / "community_findings.parquet",
        _dump_models(batch.community_findings),
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
