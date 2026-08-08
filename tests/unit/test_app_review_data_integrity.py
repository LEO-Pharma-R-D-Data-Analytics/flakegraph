"""Focused regression tests for local review-state and artifact integrity."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest
from flakegraph_app.backends.local import LocalBackend
from flakegraph_app.graph_store import (
    GRAPH_REVIEW_ROW_LIMITS,
    connector_edge_projection_query,
    load_local_graph,
)
from flakegraph_app.progress import read_local_progress
from flakegraph_app.run_catalog import read_run_record, write_run_record
from flakegraph_app.ui.run_workspace import _cache_graph_dataset

from kg_processor.adapters.files.azure_blob import (
    AzureBlobFileSourceConfig,
)
from kg_processor.adapters.files.azure_blob import (
    _download_root as azure_download_root,
)
from kg_processor.adapters.files.common import object_download_path
from kg_processor.adapters.files.s3 import (
    S3FileSourceConfig,
)
from kg_processor.adapters.files.s3 import (
    _download_root as s3_download_root,
)
from kg_processor.adapters.writers.local_artifacts import _assert_replaceable_output
from kg_processor.domain.ids import stable_id


def test_artifact_writer_refuses_unrelated_non_empty_directory(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "notes.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(ValueError, match="unrelated non-empty"):
        _assert_replaceable_output(output)

    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_remote_downloads_are_namespaced_so_no_two_objects_share_a_file(
    tmp_path: Path,
) -> None:
    """Keep two objects off one another's staged file, whatever their keys say.

    Object keys are opaque strings, so a store accepts an empty segment, a
    backslash, a leading slash and ``..`` alike, and two accounts can hold the
    same key. Staging by the caller's identity for the whole key, under a root
    that carries the source, is what keeps one object's bytes from being read as
    another's.
    """

    azure_a = AzureBlobFileSourceConfig(
        account_url="https://one.example",
        connection_string=None,
        container="docs",
        prefix="incoming",
        sas_token=None,
        download_path=tmp_path,
    )
    azure_b = AzureBlobFileSourceConfig(
        account_url="https://two.example",
        connection_string=None,
        container="docs",
        prefix="incoming",
        sas_token=None,
        download_path=tmp_path,
    )
    s3_a = S3FileSourceConfig("docs", "incoming", "https://one.example", None, tmp_path)
    s3_b = S3FileSourceConfig("docs", "incoming", "https://two.example", None, tmp_path)

    assert azure_download_root(azure_a) != azure_download_root(azure_b)
    assert s3_download_root(s3_a) != s3_download_root(s3_b)
    # Keys a store accepts but a path parser would reject, plus one that would
    # traverse if the key were treated as a path.
    keys = (
        "folder/report.pdf",
        "folder//report.pdf",
        "folder/./report.pdf",
        "folder\\report.pdf",
        "../../etc/passwd",
        "/etc/passwd",
    )
    root = s3_download_root(s3_a)
    staged = {
        key: object_download_path(root, key, stable_id("s3_download_path", "docs", key))
        for key in keys
    }
    assert len(set(staged.values())) == len(keys)
    for key, path in staged.items():
        assert root.resolve() in path.resolve().parents, key


def test_local_progress_checkpoint_keeps_counts_beyond_recent_tail(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    records = [_progress_record("file_source", "completed", counts={"files_seen": 6})]
    records.extend(
        _progress_record("ocr", "completed", file_id=f"file-{index}") for index in range(5)
    )
    events_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    first = read_local_progress(events_path, limit=2)
    checkpoint = json.loads((tmp_path / "progress-summary.json").read_text(encoding="utf-8"))
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_progress_record("ocr", "completed", file_id="file-5")) + "\n")
    second = read_local_progress(events_path, limit=2)

    assert len(first.events) == 2
    assert first.documents_completed == 5
    assert second.documents_total == 6
    assert second.documents_completed == 6
    assert checkpoint["offset"] < events_path.stat().st_size


def test_cancellation_remains_terminal_after_restart_and_late_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    run_directory = state_root / "runs" / "run-1"
    output = tmp_path / "output"
    output.mkdir()
    pd.DataFrame([{"id": "n1", "graph_id": "g1"}]).to_parquet(output / "nodes.parquet")
    pd.DataFrame(columns=["id", "source_node_id", "target_node_id"]).to_parquet(
        output / "edges.parquet"
    )
    write_run_record(
        run_directory,
        {
            "run_id": "run-1",
            "graph_id": "g1",
            "runtime": "local",
            "status": "running",
            "pid": 999_999,
            "output_path": str(output),
        },
    )
    attempted: list[dict[str, object]] = []

    def record_cancel(record: dict[str, object]) -> bool:
        attempted.append(dict(record))
        return False

    monkeypatch.setattr(
        "flakegraph_app.backends.local.cancel_recorded_process",
        record_cancel,
    )

    snapshot = LocalBackend(tmp_path, state_root).cancel("run-1")
    write_run_record(run_directory, {"status": "succeeded"})

    assert attempted and snapshot.status == "cancelled"
    assert read_run_record(run_directory)["status"] == "cancelled"


def test_local_parquet_review_is_bounded_with_full_counts_and_coherent_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(GRAPH_REVIEW_ROW_LIMITS, "KG_NODE", 2)
    monkeypatch.setitem(GRAPH_REVIEW_ROW_LIMITS, "KG_EDGE", 2)
    pd.DataFrame(
        [{"id": f"n{index}", "graph_id": "g1", "embedding": [float(index)]} for index in range(3)]
    ).to_parquet(tmp_path / "nodes.parquet")
    pd.DataFrame(
        [
            {"id": "outside", "graph_id": "g1", "source_node_id": "n1", "target_node_id": "n2"},
            {"id": "inside", "graph_id": "g1", "source_node_id": "n0", "target_node_id": "n1"},
        ]
    ).to_parquet(tmp_path / "edges.parquet")

    dataset = load_local_graph(tmp_path)

    assert len(dataset.nodes) == 2
    assert [edge["id"] for edge in dataset.edges] == ["inside"]
    assert dataset.counts["nodes"] == 3
    assert dataset.counts["edges"] == 2
    assert "embedding" not in dataset.nodes[0]


def test_graph_store_has_no_processing_package_import_at_module_scope() -> None:
    source_path = Path(__file__).parents[2] / "app" / "flakegraph_app" / "graph_store.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = [node.module or "" for node in module.body if isinstance(node, ast.ImportFrom)]

    assert not any(name.startswith("kg_processor") for name in imported)


def test_connector_edges_are_projected_from_the_selected_node_sample() -> None:
    query = connector_edge_projection_query(25, 50)

    assert "WITH SELECTED_NODES" in query
    assert "SOURCE_NODE_ID IN (SELECT ID FROM SELECTED_NODES)" in query
    assert "TARGET_NODE_ID IN (SELECT ID FROM SELECTED_NODES)" in query
    assert query.count("?") == 2


def test_run_graph_session_cache_evicts_oldest_entries() -> None:
    cache: dict[str, object] = {}

    for index in range(6):
        _cache_graph_dataset(cache, f"run-{index}", object())

    assert list(cache) == ["run-2", "run-3", "run-4", "run-5"]


def _progress_record(
    stage: str,
    status: str,
    *,
    file_id: str | None = None,
    counts: dict[str, int] | None = None,
) -> dict[str, object]:
    """Build one core-compatible progress JSON record."""

    return {
        "event": "kg_processor.progress",
        "timestamp": f"2026-01-01T00:00:{file_id or stage}Z",
        "stage": stage,
        "status": status,
        "file_id": file_id,
        "counts": counts or {},
    }
