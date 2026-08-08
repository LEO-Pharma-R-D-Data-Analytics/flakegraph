"""Small durable catalog for runs submitted through a local app server."""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_HIDDEN_RUNS_FILE = "hidden-runs.json"
_GRAPH_NAMES_FILE = "graph-names.json"
_MAX_GRAPH_NAME_LENGTH = 120
_ASCII_CONTROL_CHARACTER_LIMIT = 32
_CATALOG_LOCK = threading.RLock()


def write_run_record(run_directory: Path, values: Mapping[str, Any]) -> Path:
    """Merge metadata atomically while preserving a requested cancellation."""

    with _CATALOG_LOCK:
        run_directory.mkdir(parents=True, exist_ok=True)
        destination = run_directory / "run.json"
        current = read_run_record(run_directory)
        payload = {
            **current,
            **{str(key): value for key, value in values.items() if value is not None},
        }
        if current.get("cancellation_requested_at") or payload.get("cancellation_requested_at"):
            payload["status"] = "cancelled"
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def read_run_record(run_directory: Path) -> dict[str, Any]:
    """Read one catalog record, returning an empty mapping for malformed history."""

    path = run_directory / "run.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(key): item for key, item in value.items()} if isinstance(value, dict) else {}


def list_run_records(state_root: Path, limit: int = 100) -> Sequence[tuple[Path, dict[str, Any]]]:
    """Return newest app-owned run records without scanning output artifacts."""

    runs_root = state_root / "runs"
    if not runs_root.exists():
        return []
    directories = sorted(
        (path for path in runs_root.iterdir() if path.is_dir() and (path / "run.json").exists()),
        key=lambda path: (path / "run.json").stat().st_mtime,
        reverse=True,
    )
    return [(path, read_run_record(path)) for path in directories[:limit]]


def hidden_run_ids(state_root: Path) -> frozenset[str]:
    """Return run IDs deliberately removed from app history.

    Distributed coordination and graph storage remain authoritative and are not
    deleted by a navigation action. A small app-owned tombstone file therefore
    keeps removed entries out of future durable-run listings without mutating the
    processing ledger.
    """

    path = state_root / _HIDDEN_RUNS_FILE
    if not path.is_file():
        return frozenset()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    if not isinstance(value, list):
        return frozenset()
    return frozenset(str(item) for item in value if str(item).strip())


def hide_run(state_root: Path, run_id: str) -> None:
    """Atomically add one non-empty run identity to the app history tombstones."""

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run ID must not be empty")
    state_root.mkdir(parents=True, exist_ok=True)
    hidden = sorted({*hidden_run_ids(state_root), normalized})
    destination = state_root / _HIDDEN_RUNS_FILE
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(hidden, indent=2), encoding="utf-8")
    temporary.replace(destination)


def graph_name(state_root: Path, graph_id: str) -> str | None:
    """Return a graph's optional display name from the app-owned catalog."""

    return _read_graph_names(state_root).get(graph_id)


def rename_graph(state_root: Path, graph_id: str, display_name: str) -> str:
    """Atomically persist a validated display name without changing graph identity."""

    normalized_id = graph_id.strip()
    if not normalized_id:
        raise ValueError("graph ID must not be empty")
    normalized_name = validate_graph_name(display_name)
    state_root.mkdir(parents=True, exist_ok=True)
    names = _read_graph_names(state_root)
    names[normalized_id] = normalized_name
    destination = state_root / _GRAPH_NAMES_FILE
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(names, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return normalized_name


def validate_graph_name(value: str) -> str:
    """Normalize a concise single-line graph name suitable for every backend."""

    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("Graph name must not be empty")
    if len(normalized) > _MAX_GRAPH_NAME_LENGTH:
        raise ValueError(f"Graph name must be at most {_MAX_GRAPH_NAME_LENGTH} characters")
    if any(ord(character) < _ASCII_CONTROL_CHARACTER_LIMIT for character in normalized):
        raise ValueError("Graph name must not contain control characters")
    return normalized


def _read_graph_names(state_root: Path) -> dict[str, str]:
    """Read the optional display-name registry and ignore malformed entries."""

    path = state_root / _GRAPH_NAMES_FILE
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(name)
        for key, name in value.items()
        if str(key).strip() and isinstance(name, str) and name.strip()
    }
