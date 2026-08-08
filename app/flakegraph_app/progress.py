"""Progress parsing and aggregation shared by all application views."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flakegraph_app.models import PIPELINE_STAGE_ORDER, ProgressEvent, StageProgress

_TAIL_BLOCK_BYTES = 64 * 1024
# Checkpoints are rebuilt from the event log whenever this changes, so the shape
# of the stored aggregates can be corrected without migrating existing files.
_SUMMARY_VERSION = 2
_SUMMARY_FILE = "progress-summary.json"
_SUMMARY_LOCK = threading.Lock()
_TERMINAL_SUCCESS = {"completed", "succeeded", "done"}


@dataclass(frozen=True)
class LocalProgress:
    """Bounded recent events plus a durable cumulative local-run summary."""

    events: list[ProgressEvent]
    stages: list[StageProgress]
    documents_total: int | None
    documents_completed: int
    documents_failed: int
    updated_at: str | None




def read_jsonl_events(path: Path, limit: int = 2_000) -> list[ProgressEvent]:
    """Read recent valid progress records without rereading an ever-growing log.

    Local workers append progress to JSONL files that can become very large for a
    long corpus run. Status fragments poll this function every few seconds, so the
    reader seeks backward in fixed-size blocks until it has enough recent lines.
    Runtime and memory therefore stay bounded by the requested review window rather
    than by the total run history.
    """

    if not path.exists() or limit <= 0:
        return []
    records: list[ProgressEvent] = []
    for line in _tail_lines(path, limit):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict) or raw.get("event") != "kg_processor.progress":
            continue
        records.append(progress_event(raw))
    return records


def read_local_progress(path: Path, limit: int = 2_000) -> LocalProgress:
    """Increment a durable summary and return it with a bounded event tail.

    The first read processes the complete log once. Later Streamlit reruns resume
    from the stored byte offset, so cumulative file and elapsed-time counters do
    not disappear when their events fall outside the recent review tail.
    """

    if not path.is_file():
        return LocalProgress([], [], None, 0, 0, None)
    with _SUMMARY_LOCK:
        summary_path = path.parent / _SUMMARY_FILE
        summary = _load_summary(summary_path)
        stat = path.stat()
        source_identity = f"{stat.st_dev}:{stat.st_ino}"
        offset = _summary_int(summary.get("offset"))
        rebuilt = (
            summary.get("version") != _SUMMARY_VERSION
            or summary.get("source_identity") != source_identity
            or offset > stat.st_size
        )
        if rebuilt:
            summary = _empty_summary(source_identity)
            offset = 0
        summary["offset"] = _accumulate_new_records(path, offset, summary)
        # A run page polls every few seconds, and the checkpoint holds one entry
        # per file per stage so it can count a retried file once. Rewriting it
        # when the log has not advanced would spend that whole cost on nothing.
        if rebuilt or summary["offset"] != offset:
            _write_summary(summary_path, summary)
    events = read_jsonl_events(path, limit)
    return LocalProgress(
        events=events,
        stages=_summary_stages(summary),
        documents_total=_summary_optional_int(summary.get("documents_total")),
        documents_completed=_stage_file_count(summary, "completed_files", "ocr"),
        documents_failed=_stage_file_count(summary, "failed_files", "ocr"),
        updated_at=str(summary.get("updated_at") or "") or None,
    )


def _empty_summary(source_identity: str) -> dict[str, Any]:
    """Return a fresh cumulative state for one physical event-log file."""

    return {
        "version": _SUMMARY_VERSION,
        "source_identity": source_identity,
        "offset": 0,
        "latest": {},
        "elapsed": {},
        # Which files a stage finished, not how many events said so. A file that
        # is retried or resumed reports its terminal event again, and counting
        # each one puts a stage past its own total.
        "completed_files": {},
        "failed_files": {},
        "completed_counts": {},
        "totals": {},
    }


def _load_summary(path: Path) -> dict[str, Any]:
    """Load a prior checkpoint, resetting safely after malformed state."""

    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _accumulate_new_records(path: Path, offset: int, summary: dict[str, Any]) -> int:
    """Stream complete appended lines into ``summary`` and return the next offset."""

    next_offset = offset
    with path.open("rb") as stream:
        stream.seek(offset)
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break
            next_offset = stream.tell()
            try:
                raw = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(raw, dict) and raw.get("event") == "kg_processor.progress":
                _accumulate_summary(summary, raw)
    return next_offset


def _accumulate_summary(  # noqa: PLR0912 - each event counter has an independent contract.
    summary: dict[str, Any],
    raw: Mapping[str, Any],
) -> None:
    """Fold one never-before-seen progress record into compact cumulative state."""

    event = progress_event(raw)
    latest = _summary_mapping(summary, "latest")
    elapsed = _summary_mapping(summary, "elapsed")
    completed_files = _summary_mapping(summary, "completed_files")
    failed_files = _summary_mapping(summary, "failed_files")
    completed_counts = _summary_mapping(summary, "completed_counts")
    totals = _summary_mapping(summary, "totals")
    latest[event.stage] = _event_record(event)
    summary["updated_at"] = event.timestamp
    if event.elapsed_ms is not None and event.status in _TERMINAL_SUCCESS:
        elapsed[event.stage] = _summary_int(elapsed.get(event.stage)) + event.elapsed_ms
    if event.file_id and event.status in _TERMINAL_SUCCESS:
        _record_file(completed_files, event.stage, event.file_id)
        # A file that later succeeds is no longer a failure of that stage.
        _forget_file(failed_files, event.stage, event.file_id)
    if event.file_id and event.status == "failed":
        _record_file(failed_files, event.stage, event.file_id)
    for completed_key, total_key in (
        ("batches_completed", "batches_total"),
        ("reports_completed", "reports_total"),
    ):
        completed_value = event.counts.get(completed_key)
        total_value = event.counts.get(total_key)
        if isinstance(completed_value, int):
            completed_counts[event.stage] = completed_value
        if isinstance(total_value, int):
            totals[event.stage] = total_value
    files_seen = event.counts.get("files_seen")
    if event.stage == "file_source" and isinstance(files_seen, int):
        totals[event.stage] = files_seen
        summary["documents_total"] = files_seen
        if event.status in _TERMINAL_SUCCESS:
            completed_counts[event.stage] = files_seen
    for key in ("files_total", "documents_total", "total"):
        value = event.counts.get(key)
        if isinstance(value, int):
            totals[event.stage] = value
            if event.stage in {"file_source", "ocr"}:
                summary["documents_total"] = value
            break


def _summary_stages(summary: Mapping[str, Any]) -> list[StageProgress]:
    """Materialize stage rows from cumulative checkpoint dictionaries."""

    latest = summary.get("latest")
    if not isinstance(latest, Mapping):
        return []
    elapsed = _readonly_summary_mapping(summary, "elapsed")
    completed_files = _readonly_summary_mapping(summary, "completed_files")
    completed_counts = _readonly_summary_mapping(summary, "completed_counts")
    totals = _readonly_summary_mapping(summary, "totals")
    order = {stage: index for index, stage in enumerate(PIPELINE_STAGE_ORDER)}
    rows: list[StageProgress] = []
    for stage, raw in sorted(
        latest.items(),
        key=lambda item: (order.get(str(item[0]), len(order)), str(item[0])),
    ):
        if not isinstance(raw, Mapping):
            continue
        event = progress_event(raw)
        total = _summary_optional_int(totals.get(stage))
        completed = max(
            _distinct_files(completed_files.get(stage)),
            _summary_int(completed_counts.get(stage)),
        )
        if event.status in _TERMINAL_SUCCESS and not event.file_id and total is not None:
            completed = total
        rows.append(
            StageProgress(
                stage=str(stage),
                status=event.status,
                completed=completed,
                total=total,
                elapsed_ms=_summary_int(elapsed.get(stage)),
                message=event.message,
            )
        )
    return rows


def _record_file(files: dict[str, Any], stage: str, file_id: str) -> None:
    """Record that one file reached a state in a stage, however often it says so."""

    seen = files.get(stage)
    if not isinstance(seen, dict):
        seen = {}
        files[stage] = seen
    seen[file_id] = True


def _forget_file(files: dict[str, Any], stage: str, file_id: str) -> None:
    """Withdraw one file from a stage's set after a later event supersedes it."""

    seen = files.get(stage)
    if isinstance(seen, dict):
        seen.pop(file_id, None)


def _distinct_files(value: object) -> int:
    """Return how many distinct files one stage set holds."""

    return len(value) if isinstance(value, Mapping) else 0


def _stage_file_count(summary: Mapping[str, Any], aggregate: str, stage: str) -> int:
    """Return one stage's distinct file count from a named checkpoint aggregate."""

    return _distinct_files(_readonly_summary_mapping(summary, aggregate).get(stage))


def _summary_mapping(summary: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a mutable string-keyed checkpoint mapping for one aggregate."""

    value = summary.get(key)
    if not isinstance(value, dict):
        value = {}
        summary[key] = value
    return value


def _readonly_summary_mapping(summary: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return one checkpoint mapping with a stable typed empty fallback."""

    value = summary.get(key)
    return value if isinstance(value, Mapping) else {}


def _event_record(event: ProgressEvent) -> dict[str, Any]:
    """Serialize the validated event subset used to reconstruct stage state."""

    return {
        "timestamp": event.timestamp,
        "stage": event.stage,
        "status": event.status,
        "file_id": event.file_id,
        "message": event.message,
        "elapsed_ms": event.elapsed_ms,
        "counts": dict(event.counts),
        "metadata": dict(event.metadata),
    }


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Atomically persist one checkpoint without sharing temporary filenames."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary_int(value: object) -> int:
    """Return a non-negative checkpoint integer, treating malformed values as zero."""

    return value if isinstance(value, int) and value >= 0 else 0


def _summary_optional_int(value: object) -> int | None:
    """Return a non-negative checkpoint integer while preserving missing totals."""

    return value if isinstance(value, int) and value >= 0 else None


def _tail_lines(path: Path, limit: int) -> list[str]:
    """Decode at most the final ``limit`` lines using bounded reverse reads."""

    chunks: list[bytes] = []
    newline_count = 0
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position = stream.tell()
        while position > 0 and newline_count <= limit:
            block_size = min(_TAIL_BLOCK_BYTES, position)
            position -= block_size
            stream.seek(position)
            chunk = stream.read(block_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    payload = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
    return payload.splitlines()[-limit:]


def progress_event(raw: Mapping[str, Any]) -> ProgressEvent:
    """Validate the stable subset of one core progress record."""

    return ProgressEvent(
        timestamp=str(raw.get("timestamp", "")),
        stage=str(raw.get("stage", "unknown")),
        status=str(raw.get("status", "unknown")),
        file_id=str(raw["file_id"]) if raw.get("file_id") else None,
        message=str(raw["message"]) if raw.get("message") else None,
        elapsed_ms=int(raw["elapsed_ms"]) if raw.get("elapsed_ms") is not None else None,
        counts=dict(raw.get("counts") or {}),
        metadata=dict(raw.get("metadata") or {}),
    )


def aggregate_stages(events: Iterable[ProgressEvent]) -> list[StageProgress]:
    """Collapse event history into one deterministic row per observed stage."""

    latest: dict[str, ProgressEvent] = {}
    elapsed: dict[str, int] = defaultdict(int)
    completed_files: dict[str, set[str]] = defaultdict(set)
    completed_counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for event in events:
        latest[event.stage] = event
        if event.elapsed_ms is not None and event.status in {"completed", "succeeded", "done"}:
            elapsed[event.stage] += event.elapsed_ms
        if event.file_id and event.status in {"completed", "succeeded", "done"}:
            completed_files[event.stage].add(event.file_id)
        for completed_key, total_key in (
            ("batches_completed", "batches_total"),
            ("reports_completed", "reports_total"),
        ):
            completed_value = event.counts.get(completed_key)
            total_value = event.counts.get(total_key)
            if isinstance(completed_value, int):
                completed_counts[event.stage] = completed_value
            if isinstance(total_value, int):
                totals[event.stage] = total_value
        files_seen = event.counts.get("files_seen")
        if event.stage == "file_source" and isinstance(files_seen, int):
            totals[event.stage] = files_seen
            if event.status in {"completed", "succeeded", "done"}:
                completed_counts[event.stage] = files_seen
        for key in ("files_total", "documents_total", "total"):
            value = event.counts.get(key)
            if isinstance(value, int):
                totals[event.stage] = value
                break

    order = {stage: index for index, stage in enumerate(PIPELINE_STAGE_ORDER)}
    return [
        StageProgress(
            stage=stage,
            status=event.status,
            completed=(
                totals[stage]
                if (
                    event.status in {"completed", "succeeded", "done"}
                    and not event.file_id
                    and stage in totals
                )
                else max(len(completed_files[stage]), completed_counts.get(stage, 0))
            ),
            total=totals.get(stage),
            elapsed_ms=elapsed[stage],
            message=event.message,
        )
        for stage, event in sorted(
            latest.items(),
            key=lambda item: (order.get(item[0], len(order)), item[0]),
        )
    ]
