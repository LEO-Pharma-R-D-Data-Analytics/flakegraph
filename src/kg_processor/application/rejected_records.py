# SPDX-License-Identifier: Apache-2.0
"""List every entity and relation validation turned away, and why.

Both engines derive these rows from the extraction trace, as they do the
discarded windows: the local pipeline from its in-process trace, the Spark
finalizer from its trace column. The summary names the relation type rules the
model broke most, so an ontology's author can see which rules cost true facts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from kg_processor.domain.graph import Chunk, RejectedRecord
from kg_processor.domain.ids import stable_id

# The passes whose rejected records are listed.
REJECTING_STAGES = frozenset(
    {"entity_extraction", "document_context_extraction", "relation_extraction"}
)
REJECTED_RECORD_FIELDS = (
    "name",
    "type",
    "source",
    "source_type",
    "target",
    "target_type",
    "quote",
    "chunk_id",
)
# How many broken type rules a run report names.
MAX_REPORTED_TYPE_RULES = 25


def rejected_records_from_trace(
    trace: Iterable[Mapping[str, Any]],
    chunks: Iterable[Chunk],
    graph_id: str,
) -> list[RejectedRecord]:
    """Turn the trace's rejected records into rows, one per distinct record.

    A record rejected again by a later pass over the same text is one row.
    """

    by_id = {chunk.id: chunk for chunk in chunks}
    rows: dict[str, RejectedRecord] = {}
    for event in trace:
        if str(event.get("stage") or "") not in REJECTING_STAGES:
            continue
        window_id = str(event.get("window_id") or "")
        for record in event.get("rejected_records") or []:
            if not isinstance(record, Mapping):
                continue
            values = {field: str(record.get(field) or "") for field in REJECTED_RECORD_FIELDS}
            kind = "relation" if record.get("kind") == "relation" else "entity"
            reason = str(record.get("reason") or "rejected")
            chunk = by_id.get(values["chunk_id"])
            document_id = str(event.get("document_id") or (chunk.document_id if chunk else ""))
            row_id = stable_id(
                "rejected_record",
                graph_id,
                window_id,
                kind,
                reason,
                *values.values(),
            )
            rows.setdefault(
                row_id,
                RejectedRecord(
                    id=row_id,
                    graph_id=graph_id,
                    document_id=document_id,
                    file_id=chunk.file_id if chunk else document_id,
                    window_id=window_id,
                    page_number=chunk.page_number if chunk else None,
                    kind=kind,
                    reason=reason,
                    **values,
                ),
            )
    return sorted(
        rows.values(),
        key=lambda row: (row.document_id, row.page_number or 0, row.kind, row.reason, row.id),
    )


def rejected_record_metrics(rows: Iterable[RejectedRecord]) -> dict[str, Any]:
    """The summary a run report carries: what was rejected, why, and which rules."""

    by_kind: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    rules: Counter[tuple[str, str, str]] = Counter()
    documents: set[str] = set()
    for row in rows:
        by_kind[row.kind] += 1
        by_reason[f"{row.kind}:{row.reason}"] += 1
        documents.add(row.document_id)
        if row.kind == "relation" and row.reason == "domain_or_range_violation":
            rules[(row.source_type, row.name, row.target_type)] += 1
    return {
        "records": sum(by_kind.values()),
        "documents": len(documents),
        "by_kind": dict(sorted(by_kind.items())),
        "by_reason": dict(sorted(by_reason.items(), key=lambda item: (-item[1], item[0]))),
        "broken_type_rules": [
            {"source_type": source, "relation_type": relation, "target_type": target, "count": n}
            for (source, relation, target), n in sorted(
                rules.items(), key=lambda item: (-item[1], *item[0])
            )[:MAX_REPORTED_TYPE_RULES]
        ],
    }
