"""Live stage progress for Snowflake runs.

File rows only move between queued, claimed, and done. A stage that runs for many
minutes over one claimed batch therefore reports identical zero progress
throughout, which is indistinguishable from a stalled worker — the app showed
"0/10" while extraction was at batch 20 of 21.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))

from flakegraph_app.backends.snowflake import (  # noqa: E402
    _progress_counts,
    _variant_mapping,
    _with_live_stage_progress,
)
from flakegraph_app.models import StageProgress  # noqa: E402


def test_worker_counts_replace_file_derived_zero() -> None:
    stages = [StageProgress("graph_extraction", "running", 0, 10)]

    updated = _with_live_stage_progress(
        stages,
        {"stage": "graph_extraction", "status": "progress",
         "counts": {"batches_completed": 20, "batches_total": 21}},
    )

    assert (updated[0].completed, updated[0].total) == (20, 21)


def test_file_counts_survive_when_the_worker_reports_nothing() -> None:
    """An older worker writes no progress; the previous behaviour must remain."""

    stages = [StageProgress("graph_extraction", "running", 3, 10)]

    assert _with_live_stage_progress(stages, None) == stages


def test_a_stage_the_file_rows_have_not_seen_is_still_shown() -> None:
    updated = _with_live_stage_progress(
        [StageProgress("ocr", "completed", 10, 10)],
        {"stage": "community_detection", "status": "progress",
         "counts": {"reports_completed": 4, "reports_total": 12}},
    )

    assert [s.stage for s in updated] == ["ocr", "community_detection"]
    assert (updated[1].completed, updated[1].total) == (4, 12)


def test_counts_are_discovered_not_assumed() -> None:
    """Stages count in their own units, so no single key pair can be hardcoded."""

    assert _progress_counts({"batches_completed": 2, "batches_total": 5}) == (2, 5)
    assert _progress_counts({"reports_completed": 1, "reports_total": 3}) == (1, 3)
    assert _progress_counts({"entities": 40}) == (0, None)
    assert _progress_counts({"batches_completed": "x", "batches_total": 5}) == (0, None)


def test_variant_arrives_as_mapping_or_json_text() -> None:
    """Snowpark returns VARIANT as a mapping or as JSON text depending on path."""

    assert _variant_mapping({"stage": "ocr"}) == {"stage": "ocr"}
    assert _variant_mapping('{"stage": "ocr"}') == {"stage": "ocr"}
    assert _variant_mapping("not json") == {}
    assert _variant_mapping(None) == {}
