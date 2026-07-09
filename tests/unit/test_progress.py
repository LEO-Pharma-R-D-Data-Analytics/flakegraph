from __future__ import annotations

import io
import json

from kg_processor.application.progress import (
    CompositeProgressSink,
    JsonLineProgressSink,
    ProgressEvent,
)


def test_json_line_progress_sink_writes_parseable_redirection_safe_records() -> None:
    stream = io.StringIO()
    sink = JsonLineProgressSink(stream)

    sink.emit(
        ProgressEvent(
            job_id="job",
            graph_id="graph",
            stage="ocr",
            status="completed",
            file_id="file_1",
            counts={"pages": 2},
            metadata={"provider": "mineru_internal"},
            elapsed_ms=123,
            timestamp="2026-07-03T00:00:00+00:00",
        )
    )

    payload = json.loads(stream.getvalue())

    assert payload == {
        "counts": {"pages": 2},
        "elapsed_ms": 123,
        "event": "kg_processor.progress",
        "file_id": "file_1",
        "graph_id": "graph",
        "job_id": "job",
        "metadata": {"provider": "mineru_internal"},
        "stage": "ocr",
        "status": "completed",
        "timestamp": "2026-07-03T00:00:00+00:00",
    }


def test_composite_progress_sink_fans_out_events_in_order() -> None:
    first = RecordingSink()
    second = RecordingSink()
    event = ProgressEvent(job_id="job", graph_id="graph", stage="merge", status="started")

    CompositeProgressSink([first, second]).emit(event)

    assert first.events == [event]
    assert second.events == [event]


def test_progress_event_redacts_sensitive_metadata_before_json_serialization() -> None:
    event = ProgressEvent(
        job_id="job",
        graph_id="graph",
        stage="ocr",
        status="completed",
        metadata={
            "provider": "mineru_api",
            "api_key": "ocr-secret",
            "nested": {"authorization": "Bearer token"},
        },
    )

    payload = event.as_log_record()

    assert payload["metadata"] == {
        "provider": "mineru_api",
        "api_key": "***",
        "nested": {"authorization": "***"},
    }


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)
