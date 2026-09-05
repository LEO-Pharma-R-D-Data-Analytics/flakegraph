"""Contracts the Streamlit control plane owes an operator.

These cover the places where the app speaks for a run: what it says a run was
configured to do, what it says a run has done, what it keeps out of a rendered
configuration, and what it refuses to read.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from flakegraph_app.backends.kubernetes import _semantic_mismatches
from flakegraph_app.backends.local import LocalBackend, _failure_summary
from flakegraph_app.backends.snowflake import SnowflakeBackend
from flakegraph_app.configuration import (
    _ontology_profile_path,
    build_run_config,
    redacted_config,
    resolve_profile_path,
)
from flakegraph_app.explorer import (
    build_graph_figure,
    filter_graph,
    layout_fallback_notice,
)
from flakegraph_app.models import (
    GraphDataset,
    IngestionRequest,
    OutputDestination,
    ProviderSelection,
    RuntimeMode,
    SourceKind,
    StorageKind,
)
from flakegraph_app.processes import ManagedProcess
from flakegraph_app.progress import read_local_progress
from flakegraph_app.providers import (
    CORTEX_EMBEDDING_MODELS_BY_WIDTH,
    DEFAULTS,
    EMBEDDING_MODEL_DIMENSIONS,
    cortex_embedding_model_for_width,
)
from flakegraph_app.ui.ingestion import (
    _adopt_stored_embedding_width,
    _source_controls,
    _with_fleet_ocr_routing,
)
from flakegraph_app.ui.navigation import _dismiss_bulk_forget, _dismiss_forget
from streamlit.testing.v1 import AppTest

from kg_processor.adapters.snowflake import stage_path

_ROOT = Path(__file__).resolve().parents[2]
# The tables the Snowflake control plane requires before a run may start.
_REQUIRED_SNOWFLAKE_APP_TABLES = (
    "KG_GRAPH",
    "KG_JOB",
    "KG_JOB_FILE",
    "KG_NODE",
    "KG_EDGE",
)
_APP_PACKAGE = _ROOT / "app" / "flakegraph_app"


# --------------------------------------------------------------------------- #
# Effective configuration an operator approves
# --------------------------------------------------------------------------- #


def test_snowflake_deployment_destination_declares_managed_oauth() -> None:
    """Say the authentication the deployed app actually uses.

    Inside Snowflake there is nothing to choose: the app and its SPCS worker
    share a Snowflake-managed OAuth token. The effective configuration is what an
    operator reads before approving a run and is what the approval digest covers,
    so it has to name that mechanism rather than one nothing in the deployment
    can perform.
    """

    app = AppTest.from_string(
        """
from pathlib import Path
import streamlit as st
from flakegraph_app.models import RuntimeMode
from flakegraph_app.ui.ingestion import _output_controls

class Backend:
    capabilities = frozenset()

    def current_context(self):
        return {"account": "ACME", "user": "SERVICE", "database": "DB", "schema": "GRAPH"}

    def list_databases(self):
        return ["DB"]

    def list_schemas(self, database=""):
        return ["GRAPH"]

    def list_warehouses(self):
        return ["WH"]

    def list_roles(self):
        return ["APP_ROLE"]

    def list_stages(self):
        return ["KG_LOAD_STAGE"]

output = _output_controls(
    RuntimeMode.SNOWFLAKE,
    Path("."),
    "graph-1",
    "job-1",
    Path("missing-profile.yaml"),
    Backend(),
)
st.session_state["destination"] = output
"""
    ).run()

    destination = app.session_state["destination"]
    assert destination is not None
    assert destination.snowflake.authenticator == "oauth"
    assert destination.snowflake.credential_environment_variable is None


def test_credentials_inside_url_values_are_redacted_from_configuration_previews() -> None:
    """Redact a credential the value carries, not only one its key names.

    A shared-access signature lives in ``account_url``, a database password in
    ``url``, an OCR key in ``endpoint``. None of those keys reads as sensitive,
    and the redacted configuration is rendered on the page, written to the run
    profile, and inlined into the SPCS specification uploaded to a stage.
    """

    redacted = redacted_config(
        {
            "azure_blob": {"account_url": "https://acct.blob.core.windows.net/?sv=x&sig=SIGNATURE"},
            "distributed": {"url": "postgresql://kg:HUNTER2@db.internal:5432/graph"},
            "generic_http_ocr": {"endpoint": "https://ocr.example/v1/parse?api_key=KEYVALUE"},
            "files": {"input_path": "/data/martial_arts"},
        }
    )

    rendered = yaml.safe_dump(redacted)
    assert "SIGNATURE" not in rendered
    assert "HUNTER2" not in rendered
    assert "KEYVALUE" not in rendered
    # Everything that is not a credential survives, or the preview stops being
    # worth reading.
    assert "acct.blob.core.windows.net" in rendered
    assert "db.internal" in rendered
    assert redacted["files"]["input_path"] == "/data/martial_arts"


def test_a_credential_bearing_source_url_is_rejected_before_a_run_is_composed(
    tmp_path: Path,
) -> None:
    """Refuse a literal credential wherever it is written, including inside a URL."""

    request = _request(
        tmp_path,
        SourceKind.AZURE_BLOB,
        {
            "kind": "azure_blob",
            "account_url": "https://acct.blob.core.windows.net/?sig=SIGNATURE",
            "container": "docs",
            "prefix": "",
        },
    )

    with pytest.raises(ValueError, match="carries a credential"):
        build_run_config(request)


def test_base_configuration_is_confined_to_the_applications_own_roots(
    tmp_path: Path,
) -> None:
    """Keep the profile field naming a reviewed profile, not any readable file.

    Its contents are parsed and rendered back into the browser, so an unbounded
    path turns one text box into a general file-disclosure control.
    """

    repository = tmp_path / "repository"
    (repository / "configs").mkdir(parents=True)
    profile = repository / "configs" / "app-defaults.yaml"
    profile.write_text("graph: {}\n", encoding="utf-8")
    private = tmp_path / "home" / ".docker" / "config.json"
    private.parent.mkdir(parents=True)
    private.write_text(json.dumps({"auths": {"registry": {"auth": "SECRET"}}}), encoding="utf-8")

    assert resolve_profile_path(str(profile), repository) == profile.resolve()
    with pytest.raises(ValueError, match="must be a file under"):
        resolve_profile_path(str(private), repository)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_profile_path(str(repository / "configs" / "absent.yaml"), repository)


def test_an_ontology_reference_cannot_traverse_out_of_the_repository(
    tmp_path: Path,
) -> None:
    """Resolve an ontology only where the repository actually keeps one.

    The reference is resolved by walking up from the profile, so a relative path
    that climbs out of the tree would otherwise find a file outside it and inline
    that file's contents into the configuration the run publishes.
    """

    outside = tmp_path / "private" / "ontology.yaml"
    outside.parent.mkdir(parents=True)
    outside.write_text("entities: []\n", encoding="utf-8")
    profile = tmp_path / "repository" / "configs" / "profile.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("ontology:\n  profile_path: ../private/ontology.yaml\n", encoding="utf-8")

    resolved = _ontology_profile_path(
        {"ontology": {"profile_path": "../private/ontology.yaml"}},
        profile,
    )

    assert resolved is None


# --------------------------------------------------------------------------- #
# What the app reports a run has done
# --------------------------------------------------------------------------- #


def test_a_cancelled_run_still_reports_the_documents_it_registered() -> None:
    """Count every registered file, whatever state it ended in.

    Cancellation rewrites queued and claimed rows to CANCELLED. The sidebar
    counts file rows outright, and the per-stage breakdown lists them, so a total
    assembled from only the states work passes through makes one page disagree
    with itself.
    """

    job = {
        "ID": "job-1",
        "GRAPH_ID": "graph-1",
        "GRAPH_NAME": "Martial arts",
        "STATUS": "CANCELLED",
        "ERROR": None,
        "CREATED_AT": "2026-07-16T10:00:00",
        "UPDATED_AT": "2026-07-16T10:05:00",
        "PROGRESS": None,
        "SECONDS_SINCE_UPDATE": 4,
    }
    session = _Session(
        [
            _Row({**job, "STAGE": "ocr", "FILE_STATUS": "DONE", "FILE_COUNT": 1}),
            _Row({**job, "STAGE": "ocr", "FILE_STATUS": "CANCELLED", "FILE_COUNT": 3}),
        ]
    )

    snapshot = SnowflakeBackend(session).status("job-1")

    assert snapshot.documents_total == 4
    assert snapshot.documents_completed == 1
    assert snapshot.documents_failed == 0
    assert sum(stage.total or 0 for stage in snapshot.stages) == snapshot.documents_total


def test_a_retried_file_is_counted_once_by_cumulative_progress(tmp_path: Path) -> None:
    """Count files, not the events that mention them.

    Cumulative checkpoints survive the bounded event tail, so a file whose stage
    is retried or resumed reports its terminal event more than once. Counting
    each occurrence walks a stage past its own total.
    """

    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "event": "kg_processor.progress",
                    "timestamp": "2026-07-16T10:00:00Z",
                    "stage": "file_source",
                    "status": "completed",
                    "counts": {"files_seen": 2},
                },
                _ocr_event("file-a", "completed", "2026-07-16T10:00:01Z"),
                _ocr_event("file-b", "failed", "2026-07-16T10:00:02Z"),
                _ocr_event("file-a", "completed", "2026-07-16T10:00:03Z"),
                _ocr_event("file-b", "completed", "2026-07-16T10:00:04Z"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    progress = read_local_progress(events)
    ocr = next(stage for stage in progress.stages if stage.stage == "ocr")

    assert progress.documents_total == 2
    assert progress.documents_completed == 2
    # The earlier failure was superseded by a success for the same file.
    assert progress.documents_failed == 0
    assert ocr.completed == 2


def test_a_resumed_read_keeps_counting_files_rather_than_events(tmp_path: Path) -> None:
    """Hold the de-duplication across the checkpoint reads that build it up."""

    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(_ocr_event("file-a", "completed", "2026-07-16T10:00:01Z")) + "\n",
        encoding="utf-8",
    )
    assert read_local_progress(events).documents_completed == 1

    with events.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_ocr_event("file-a", "completed", "2026-07-16T10:00:09Z")) + "\n")

    assert read_local_progress(events).documents_completed == 1


def test_a_failed_local_run_reports_a_cause_not_its_log(tmp_path: Path) -> None:
    """Summarize a stopped worker instead of quoting the stream it shares.

    Structured progress and the worker's own diagnostics go to one stream, so its
    tail interleaves JSON records with traceback frames and the useful line is
    not the last one.
    """

    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps(_ocr_event("file-a", "completed", "2026-07-16T10:00:01Z")),
                json.dumps(
                    {
                        "event": "kg_processor.progress",
                        "timestamp": "2026-07-16T10:00:02Z",
                        "stage": "graph_extraction",
                        "status": "failed",
                        "file_id": "file-b",
                        "message": "The LLM endpoint refused the request: 401 Unauthorized",
                    }
                ),
                "Traceback (most recent call last):",
                '  File "/app/kg_processor/cli.py", line 88, in worker',
                "    run(settings)",
                "RuntimeError: provider unavailable",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _failure_summary(events, [])

    assert summary == "RuntimeError: provider unavailable"
    assert "\n" not in summary


def test_a_failed_local_snapshot_carries_one_sentence(tmp_path: Path) -> None:
    """Keep the run page's error a diagnosis rather than a wall of log."""

    run_directory = tmp_path / "state" / "runs" / "job-1"
    run_directory.mkdir(parents=True)
    events = run_directory / "events.jsonl"
    events.write_text(
        "\n".join(
            [
                json.dumps(_ocr_event(f"file-{index}", "completed", "2026-07-16T10:00:00Z"))
                for index in range(10)
            ]
            + ["RuntimeError: the configured warehouse is suspended"]
        )
        + "\n",
        encoding="utf-8",
    )
    managed = ManagedProcess(
        run_id="job-1",
        graph_id="graph-1",
        process=_StoppedProcess(),  # type: ignore[arg-type]
        events_path=events,
        output_path=tmp_path / "out",
        stdout_path=run_directory / "stdout.log",
        config_path=run_directory / "config.yaml",
        started_at="2026-07-16T09:59:00Z",
        storage_kind=StorageKind.LOCAL,
        storage_location=str(tmp_path / "out"),
    )

    snapshot = LocalBackend(tmp_path, tmp_path / "state")._snapshot(managed)

    assert snapshot.status == "failed"
    assert snapshot.error == "RuntimeError: the configured warehouse is suspended"


def test_stage_submission_reads_the_stage_through_its_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List the queue's files through Snowflake, whatever an instance carries.

    Submission has exactly one way to discover stage objects. A production path
    that switched on the presence of an instance attribute would let the shape of
    a caller's object decide which statement the warehouse runs.
    """

    session = _Session(
        [_Row({"name": "DB/GRAPH/DOCS/incoming/paper.pdf", "size": 42, "md5": "abc"})]
    )
    backend = SnowflakeBackend(session)
    monkeypatch.setattr(
        backend,
        "list_source_objects",
        lambda *_args, **_kwargs: [],
    )

    source = {"stage": "@DB.GRAPH.DOCS", "prefix": "incoming"}
    items = list(backend._iter_submission_objects(source))

    assert [item.name for item in items] == ["DB/GRAPH/DOCS/incoming/paper.pdf"]
    assert session.statements == ["LIST @DB.GRAPH.DOCS/incoming"]


# --------------------------------------------------------------------------- #
# Page behaviour
# --------------------------------------------------------------------------- #


def _embedding_request(tmp_path: Path, *, dimension: int) -> IngestionRequest:
    """Build a Snowflake-destined request that asks for one embedding width."""

    return dataclasses.replace(
        _request(tmp_path, SourceKind.SNOWFLAKE_STAGE, {"stage": "@DB.GRAPH.DOCS", "prefix": ""}),
        runtime=RuntimeMode.SNOWFLAKE,
        embedding=ProviderSelection(
            "snowflake_cortex",
            model="snowflake-arctic-embed-m-v1.5",
            dimension=dimension,
        ),
    )


def test_the_embedding_model_defaults_to_what_the_destination_can_store() -> None:
    """Offer a run that can be submitted, not one preflight will refuse.

    A Snowflake VECTOR column fixes its width at creation, so a destination that
    stores 1024-wide vectors can never accept the 768-wide default. Left to the
    operator to discover, the form's own defaults produce a run whose only
    outcome is a refusal.
    """

    state: dict[str, object] = {
        "embedding_model": "snowflake-arctic-embed-m-v1.5",
        "embedding_dimension": 768,
    }

    _adopt_stored_embedding_width(cast(Any, state), 1024)

    assert state["embedding_model"] == "snowflake-arctic-embed-l-v2.0"
    assert state["embedding_dimension"] == 1024
    assert cortex_embedding_model_for_width(1024) == "snowflake-arctic-embed-l-v2.0"
    assert EMBEDDING_MODEL_DIMENSIONS["snowflake-arctic-embed-l-v2.0"] == 1024


def test_the_default_model_and_the_default_width_never_disagree() -> None:
    """Move the model with the width, never the width alone.

    Setting only the width leaves the form asking a 768-wide model to fill a
    1024-wide column: preflight compares the declared width and passes, and the
    run is rejected at the write with vectors of the wrong shape — the original
    failure, now reached by a route that looks validated.
    """

    state: dict[str, object] = {}

    _adopt_stored_embedding_width(cast(Any, state), 1024)

    model = str(state["embedding_model"])
    assert EMBEDDING_MODEL_DIMENSIONS[model] == state["embedding_dimension"] == 1024


def test_a_width_with_no_known_model_still_constrains_the_run() -> None:
    """Follow the destination even where no adapter default is known for it.

    The width is what the tables can store, whatever produces the vectors. A
    schema built at a width this application ships no model for still has to be
    written at that width, by whichever provider the operator configures.
    """

    state: dict[str, object] = {
        "embedding_model": "some-provider/embed-v9",
        "embedding_dimension": 768,
    }

    _adopt_stored_embedding_width(cast(Any, state), 3072)

    assert state["embedding_dimension"] == 3072
    # No model is invented for a width this application does not recognize.
    assert state["embedding_model"] == "some-provider/embed-v9"


def test_a_non_cortex_provider_keeps_its_own_model_default() -> None:
    """Leave adapters this destination knows nothing about entirely alone.

    Only a Cortex default is exchanged for a Cortex model of the right width. A
    sentence-transformers or OpenAI-compatible default is not a model whose width
    this application can substitute, so it is offered unchanged.
    """

    for provider, expected in (
        ("sentence_transformers", "sentence-transformers/all-MiniLM-L6-v2"),
        ("openai_compatible", ""),
        ("azure_openai", ""),
    ):
        _endpoint, model_default = DEFAULTS["embedding"][provider]
        fitted = cortex_embedding_model_for_width(1024)
        if fitted and model_default in CORTEX_EMBEDDING_MODELS_BY_WIDTH.values():
            model_default = fitted

        assert model_default == expected, provider


def test_only_this_graphs_tables_decide_the_required_width() -> None:
    """Ignore an embedding column that belongs to something else.

    A Snowflake schema is shared. Another team's vector table in the same schema
    is not evidence about where this graph is written, and reading it would let
    an unrelated design refuse every FlakeGraph run.
    """

    class Session:
        """A schema holding a foreign vector table beside the graph tables."""

        def sql(self, statement: str, params: object = None) -> Any:
            del params
            if statement.startswith("SHOW TABLES"):
                rows = [{"name": name} for name in _REQUIRED_SNOWFLAKE_APP_TABLES]
            elif statement.startswith("SHOW COLUMNS"):
                rows = [
                    {"table_name": "KG_NODE", "data_type": '{"type":"VECTOR","dimension":1024}'},
                    {"table_name": "KG_EDGE", "data_type": '{"type":"VECTOR","dimension":1024}'},
                    # Someone else's table, with a type this check cannot read.
                    {"table_name": "MARKETING_DOCS", "data_type": '{"type":"VARIANT"}'},
                ]
            else:
                rows = []
            return _Query([_Row(row) for row in rows])

    backend = SnowflakeBackend(Session())

    assert backend.graph_embedding_width() == 1024


def test_a_deliberately_chosen_embedding_model_is_not_overwritten() -> None:
    """Move a default aside, never an operator's own choice.

    Only a model this application offered by default is replaced. A model the
    operator typed is left alone, so a destination width cannot silently discard
    a deliberate selection.
    """

    state: dict[str, object] = {
        "embedding_model": "a-custom-deployment-model",
        "embedding_dimension": 512,
    }

    _adopt_stored_embedding_width(cast(Any, state), 1024)

    assert state["embedding_model"] == "a-custom-deployment-model"
    # The width still follows the destination, which is the only writable width.
    assert state["embedding_dimension"] == 1024


def test_a_destination_without_a_fixed_width_leaves_the_selection_alone() -> None:
    """Change nothing where the destination imposes nothing."""

    state: dict[str, object] = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384,
    }

    _adopt_stored_embedding_width(cast(Any, state), None)

    assert state["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert state["embedding_dimension"] == 384


def test_an_embedding_width_the_graph_tables_cannot_store_stops_preflight(
    tmp_path: Path,
) -> None:
    """Refuse the run before Cortex is billed for it.

    A Snowflake VECTOR column fixes its width when the table is created and the
    writer is the last stage of a run, so a mismatch is otherwise discovered only
    after every document has been parsed, extracted and summarised — and the
    whole run is rejected at the write, having already been paid for.
    """

    class Session:
        """A session whose graph tables exist and store 1024-wide vectors."""

        def __init__(self) -> None:
            self.statements: list[str] = []

        def sql(self, statement: str, params: object = None) -> Any:
            del params
            self.statements.append(statement)
            if statement.startswith("SHOW TABLES"):
                rows = [{"name": name} for name in _REQUIRED_SNOWFLAKE_APP_TABLES]
            elif statement.startswith("SHOW COLUMNS"):
                rows = [
                    {
                        "table_name": "KG_NODE",
                        "data_type": '{"type":"VECTOR","dimension":1024}',
                    }
                ]
            else:
                rows = []
            return _Query([_Row(row) for row in rows])

    session = Session()
    request = _embedding_request(tmp_path, dimension=768)

    result = SnowflakeBackend(session).preflight(request)

    assert result["ok"] is False
    checks = {str(item["name"]): bool(item["ok"]) for item in cast(Any, result["checks"])}
    assert checks["embedding_dimension"] is False
    errors = cast(Any, result["errors"])
    assert any("1024" in error and "768" in error for error in errors)
    assert any(statement.startswith("SHOW COLUMNS") for statement in session.statements)


def test_an_unreadable_embedding_width_is_reported_rather_than_assumed(
    tmp_path: Path,
) -> None:
    """Do not pass a run the guard could not actually check.

    Treating an unrecognized column type as agreement is the one outcome this
    check must never produce: it exists to stop a run that would otherwise be
    billed in full and then rejected.
    """

    class Session:
        """A session whose embedding column reports a type with no width."""

        def sql(self, statement: str, params: object = None) -> Any:
            del params
            if statement.startswith("SHOW TABLES"):
                rows = [{"name": name} for name in _REQUIRED_SNOWFLAKE_APP_TABLES]
            elif statement.startswith("SHOW COLUMNS"):
                rows = [{"table_name": "KG_NODE", "data_type": '{"type":"VECTOR"}'}]
            else:
                rows = []
            return _Query([_Row(row) for row in rows])

    result = SnowflakeBackend(Session()).preflight(_embedding_request(tmp_path, dimension=1024))

    assert result["ok"] is False
    assert any("cannot be checked" in error for error in cast(Any, result["errors"]))


def test_a_named_source_stage_reaches_the_worker_fully_qualified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Send a stage the worker can resolve, not the name the operator sees.

    Pickers and Snowsight both show a bare stage name, but the worker resolves it
    from inside its container, where that name has no namespace to resolve
    against. Passed through unqualified it is rejected as an unsafe location, and
    the run fails after the service has started rather than at preflight.
    """

    streamlit = _Streamlit(clicked={"List source"})
    streamlit.values["Input source"] = SourceKind.SNOWFLAKE_STAGE
    streamlit.values["Stage"] = "KG_DOCS"
    streamlit.values["Prefix"] = "verify"

    class Backend:
        capabilities = frozenset({"upload", "snowflake_stage"})

        def list_stages(self) -> list[str]:
            return ["KG_DOCS"]

        def current_context(self) -> Mapping[str, str]:
            return {"database": "DB", "schema": "GRAPH"}

        def list_source_objects(self, *args: object, **kwargs: object) -> list[object]:
            del args, kwargs
            return []

    monkeypatch.setattr("flakegraph_app.ui.ingestion.st", streamlit)

    kind, source = _source_controls(
        Backend(),  # type: ignore[arg-type]
        RuntimeMode.SNOWFLAKE,
        tmp_path,
        "job-1",
    )

    assert kind == SourceKind.SNOWFLAKE_STAGE
    assert source is not None
    # The worker's own validator is the contract this has to satisfy.
    assert stage_path(str(source["stage"]), str(source["prefix"])) == "@DB.GRAPH.KG_DOCS/verify"


def test_a_failed_stage_upload_stays_an_inline_page_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Report an upload failure where the operator asked for the upload.

    An unhandled backend error replaces the ingestion form with a traceback, and
    the selections made to reach that point go with it.
    """

    streamlit = _Streamlit(clicked={"Upload to stage"})
    streamlit.uploads = [_Upload()]
    streamlit.values["Input source"] = SourceKind.UPLOAD

    class Backend:
        capabilities = frozenset({"upload", "snowflake_stage"})

        def list_stages(self) -> list[str]:
            return ["KG_DOCS"]

        def current_context(self) -> Mapping[str, str]:
            return {"database": "DB", "schema": "GRAPH"}

        def upload_files(self, paths: object, stage: str, prefix: str = "uploads") -> int:
            del paths, stage, prefix
            raise RuntimeError("Internal stage KG_DOCS does not exist in this schema.")

    monkeypatch.setattr("flakegraph_app.ui.ingestion.st", streamlit)

    kind, source = _source_controls(
        Backend(),  # type: ignore[arg-type]
        RuntimeMode.SNOWFLAKE,
        tmp_path,
        "job-1",
    )

    assert source is None
    assert kind == SourceKind.UPLOAD
    assert streamlit.errors == ["Internal stage KG_DOCS does not exist in this schema."]


def test_submission_hands_the_run_identity_over_to_the_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Free the identity fields once a run owns them.

    A resubmitted job id merges into a durable record that is inserted only when
    absent, so the earlier status and configuration survive while new files join
    its queue and a second worker starts against settings nobody reviewed.
    """

    request = _request(tmp_path, SourceKind.LOCAL, {"path": str(tmp_path)})
    monkeypatch.setattr("flakegraph_app.ui.ingestion.page_heading", lambda *args: None)
    monkeypatch.setattr("flakegraph_app.ui.ingestion._request_controls", lambda *args: request)

    app = AppTest.from_string(
        """
from pathlib import Path
from flakegraph_app.models import RunSnapshot, RuntimeMode
from flakegraph_app.ui.ingestion import render_ingestion

class Backend:
    capabilities = frozenset({"local"})

    def preflight(self, request):
        return {"ok": True, "errors": []}

    def submit(self, request):
        return RunSnapshot(
            run_id="job-1",
            graph_id="graph-1",
            status="running",
            started_at=None,
            updated_at=None,
        )

render_ingestion(Backend(), RuntimeMode.LOCAL, Path("."))
"""
    )
    app.session_state["ingest_job_id"] = "graph-20260716-000000-aaaaaaaaaaaaaaaa"
    app.session_state["ingest_graph_id"] = "graph-20260716-000000-aaaaaaaaaaaaaaaa"
    app.run()

    app.button[0].click().run()
    app.button[1].click().run()

    assert app.session_state["selected_run_id"] == "job-1"
    assert app.session_state["active_page"] == "run"
    assert "ingest_job_id" not in app.session_state
    assert "ingest_graph_id" not in app.session_state
    # Nothing reads these, and a submission that writes them invites a reader to
    # believe one of them is the app's notion of the current run.
    assert "active_run_id" not in app.session_state
    assert "active_config_path" not in app.session_state


def test_history_confirmations_disarm_when_their_dialog_is_dismissed() -> None:
    """Treat closing a confirmation as a decision, not as an unanswered question.

    The pending selection is what opens the dialog, so a dialog closed with its
    ✕, with Escape, or by clicking outside would otherwise reopen on the next
    interaction anywhere in the sidebar.
    """

    tree = ast.parse((_APP_PACKAGE / "ui" / "navigation.py").read_text(encoding="utf-8"))
    dialogs = {
        node.name: {keyword.arg for keyword in decorator.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call) and getattr(decorator.func, "attr", "") == "dialog"
    }

    assert dialogs
    assert all("on_dismiss" in keywords for keywords in dialogs.values())


def test_dismissing_a_confirmation_clears_every_pending_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave no armed removal behind after a dialog is closed unconfirmed."""

    state: dict[str, Any] = {
        "pending_forget_run_id": "run-1",
        "pending_bulk_forget_local": ["run-2", "run-3"],
        "bulk_forget_errors_local": {"run-2": "backend unavailable"},
        "selected_run_id": "run-9",
    }
    monkeypatch.setattr("flakegraph_app.ui.navigation.st.session_state", state)

    _dismiss_forget()
    _dismiss_bulk_forget()

    assert state == {"selected_run_id": "run-9"}


def test_consumption_filters_are_scoped_to_the_graph_they_describe() -> None:
    """Keep one graph's stage, provider, and model selections off the next graph.

    Streamlit keys a widget by name for the whole session, so shared keys carry a
    selection onto a graph where it matches nothing and the table reports no
    calls for a graph that made thousands.
    """

    app = AppTest.from_string(
        """
from flakegraph_app.ui.consumption import render_consumption

metrics = {
    "consumption": {
        "totals": {"billed_usd": 1.5, "calls": 2, "total_tokens": 40, "pages": 3},
        "by_stage": {"graph_extraction": {"calls": 2, "billed_usd": 1.5}},
        "events": [
            {
                "stage": "graph_extraction",
                "operation": "complete",
                "provider": "snowflake_cortex",
                "model": "mistral-7b",
                "locality": "hosted",
                "calls": 1,
                "pages": 0,
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
        ],
    }
}
render_consumption(metrics, "graph-alpha")
"""
    ).run()

    keys = set(app.session_state.filtered_state)
    assert "cf_stage_graph-alpha" in keys
    assert "consumption_grouping_graph-alpha" in keys
    assert "consumption_sort_graph-alpha" in keys
    assert not {key for key in keys if key in {"cf_stage", "consumption_grouping"}}


def test_a_graph_that_cannot_be_loaded_says_so_in_every_tab() -> None:
    """Report a failed read as a failed read.

    A missing dataset also means missing metrics, and the consumption view's own
    explanation for missing metrics is that the graph predates the feature. Left
    to fall through, a warehouse error is reported as a graph too old to have
    recorded anything.
    """

    app = AppTest.from_string(
        """
from flakegraph_app.models import RunSnapshot
from flakegraph_app.ui.run_workspace import _render_completed_run

class Backend:
    capabilities = frozenset()

    def load_run_graph(self, snapshot, config_path=None):
        raise RuntimeError("Warehouse COMPUTE_WH is suspended")

snapshot = RunSnapshot(
    run_id="run-1",
    graph_id="graph-1",
    status="succeeded",
    started_at=None,
    updated_at="2026-07-16T10:00:00Z",
)
_render_completed_run(Backend(), snapshot, None)
"""
    ).run()

    messages = [item.value for item in app.error]
    assert messages == [
        "The graph could not be loaded: Warehouse COMPUTE_WH is suspended",
        "The graph could not be loaded: Warehouse COMPUTE_WH is suspended",
    ]
    assert not [item.value for item in app.info]


# --------------------------------------------------------------------------- #
# Graph explorer
# --------------------------------------------------------------------------- #


def test_the_confidence_filter_describes_only_what_it_filters() -> None:
    """Promise relation filtering, because that is all confidence can filter.

    Confidence is a property of an asserted relation. Canonical entities carry no
    such score, so a control offering to hide low-confidence entities describes a
    filter the graph cannot support and invites the reader to conclude the
    remaining entities passed it.
    """

    dataset = GraphDataset(
        graph_id="graph-1",
        nodes=[
            {"id": "n1", "name": "Aikido", "primary_type": "Discipline", "degree": 1},
            {"id": "n2", "name": "Ueshiba", "primary_type": "Person", "degree": 1},
        ],
        edges=[
            {
                "id": "e1",
                "source_node_id": "n1",
                "target_node_id": "n2",
                "relation_type": "founded_by",
                "confidence": 0.2,
            }
        ],
    )

    nodes, edges = filter_graph(dataset, minimum_confidence=0.9)

    assert [node["id"] for node in nodes] == ["n1", "n2"]
    assert edges == []

    app = AppTest.from_string(
        """
from flakegraph_app.models import GraphDataset
from flakegraph_app.ui.graph_explorer import render_graph_dataset

render_graph_dataset(GraphDataset(graph_id="graph-1", nodes=[], edges=[]))
"""
    ).run()
    help_text = next(item.help or "" for item in app.slider if item.label == "Minimum confidence")

    assert "Hide entities and relations" not in help_text
    assert "never hidden" in help_text


def test_changing_runtime_does_not_accuse_the_page_of_contradicting_itself() -> None:
    """Switch the adapter defaults without Streamlit reporting a conflict.

    Each runtime has its own default adapters, so changing runtime rewrites the
    provider selections in session state. A selector that also carries a default
    index makes Streamlit print that the widget's default and its stored value
    disagree — a message about this application's internals, shown to an operator
    who only chose where their run should execute.
    """

    app = AppTest.from_string(
        """
from pathlib import Path
import streamlit as st
from flakegraph_app.models import RuntimeMode
from flakegraph_app.ui.ingestion import _provider_defaults, _synchronize_runtime_provider_state
from flakegraph_app.providers import EMBEDDING_PROVIDERS, LLM_PROVIDERS, OCR_PROVIDERS
from flakegraph_app.ui.shared import provider_controls

runtime = RuntimeMode[st.session_state.get("runtime_name", "LOCAL")]
defaults = _provider_defaults(runtime, Path("missing-profile.yaml"))
_synchronize_runtime_provider_state(st.session_state, runtime, defaults)
st.session_state["chosen"] = {
    "ocr": provider_controls("OCR", "ocr", OCR_PROVIDERS, defaults["ocr"]).provider,
    "llm": provider_controls("LLM", "llm", LLM_PROVIDERS, defaults["llm"]).provider,
    "embedding": provider_controls(
        "Embeddings", "embedding", EMBEDDING_PROVIDERS, defaults["embedding"]
    ).provider,
}
"""
    ).run(timeout=60)

    assert not app.exception
    seen = [(RuntimeMode.LOCAL, app.session_state["chosen"], [item.value for item in app.warning])]
    for runtime in (RuntimeMode.SNOWFLAKE, RuntimeMode.KUBERNETES, RuntimeMode.LOCAL):
        app.session_state["runtime_name"] = runtime.name
        app.run(timeout=60)
        assert not app.exception, runtime
        seen.append((runtime, app.session_state["chosen"], [item.value for item in app.warning]))

    for runtime, chosen, warnings in seen:
        assert warnings == [], (runtime, warnings)
        # The selection still follows the runtime, which is what the reset is for,
        # and every adapter stays chosen: this form has no meaning for an empty one.
        assert set(chosen) == {"ocr", "llm", "embedding"}, runtime
        assert all(chosen.values()), (runtime, chosen)
    assert seen[1][1] != seen[0][1]


def test_unconnected_entities_do_not_squeeze_relations_out_of_sight() -> None:
    """Keep every drawn relation longer than the markers drawn on top of it.

    Entities with no relations are repelled by the layout and pulled back by
    nothing, so a single solve over the whole graph lets them set the extent and
    compresses the connected entities into it. Past a point the relation between
    two entities is shorter than the two markers covering its ends, and a graph
    that has relations is presented as a field of loose dots.
    """

    nodes = [
        {"id": f"n{index}", "name": f"Entity {index}", "primary_type": "Concept", "degree": 1}
        for index in range(60)
    ]
    edges = [
        {
            "id": f"e{index}",
            "source_node_id": f"n{index}",
            "target_node_id": f"n{index + 1}",
            "relation_type": "related_to",
            "confidence": 0.9,
        }
        # Twelve connected pairs among thirty-six entities that stand alone.
        for index in range(0, 24, 2)
    ]

    figure = build_graph_figure(nodes, edges)
    relations, entities = figure.data[0], figure.data[2]
    span = max(
        max(entities.x) - min(entities.x),
        max(entities.y) - min(entities.y),
    )
    drawn = [
        math.dist(
            (relations.x[start], relations.y[start]),
            (relations.x[start + 1], relations.y[start + 1]),
        )
        # Plotly separates each line segment from the next with a None gap.
        for start in range(0, len(relations.x), 3)
    ]

    assert len(drawn) == len(edges)
    # The largest marker is 28px across on a 680px canvas, so a relation shorter
    # than 5% of the extent is drawn entirely underneath its own endpoints.
    assert min(drawn) > span * 0.05


def test_one_graph_is_drawn_the_same_way_in_every_process() -> None:
    """Keep a graph's positions fixed across processes, not just within one.

    Both solvers place nodes by their index in the graph's iteration order, and a
    subgraph view iterates a set once its group is small relative to the whole
    graph. Left that way, the canvas is cached per process, so a restart or a
    second replica redraws the same graph differently and the operator reads the
    move as a change in the data.
    """

    script = (
        "import json, sys\n"
        "sys.path.insert(0, 'app')\n"
        "from flakegraph_app.explorer import _positions, _projection\n"
        # Eleven entities standing alone against a ten-entity chain puts the
        # group under half the graph, which is where the view switches to set
        # order.
        "nodes = [{'id': f'n{i}'} for i in range(21)]\n"
        "edges = [{'source_node_id': f'n{i}', 'target_node_id': f'n{i + 1}'} for i in range(9)]\n"
        "positions = _positions(_projection(nodes, edges), 'Communities')\n"
        "rounded = {k: [round(v[0], 9), round(v[1], 9)] for k, v in sorted(positions.items())}\n"
        "print(json.dumps(rounded))"
    )
    drawn = {
        subprocess.run(
            [sys.executable, "-c", script],
            cwd=_ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("0", "1", "2", "12345")
    }

    assert len(drawn) == 1


def test_an_installed_solver_draws_its_layout_without_comment() -> None:
    """Say nothing when the layout on screen is the layout that was asked for.

    The fallback notice tells the reader their positions carry no meaning. Raised
    where the solver is present and the positions are real, it would discredit an
    accurate picture of the graph.
    """

    chain = _connected_chain(600)

    assert layout_fallback_notice("Hierarchy", *chain) is None
    assert layout_fallback_notice("Communities", *chain) is None


def test_a_layout_that_falls_back_to_a_ring_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name the fallback, because a ring is not a degraded hierarchy.

    Positions are the whole message of a layout. Rendered silently, a graph whose
    hierarchy and communities are the same evenly spaced circle reads as a
    finding about the graph rather than about a missing solver.
    """

    monkeypatch.setattr("flakegraph_app.explorer._numeric_layout_available", lambda: False)

    assert "ring" in (layout_fallback_notice("Hierarchy", *_connected_chain(10)) or "")
    assert "ring" in (layout_fallback_notice("Communities", *_connected_chain(600)) or "")
    # NetworkX reaches for the sparse solver AT this size, not above it, so the
    # boundary group is drawn as a ring and has to say so.
    assert "ring" in (layout_fallback_notice("Communities", *_connected_chain(500)) or "")
    assert layout_fallback_notice("Communities", *_connected_chain(499)) is None


def test_a_layout_small_enough_to_solve_without_scipy_stays_quiet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warn about the solver that is actually reached, not the one that is not.

    Layouts are solved per connected group, so an entity total far above the
    sparse-solver threshold still solves densely when it is made of small groups.
    Warning there would report a ring the reader is not being shown.
    """

    monkeypatch.setattr("flakegraph_app.explorer._numeric_layout_available", lambda: False)
    nodes = [{"id": f"n{index}"} for index in range(10_000)]
    pairs = [
        {"source_node_id": f"n{index}", "target_node_id": f"n{index + 1}"}
        for index in range(0, 10_000, 2)
    ]

    assert layout_fallback_notice("Communities", nodes, pairs) is None


def _connected_chain(size: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build one connected group of ``size`` entities for layout assertions."""

    nodes = [
        {"id": f"n{index}", "name": f"Entity {index}", "primary_type": "Person", "degree": 2}
        for index in range(size)
    ]
    edges: list[dict[str, object]] = [
        {"id": f"e{index}", "source_node_id": f"n{index}", "target_node_id": f"n{index + 1}"}
        for index in range(size - 1)
    ]
    return nodes, edges


def test_a_layout_without_its_numeric_solver_warns_on_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface the fallback where the graph is being read."""

    monkeypatch.setattr(
        "flakegraph_app.explorer._numeric_layout_available",
        lambda: False,
    )

    assert layout_fallback_notice("Radial", *_connected_chain(600)) is None
    assert "SciPy" in (layout_fallback_notice("Hierarchy", *_connected_chain(10)) or "")
    assert "SciPy" in (layout_fallback_notice("Communities", *_connected_chain(600)) or "")

    app = AppTest.from_string(
        """
from flakegraph_app.models import GraphDataset
from flakegraph_app.ui.graph_explorer import render_graph_dataset

dataset = GraphDataset(
    graph_id="graph-1",
    nodes=[
        {"id": f"n{index}", "name": f"Entity {index}", "primary_type": "Person", "degree": 2}
        # One connected group above the size at which the community layout is
        # solved numerically; the threshold applies per group, not per graph.
        for index in range(501)
    ],
    edges=[
        {"id": f"e{index}", "source_node_id": f"n{index}", "target_node_id": f"n{index + 1}"}
        for index in range(500)
    ],
)
render_graph_dataset(dataset)
"""
    ).run(timeout=60)

    assert any("SciPy" in item.value for item in app.warning)


def test_the_snowflake_environment_installs_the_layout_solver() -> None:
    """Ship the solver the explorer's layouts require.

    Snowflake resolves the deployed app's dependencies from this file alone, so
    an omission here is not a slower layout: it is a hierarchy that is always a
    circle, on the one deployment nobody can pip-install into.
    """

    environment = yaml.safe_load((_ROOT / "app" / "environment.yml").read_text(encoding="utf-8"))
    packages = {
        str(item).split("=")[0].strip()
        for item in environment["dependencies"]
        if isinstance(item, str)
    }

    assert "scipy" in packages


# --------------------------------------------------------------------------- #
# Module surface
# --------------------------------------------------------------------------- #


def test_source_browsers_expose_only_helpers_the_application_reaches() -> None:
    """Keep the read-only source module free of surface nothing calls.

    An unreferenced helper reads as a supported capability. Anyone extending a
    source browser has to establish that it is not one before they can ignore it.
    """

    module = _APP_PACKAGE / "sources.py"
    declared = {
        node.name
        for node in ast.parse(module.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }
    callers = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [*_APP_PACKAGE.rglob("*.py"), *(_ROOT / "tests").rglob("*.py")]
        if path != module
    )

    assert declared
    assert {name for name in declared if name not in callers} == set()


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class _Row:
    """One Snowpark result row addressed by column name."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self.values = dict(values)

    def as_dict(self) -> dict[str, Any]:
        """Return the row's columns the way Snowpark presents them."""

        return dict(self.values)


class _Query:
    """A prepared statement that yields a fixed result."""

    def __init__(self, rows: Sequence[_Row]) -> None:
        self.rows = list(rows)

    def limit(self, count: int) -> _Query:
        """Bound the result the way Snowpark's dataframe API does."""

        return _Query(self.rows[:count])

    def collect(self) -> list[_Row]:
        """Materialize the result."""

        return list(self.rows)


class _Session:
    """A Snowpark session that answers every statement with the same rows."""

    def __init__(self, rows: Sequence[_Row]) -> None:
        self.rows = list(rows)
        self.statements: list[str] = []

    def sql(self, statement: str, params: object = None) -> _Query:
        """Record one statement and return its canned result."""

        del params
        self.statements.append(statement)
        return _Query(self.rows)


class _StoppedProcess:
    """A child process that has already exited unsuccessfully."""

    def poll(self) -> int:
        """Report the non-zero exit status of a failed worker."""

        return 1


class _Upload:
    """One Streamlit upload selection entry."""

    name = "martial-arts.pdf"
    size = 7
    file_id = "upload-1"

    def getvalue(self) -> bytes:
        """Return the uploaded bytes."""

        return b"content"


class _Streamlit:
    """The Streamlit surface one page function uses, with recorded output."""

    def __init__(self, clicked: set[str] | None = None) -> None:
        self.session_state: dict[str, Any] = {}
        self.errors: list[str] = []
        self.infos: list[str] = []
        self.values: dict[str, Any] = {}
        self.clicked = set(clicked or ())
        self.uploads: list[Any] = []

    def __enter__(self) -> _Streamlit:
        """Enter a layout container."""

        return self

    def __exit__(self, *exception: object) -> None:
        """Leave a layout container."""

        return None

    def columns(self, spec: Any, **kwargs: Any) -> list[_Streamlit]:
        """Return one placeholder per requested column."""

        del kwargs
        count = spec if isinstance(spec, int) else len(spec)
        return [self] * count

    def segmented_control(self, label: str, options: Sequence[Any], **kwargs: Any) -> Any:
        """Return the chosen option for one segmented control."""

        del kwargs
        return self.values.get(label, options[0] if options else None)

    def selectbox(self, label: str, options: Sequence[Any], **kwargs: Any) -> Any:
        """Return the chosen option for one select control."""

        index = int(kwargs.get("index", 0) or 0)
        return self.values.get(label, options[index] if options else "")

    def text_input(self, label: str, value: str = "", **kwargs: Any) -> str:
        """Return the current text for one field."""

        del kwargs
        return str(self.values.get(label, value))

    def file_uploader(self, label: str, **kwargs: Any) -> list[Any]:
        """Return the current upload selection."""

        del label, kwargs
        return self.uploads

    def button(self, label: str = "", **kwargs: Any) -> bool:
        """Report whether one labelled button was pressed."""

        del kwargs
        return label in self.clicked

    def error(self, message: object) -> None:
        """Record one page error."""

        self.errors.append(str(message))

    def info(self, message: object) -> None:
        """Record one page notice."""

        self.infos.append(str(message))

    def caption(self, *args: object, **kwargs: object) -> None:
        """Ignore captions."""

        del args, kwargs

    def dataframe(self, *args: object, **kwargs: object) -> None:
        """Ignore tables."""

        del args, kwargs

    @contextmanager
    def spinner(self, *args: object, **kwargs: object) -> Iterator[None]:
        """Run the guarded block without a progress indicator."""

        del args, kwargs
        yield


def _ocr_event(file_id: str, status: str, timestamp: str) -> dict[str, Any]:
    """Build one document-level progress record for the OCR stage."""

    return {
        "event": "kg_processor.progress",
        "timestamp": timestamp,
        "stage": "ocr",
        "status": status,
        "file_id": file_id,
    }


def _request(
    tmp_path: Path,
    source_kind: SourceKind,
    source: dict[str, str],
) -> IngestionRequest:
    """Build one complete local request with only non-secret provider data."""

    return IngestionRequest(
        runtime=RuntimeMode.LOCAL,
        job_id="job-1",
        graph_id="graph-1",
        source_kind=source_kind,
        source=source,
        ocr=ProviderSelection("fallback"),
        llm=ProviderSelection(
            "openai_compatible",
            model="model",
            endpoint="https://llm.example/v1",
            api_key_environment_variable="TEST_LLM_KEY",
        ),
        embedding=ProviderSelection(
            "sentence_transformers",
            model="sentence-transformers/all-MiniLM-L6-v2",
        ),
        output=OutputDestination(StorageKind.LOCAL, tmp_path / "out"),
    )


def test_an_unchanged_log_leaves_its_checkpoint_alone(tmp_path: Path) -> None:
    """A run page polls every few seconds while the log is often unchanged.

    The checkpoint holds one entry per file per stage so a retried file counts
    once, which makes it proportional to the corpus. Rewriting it on every poll
    would pay that cost repeatedly for no new information.
    """

    log = tmp_path / "events.jsonl"
    log.write_text(
        '{"event": "kg_processor.progress", "timestamp": "2026-01-01T00:00:00Z", '
        '"stage": "ocr", "status": "completed", "file_id": "file-1"}\n',
        encoding="utf-8",
    )

    read_local_progress(log)
    checkpoint = tmp_path / "progress-summary.json"
    first = checkpoint.stat().st_mtime_ns

    read_local_progress(log)

    assert checkpoint.stat().st_mtime_ns == first, "an unchanged log rewrote its checkpoint"

    log.write_text(
        log.read_text(encoding="utf-8")
        + '{"event": "kg_processor.progress", "timestamp": "2026-01-01T00:00:01Z", '
        '"stage": "ocr", "status": "completed", "file_id": "file-2"}\n',
        encoding="utf-8",
    )
    progress = read_local_progress(log)

    assert checkpoint.stat().st_mtime_ns != first, "new records must reach the checkpoint"
    assert progress.documents_completed == 2


def test_a_field_the_form_never_shows_still_blocks_an_unclaimable_run() -> None:
    """The digest covers whole sections, so a named-field check is not enough.

    These fields once passed preflight and left the run queued forever: the form
    does not show them, and the comparison did not cover them. Only keys both
    sides state are compared, so mineru_backend — which the fleet omitted — is
    not caught here; either of the other two is enough to block the run.
    """

    effective = {
        "ocr": {"provider": "fallback", "mineru_method": "ocr", "mineru_backend": "pipeline"},
        "graph": {"fail_on_quality_error": True},
    }
    deployed = {
        "ocr": {"provider": "fallback", "mineru_method": "auto"},
        "graph": {"fail_on_quality_error": False},
    }
    problems = _semantic_mismatches(effective, deployed)
    joined = " ".join(problems)
    assert "ocr.mineru_method" in joined
    assert "graph.fail_on_quality_error" in joined


def test_matching_configurations_report_nothing() -> None:
    """A run that equals its fleet must not be blocked by noise."""

    shared = {
        "ocr": {"provider": "builtin_text"},
        "llm": {"provider": "vllm_local", "model": "qwen"},
        "graph": {"fail_on_quality_error": True},
    }
    assert _semantic_mismatches(shared, shared) == []


def test_deployment_local_differences_are_not_reported() -> None:
    """Endpoints and parallelism never prevent a claim, so they are not errors."""

    effective = {
        "llm": {"provider": "vllm_local", "endpoint": "http://localhost:8000/v1"},
        "graph": {"extraction_parallelism": 16},
    }
    deployed = {
        "llm": {"provider": "vllm_local", "endpoint": "http://vllm:8000/v1"},
        "graph": {"extraction_parallelism": 8},
    }
    assert _semantic_mismatches(effective, deployed) == []


_FLEET_OCR_SECTION = {
    "provider": "fallback",
    "fallback_primary_provider": "builtin_text",
    "fallback_secondary_provider": "mineru_api",
    "mineru_backend": "pipeline",
    "mineru_method": "ocr",
}


class _FleetOcrBackend:
    """One backend that answers with a deployed document plane's OCR routing."""

    def fleet_ocr_options(self) -> dict[str, object]:
        """Return the routing plus the endpoint read from the worker contract."""

        return {
            key: value for key, value in _FLEET_OCR_SECTION.items() if key != "provider"
        } | {"mineru_api_url": "http://flakegraph-flakegraph-ocr:8080"}


def test_a_fleet_submission_carries_the_deployed_parsing_route(tmp_path: Path) -> None:
    """The routing under a fallback provider is not on the form but is in the digest.

    A fleet parses through a deployed document plane over HTTP; the base profile
    names the in-process parser a local run should use. Left alone, a submission
    would carry the local one, hash to something no worker shares, and sit queued
    with nothing on the page to explain it.
    """

    repository = tmp_path / "repository"
    (repository / "configs").mkdir(parents=True)
    profile = repository / "configs" / "app-defaults.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "ocr": {
                    "provider": "fallback",
                    "fallback_primary_provider": "builtin_text",
                    "fallback_secondary_provider": "mineru_internal",
                    "mineru_backend": "pipeline",
                    "mineru_method": "ocr",
                }
            }
        ),
        encoding="utf-8",
    )
    request = dataclasses.replace(
        _request(tmp_path, SourceKind.LOCAL, {"kind": "local", "path": str(tmp_path)}),
        ocr=_with_fleet_ocr_routing(ProviderSelection("fallback"), _FleetOcrBackend(), "fallback"),
        base_config_path=profile,
    )

    section = build_run_config(request)["ocr"]

    assert section["fallback_secondary_provider"] == "mineru_api"
    # The endpoint is transport, so it never enters the digest, but the run has
    # to state it or the submitting host's own preflight rejects the run before
    # any worker sees it.
    assert section["mineru_api_url"] == "http://flakegraph-flakegraph-ocr:8080"
    assert _semantic_mismatches({"ocr": section}, {"ocr": _FLEET_OCR_SECTION}) == []


def test_an_operator_who_chose_another_parser_keeps_it() -> None:
    """A stated choice is reported by preflight, never silently rewritten."""

    chosen = ProviderSelection("builtin_text")

    assert _with_fleet_ocr_routing(chosen, _FleetOcrBackend(), "fallback") is chosen
    # A runtime with no fleet to read leaves the base profile's routing in place,
    # which is the routing a local run should execute.
    assert _with_fleet_ocr_routing(chosen, object(), "builtin_text") is chosen
