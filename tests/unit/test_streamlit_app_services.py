"""High-value tests for the standalone Streamlit control-plane services."""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import pytest
import yaml
from flakegraph_app.backends.kubernetes import (
    KubernetesBackend,
    _cluster_service,
    _command_failure_message,
    _database_cluster_service,
    _distributed_overview_snapshot,
    _distributed_snapshot,
    _fleet_artifact_environment,
    _fleet_database_environment,
    _fleet_database_url,
    _fleet_preflight,
    _gpu_utilization_by_node,
    _node_status,
    _pod_status,
    _recover_kubernetes_workers,
    _replace_url_host,
    _service_port_forward_target,
)
from flakegraph_app.backends.local import LocalBackend
from flakegraph_app.backends.snowflake import (
    SnowflakeBackend,
    _as_results,
    _job_error_text,
    _stage_location,
)
from flakegraph_app.cluster_catalog import ClusterTarget
from flakegraph_app.configuration import (
    _load_base_config_cached,
    build_run_config,
    container_resources,
    environment_for_request,
    load_base_config,
    provider_parallelism_settings,
    redacted_config,
)
from flakegraph_app.explorer import build_graph_figure, filter_graph
from flakegraph_app.graph_store import load_local_graph
from flakegraph_app.models import (
    GraphDataset,
    IngestionRequest,
    OutputDestination,
    ProgressEvent,
    ProviderSelection,
    RunSnapshot,
    RuntimeMode,
    SnowflakeOutput,
    SourceKind,
    StorageKind,
    WorkloadStatus,
)
from flakegraph_app.progress import aggregate_stages, read_jsonl_events
from flakegraph_app.providers import (
    EMBEDDING_PROVIDERS,
    LLM_PROVIDERS,
    OCR_PROVIDERS,
)
from flakegraph_app.run_catalog import (
    graph_name,
    hidden_run_ids,
    hide_run,
    rename_graph,
    validate_graph_name,
    write_run_record,
)
from flakegraph_app.sources import SUPPORTED_SUFFIXES as APP_SUPPORTED_SUFFIXES
from flakegraph_app.spcs import build_spcs_launch
from flakegraph_app.ui.graph_explorer import _review_frame
from flakegraph_app.ui.ingestion import (
    _default_profile,
    _discovered_field,
    _persist_uploads,
    _qualified_stage,
    render_ingestion,
)
from flakegraph_app.ui.navigation import (
    _filter_runs,
    _forget_runs,
    _history_page,
    _selected_bulk_ids,
)
from flakegraph_app.ui.run_workspace import _needs_recovery, _requires_status_refresh
from flakegraph_app.ui.shared import (
    _format_duration,
    concise_error,
    format_relative_time,
    secret_text_input,
)
from streamlit.testing.v1 import AppTest
from streamlit_app import _configured_default_runtime

from kg_processor.adapters.files.common import SUPPORTED_SUFFIXES as CORE_SUPPORTED_SUFFIXES
from kg_processor.config.provider_registry import provider_names
from kg_processor.config.settings import GraphSettings, Settings

_ROOT = Path(__file__).resolve().parents[2]
_APP_ROOT = _ROOT / "app"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-07-15T11:59:56Z", "Just now"),
        ("2026-07-15T11:59:20+00:00", "40 seconds ago"),
        ("2026-07-15T11:55:00+00:00", "5 minutes ago"),
        ("2026-07-15T10:00:00+00:00", "2 hours ago"),
        ("2026-07-13T12:00:00+00:00", "2 days ago"),
        (None, "Not reported"),
        ("not-a-date", "Not reported"),
    ],
)
def test_fleet_task_timestamps_are_operator_friendly(
    value: str | None,
    expected: str,
) -> None:
    """Keep active-work ages concise while tolerating missing provider timestamps."""

    assert format_relative_time(value, now=datetime(2026, 7, 15, 12, tzinfo=UTC)) == expected


@pytest.mark.parametrize(
    ("elapsed_ms", "expected"),
    [(900, "1s"), (14_600, "15s"), (125_000, "2m 5s"), (7_505_000, "2h 5m")],
)
def test_pipeline_durations_are_compact(elapsed_ms: int, expected: str) -> None:
    """Keep stage timings readable from seconds through multi-hour corpus runs."""

    assert _format_duration(elapsed_ms) == expected


def test_app_sources_are_python_311_compatible() -> None:
    """Protect Snowflake deployment from accidental Python 3.12+ syntax."""

    for path in sorted(_APP_ROOT.rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 11))


def test_app_provider_catalog_matches_processing_core() -> None:
    """Prevent selectable UI adapters from drifting beyond actual core providers."""

    assert {item.name for item in OCR_PROVIDERS} <= provider_names("ocr")
    assert {item.name for item in LLM_PROVIDERS} <= provider_names("llm")
    assert {item.name for item in EMBEDDING_PROVIDERS} <= provider_names("embedding")


def test_app_and_processing_core_support_the_same_file_types() -> None:
    """Keep source listing and upload controls consistent with worker discovery."""

    assert APP_SUPPORTED_SUFFIXES == CORE_SUPPORTED_SUFFIXES


def test_app_uses_one_domain_neutral_base_profile_for_every_runtime() -> None:
    """Prevent an app runtime from silently inheriting a benchmark corpus policy."""

    expected = _ROOT / "configs" / "app-defaults.yaml"
    profiles = {_default_profile(runtime, _ROOT) for runtime in RuntimeMode}
    config = yaml.safe_load(expected.read_text(encoding="utf-8"))
    resolved = Settings.load(expected, env={})

    assert profiles == {expected}
    assert "job" not in config
    assert "graph" not in config
    assert config["files"]["input_path"] == "data"
    assert config["ocr"]["provider"] == "fallback"
    assert config["ontology"]["profile_path"] == "configs/ontologies/general.yaml"
    assert resolved.files.input_path == Path("data")
    assert resolved.graph == GraphSettings()


def test_secret_text_input_is_masked_until_explicitly_revealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent credential fields from becoming visible by default after UI changes."""

    observed: dict[str, Any] = {}

    def text_input(label: str, **kwargs: object) -> str:
        observed["input"] = {"label": label, **kwargs}
        return "KG_LLM_API_KEY"

    monkeypatch.setattr("flakegraph_app.ui.shared.st.text_input", text_input)

    result = secret_text_input(
        "API key environment variable",
        key="llm_credential",
        help_text="Credential reference",
    )

    assert result == "KG_LLM_API_KEY"
    # The field reads its value from session state under its own key; passing a
    # default alongside a keyed widget is what Streamlit reports as a conflict.
    assert observed["input"] == {
        "label": "API key environment variable",
        "type": "password",
        "key": "llm_credential",
        "help": "Credential reference",
    }


def test_invalid_credential_reference_stays_an_inline_form_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep malformed credential input from crashing the complete Streamlit page."""

    request = _request(tmp_path, SourceKind.LOCAL, {"path": str(tmp_path)})
    request = IngestionRequest(
        **{
            **request.__dict__,
            "llm": ProviderSelection(
                "openai_compatible",
                model="model",
                endpoint="https://llm.example/v1",
                api_key_environment_variable="not a variable name",
            ),
        }
    )
    errors: list[str] = []
    monkeypatch.setattr("flakegraph_app.ui.ingestion.page_heading", lambda *args: None)
    monkeypatch.setattr(
        "flakegraph_app.ui.ingestion._request_controls",
        lambda *args: request,
    )
    monkeypatch.setattr("flakegraph_app.ui.ingestion.st.error", errors.append)

    render_ingestion(
        LocalBackend(tmp_path, tmp_path / "state"),
        RuntimeMode.LOCAL,
        tmp_path,
    )

    assert errors == ["Invalid credential environment variable name: not a variable name"]


def test_generated_config_uses_secret_references_and_s3_source(tmp_path: Path) -> None:
    request = _request(tmp_path, SourceKind.S3, {"bucket": "docs", "prefix": "incoming"})

    config = build_run_config(request)

    assert config["files"]["source"] == "s3"
    assert config["s3"]["bucket"] == "docs"
    assert config["llm"]["api_key"] == "${TEST_LLM_KEY}"
    assert "secret-value" not in yaml.safe_dump(config)
    assert redacted_config({"api_key": "${TEST_LLM_KEY}"})["api_key"] == "${REDACTED}"


def test_snowflake_destination_keeps_secret_value_out_of_generated_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist connection coordinates and credential references, never credentials."""

    request = _request(tmp_path, SourceKind.LOCAL, {"path": str(tmp_path)})
    request = IngestionRequest(
        **{
            **request.__dict__,
            "output": OutputDestination(
                StorageKind.SNOWFLAKE,
                tmp_path / "workspace",
                SnowflakeOutput(
                    account="account",
                    user="user",
                    database="DB",
                    schema="GRAPH",
                    warehouse="WH",
                    bulk_stage="@DB.GRAPH.LOAD_STAGE",
                    credential_environment_variable="TEST_SNOWFLAKE_PASSWORD",
                    credential_field="password",
                ),
            ),
        }
    )
    monkeypatch.setenv("TEST_SNOWFLAKE_PASSWORD", "secret-value")
    monkeypatch.setenv("TEST_LLM_KEY", "llm-secret")

    config = build_run_config(request)
    environment = environment_for_request(request)

    assert config["writer"]["provider"] == "snowflake_bulk"
    assert config["snowflake"]["password"] == "${TEST_SNOWFLAKE_PASSWORD}"
    assert config["snowflake"]["password_environment_variable"] == ("TEST_SNOWFLAKE_PASSWORD")
    assert "secret-value" not in yaml.safe_dump(config)
    assert environment["TEST_SNOWFLAKE_PASSWORD"] == "secret-value"


def test_graph_history_filters_by_search_and_storage_without_reordering() -> None:
    """Keep sidebar discovery predictable as catalogs grow across destinations."""

    runs = [
        RunSnapshot(
            "run-b",
            "beta",
            "succeeded",
            None,
            None,
            graph_name="Customer Research",
        ),
        RunSnapshot(
            "run-a",
            "alpha",
            "succeeded",
            None,
            None,
            storage_kind=StorageKind.SNOWFLAKE,
            storage_location="DB.GRAPH",
        ),
    ]

    assert [run.run_id for run in _filter_runs(runs, "", "All")] == ["run-b", "run-a"]
    assert [run.run_id for run in _filter_runs(runs, "customer", "All")] == ["run-b"]
    assert [run.run_id for run in _filter_runs(runs, "beta", "All")] == ["run-b"]
    assert [run.run_id for run in _filter_runs(runs, "db.graph", "All")] == ["run-a"]
    assert [run.run_id for run in _filter_runs(runs, "", StorageKind.LOCAL)] == ["run-b"]


def test_local_graph_names_are_validated_and_persisted_atomically(tmp_path: Path) -> None:
    """Keep friendly names durable without changing a graph's stable identifier."""

    assert rename_graph(tmp_path, "graph-1", "  Protein\nInteractions  ") == (
        "Protein Interactions"
    )
    assert graph_name(tmp_path, "graph-1") == "Protein Interactions"
    assert json.loads((tmp_path / "graph-names.json").read_text(encoding="utf-8")) == {
        "graph-1": "Protein Interactions"
    }
    assert not (tmp_path / "graph-names.json.tmp").exists()


@pytest.mark.parametrize("value", ["", " \n ", "x" * 121, "name\x00value"])
def test_graph_name_validation_rejects_unsafe_values(value: str) -> None:
    """Reject names that cannot be rendered consistently across app backends."""

    with pytest.raises(ValueError, match="Graph name"):
        validate_graph_name(value)


def test_run_snapshot_falls_back_to_stable_graph_id() -> None:
    """Always provide a usable title for graphs created before rename metadata."""

    snapshot = RunSnapshot("run-1", "graph-1", "succeeded", None, None)

    assert snapshot.display_name == "graph-1"


def test_generated_config_maps_provider_specific_ocr_fields(tmp_path: Path) -> None:
    request = _request(tmp_path, SourceKind.LOCAL, {"path": str(tmp_path)})
    request = IngestionRequest(
        **{
            **request.__dict__,
            "ocr": ProviderSelection(
                "generic_http",
                endpoint="https://ocr.example/v1",
                api_key_environment_variable="TEST_OCR_KEY",
            ),
        }
    )

    config = build_run_config(request)

    assert config["ocr"] == {"provider": "generic_http"}
    assert config["generic_http_ocr"] == {
        "endpoint": "https://ocr.example/v1",
        "api_key": "${TEST_OCR_KEY}",
    }


def test_progress_reader_tails_logs_and_aggregates_stage_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep status polling bounded even after a worker has produced a large log."""

    path = tmp_path / "events.jsonl"
    path.write_text(
        ("ordinary provider log\n" * 10_000)
        + json.dumps(
            {
                "event": "kg_processor.progress",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "stage": "ocr",
                "status": "completed",
                "file_id": "file-1",
                "elapsed_ms": 1250,
                "counts": {"files_total": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("progress polling must not read the complete file")
        ),
    )

    events = read_jsonl_events(path, limit=50)
    stages = aggregate_stages(events)

    assert len(events) == 1
    assert stages[0].stage == "ocr"
    assert stages[0].completed == 1
    assert stages[0].total == 2
    assert stages[0].elapsed_ms == 1250


def test_base_configuration_cache_returns_isolated_copies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Parse unchanged YAML once without allowing callers to mutate cached state."""

    profile = tmp_path / "profile.yaml"
    profile.write_text("llm:\n  provider: ollama\n", encoding="utf-8")
    _load_base_config_cached.cache_clear()
    calls = 0
    original = yaml.safe_load

    def counted_load(value: str) -> object:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr("flakegraph_app.configuration.yaml.safe_load", counted_load)

    first = load_base_config(profile)
    first["llm"]["provider"] = "changed"
    second = load_base_config(profile)

    assert second["llm"]["provider"] == "ollama"
    assert calls == 1


def test_progress_stages_follow_the_processing_pipeline_order() -> None:
    """Present completion events in pipeline order even when logs arrive unsorted."""

    events = [
        ProgressEvent("2026-01-01T00:00:03Z", "write", "completed"),
        ProgressEvent("2026-01-01T00:00:01Z", "ocr", "completed", file_id="file-1"),
        ProgressEvent("2026-01-01T00:00:02Z", "graph_extraction", "completed"),
    ]

    stages = aggregate_stages(events)

    assert [stage.stage for stage in stages] == ["ocr", "graph_extraction", "write"]


def test_progress_uses_discovery_and_batch_counters() -> None:
    """Expose meaningful totals for stages that do not emit one event per document."""

    stages = aggregate_stages(
        [
            ProgressEvent(
                "2026-01-01T00:00:01Z",
                "file_source",
                "completed",
                counts={"files_seen": 10},
            ),
            ProgressEvent(
                "2026-01-01T00:00:02Z",
                "graph_extraction",
                "progress",
                counts={"batches_completed": 8, "batches_total": 24},
            ),
        ]
    )

    assert [(stage.stage, stage.completed, stage.total) for stage in stages] == [
        ("file_source", 10, 10),
        ("graph_extraction", 8, 24),
    ]


def test_graph_filters_preserve_endpoint_integrity_and_community_selection() -> None:
    dataset = GraphDataset(
        graph_id="graph",
        nodes=[
            {"id": "a", "name": "Alpha", "primary_type": "Method", "degree": 2},
            {"id": "b", "name": "Beta", "primary_type": "Person", "degree": 1},
            {"id": "c", "name": "Gamma", "primary_type": "Method", "degree": 0},
        ],
        edges=[
            {
                "id": "e1",
                "source_node_id": "a",
                "target_node_id": "b",
                "relation_type": "USES",
                "confidence": 0.9,
            },
            {
                "id": "e2",
                "source_node_id": "b",
                "target_node_id": "c",
                "relation_type": "CREATED",
                "confidence": 0.4,
            },
        ],
        communities=[{"id": "community-1", "member_node_ids": ["a", "b"]}],
    )

    nodes, edges = filter_graph(
        dataset,
        community_ids=["community-1"],
        relation_types=["USES"],
        minimum_confidence=0.8,
    )

    assert {node["id"] for node in nodes} == {"a", "b"}
    assert [edge["id"] for edge in edges] == ["e1"]


def test_local_graph_reader_omits_embeddings_and_uses_full_artifact_contract(
    tmp_path: Path,
) -> None:
    pd.DataFrame(
        [{"id": "node-1", "graph_id": "graph-1", "name": "Node", "embedding": [0.1]}]
    ).to_parquet(tmp_path / "nodes.parquet")
    pd.DataFrame(
        [
            {
                "id": "edge-1",
                "graph_id": "graph-1",
                "source_node_id": "node-1",
                "target_node_id": "node-1",
            }
        ]
    ).to_parquet(tmp_path / "edges.parquet")
    (tmp_path / "run_report.json").write_text(json.dumps({"graph_id": "graph-1"}), encoding="utf-8")

    dataset = load_local_graph(tmp_path)

    assert dataset.graph_id == "graph-1"
    assert "embedding" not in dataset.nodes[0]
    assert dataset.counts["nodes"] == 1


def test_local_run_history_survives_app_restart_and_opens_completed_graph(
    tmp_path: Path,
) -> None:
    """Rebuild a ready run from app-owned metadata without a live process object."""

    state_root = tmp_path / "state"
    output = tmp_path / "graph-output"
    output.mkdir()
    pd.DataFrame([{"id": "node-1", "graph_id": "graph-1"}]).to_parquet(output / "nodes.parquet")
    pd.DataFrame(columns=["id", "source_node_id", "target_node_id"]).to_parquet(
        output / "edges.parquet"
    )
    run_directory = state_root / "runs" / "run-1"
    write_run_record(
        run_directory,
        {
            "run_id": "run-1",
            "graph_id": "graph-1",
            "runtime": "local",
            "status": "running",
            "output_path": str(output),
            "config_path": str(run_directory / "config.yaml"),
            "started_at": "2026-01-01T00:00:00Z",
        },
    )
    backend = LocalBackend(tmp_path, state_root)

    runs = backend.list_runs()
    dataset = backend.load_run_graph(runs[0])

    assert len(runs) == 1
    assert runs[0].status == "succeeded"
    assert dataset.graph_id == "graph-1"


def test_terminal_local_history_does_not_parse_progress_until_opened(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep sidebar history metadata-only while detailed status retains events."""

    state_root = tmp_path / "state"
    run_directory = state_root / "runs" / "run-1"
    write_run_record(
        run_directory,
        {
            "run_id": "run-1",
            "graph_id": "graph-1",
            "runtime": "local",
            "status": "succeeded",
            "documents_total": 12,
            "documents_completed": 12,
        },
    )
    monkeypatch.setattr(
        "flakegraph_app.backends.local.read_jsonl_events",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("sidebar listing must not parse progress logs")
        ),
    )

    runs = LocalBackend(tmp_path, state_root).list_runs()

    assert len(runs) == 1
    assert runs[0].documents_total == 12
    assert runs[0].documents_completed == 12
    assert not runs[0].events


def test_terminal_local_status_wins_when_recorded_pid_was_recycled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    write_run_record(
        state_root / "runs" / "run-1",
        {
            "run_id": "run-1",
            "graph_id": "graph-1",
            "runtime": "local",
            "status": "succeeded",
            "pid": 1234,
        },
    )
    monkeypatch.setattr("flakegraph_app.backends.local._pid_is_running", lambda _pid: True)

    assert LocalBackend(tmp_path, state_root).list_runs()[0].status == "succeeded"


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("", RuntimeMode.LOCAL),
        ("Kubernetes", RuntimeMode.KUBERNETES),
        ("kubernetes", RuntimeMode.KUBERNETES),
        ("Local", RuntimeMode.LOCAL),
        ("Snowflake", RuntimeMode.LOCAL),
        ("nonsense", RuntimeMode.LOCAL),
    ],
)
def test_deployment_chooses_which_runtime_the_app_opens_on(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: RuntimeMode,
) -> None:
    """Let a fleet control plane open on the fleet without changing the choice.

    Snowflake is never selectable this way: it is forced by being deployed inside
    Snowflake, and configuring it elsewhere would default to a runtime that
    cannot start.
    """

    monkeypatch.setenv("FLAKEGRAPH_APP_DEFAULT_RUNTIME", configured)

    assert _configured_default_runtime() is expected


def test_local_run_without_its_artifacts_is_listed_as_unavailable(tmp_path: Path) -> None:
    """Do not offer a graph whose files this host does not have.

    App state can outlive the machine that wrote it, and its recorded paths are
    absolute. Reporting such a run as succeeded sends the explorer to a directory
    that is not there and answers a click with a failure to load.
    """

    state_root = tmp_path / "state"
    run_directory = state_root / "runs" / "run-1"
    write_run_record(
        run_directory,
        {
            "run_id": "run-1",
            "graph_id": "graph-1",
            "runtime": "local",
            "status": "succeeded",
            "storage_kind": "Local files",
            "output_path": "/elsewhere/out/graph-1",
        },
    )

    runs = LocalBackend(tmp_path, state_root).list_runs()

    assert [run.status for run in runs] == ["unavailable"]
    # The entry is a record of work that did happen, so it survives.
    assert (run_directory / "run.json").is_file()
    assert runs[0].output_path == "/elsewhere/out/graph-1"


def test_local_run_artifacts_resolve_against_the_repository_not_the_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep a relative artifact path meaningful wherever the server was started."""

    repository = tmp_path / "repository"
    output = repository / "out" / "app" / "run-1"
    output.mkdir(parents=True)
    pd.DataFrame([{"id": "node-1", "graph_id": "graph-1"}]).to_parquet(output / "nodes.parquet")
    pd.DataFrame(columns=["id", "source_node_id", "target_node_id"]).to_parquet(
        output / "edges.parquet"
    )
    state_root = tmp_path / "state"
    write_run_record(
        state_root / "runs" / "run-1",
        {
            "run_id": "run-1",
            "graph_id": "graph-1",
            "runtime": "local",
            "status": "succeeded",
            "storage_kind": "Local files",
            "output_path": "out/app/run-1",
        },
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    backend = LocalBackend(repository, state_root)

    runs = backend.list_runs()

    assert [run.status for run in runs] == ["succeeded"]
    assert backend.load_run_graph(runs[0]).graph_id == "graph-1"


def test_unreachable_fleet_store_says_the_history_is_only_this_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Never pass a local catalog off as the fleet's own durable history."""

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "app-defaults.yaml").write_text("runtime: {}\n", encoding="utf-8")
    state_root = tmp_path / "state"
    write_run_record(
        state_root / "runs" / "fleet-run",
        {
            "run_id": "fleet-run",
            "graph_id": "graph-1",
            "runtime": "kubernetes",
            "status": "succeeded",
        },
    )
    backend = KubernetesBackend(tmp_path, state_root)
    monkeypatch.setattr(
        KubernetesBackend,
        "_run_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError('error: context "leo-spark" does not exist')
        ),
    )

    runs = backend.list_runs()

    assert [run.run_id for run in runs] == ["fleet-run"]
    assert backend.listing_warning is not None
    assert "not the fleet's" in backend.listing_warning
    assert "leo-spark" in backend.listing_warning


def test_kubernetes_json_commands_allow_long_submit_and_export_operations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_timeouts: list[int] = []

    @contextmanager
    def fleet_environment(
        _namespace: str,
        _target: object,
        environment: object = None,
    ) -> Iterator[object]:
        yield environment or {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, int | float)
        observed_timeouts.append(int(timeout))
        return subprocess.CompletedProcess(command, 0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._fleet_database_environment",
        fleet_environment,
    )
    monkeypatch.setattr("flakegraph_app.backends.kubernetes.subprocess.run", run)
    backend = KubernetesBackend(tmp_path, tmp_path / "state")

    backend._run_json(["flakegraph", "distributed", "status"])  # noqa: SLF001
    backend._run_json(["flakegraph", "distributed", "submit"])  # noqa: SLF001
    backend._run_json(["flakegraph", "distributed", "export"])  # noqa: SLF001

    assert observed_timeouts == [60, 1800, 1800]


def test_runtime_run_histories_do_not_mix_local_and_kubernetes(tmp_path: Path) -> None:
    """Keep the conversation-like sidebar scoped to the currently selected runtime."""

    state_root = tmp_path / "state"
    for run_id, runtime in (("local-run", "local"), ("fleet-run", "kubernetes")):
        write_run_record(
            state_root / "runs" / run_id,
            {
                "run_id": run_id,
                "graph_id": run_id,
                "runtime": runtime,
                "status": "running",
                "output_path": str(tmp_path / run_id),
            },
        )

    local_runs = LocalBackend(tmp_path, state_root).list_runs()
    fleet_runs = KubernetesBackend(tmp_path, state_root).list_runs()

    assert [run.run_id for run in local_runs] == ["local-run"]
    assert [run.run_id for run in fleet_runs] == ["fleet-run"]


def test_review_table_normalizes_mixed_nested_values_for_arrow() -> None:
    """Keep old and new provenance shapes inspectable in one Streamlit table."""

    frame = _review_frame(
        [
            {"name": "Alpha", "sources": ["a", "b"]},
            {"name": "Beta", "sources": "legacy-source"},
        ],
        ["name", "sources"],
    )

    assert frame.to_dict(orient="records") == [
        {"name": "Alpha", "sources": '["a", "b"]'},
        {"name": "Beta", "sources": "legacy-source"},
    ]


def test_snowflake_deployment_manifests_include_every_app_module() -> None:
    repository_root = _APP_ROOT.parent
    definition = yaml.safe_load((repository_root / "snowflake.yml").read_text(encoding="utf-8"))
    entity = definition["entities"]["flakegraph_app"]
    environment = yaml.safe_load((_APP_ROOT / "environment.yml").read_text(encoding="utf-8"))

    assert entity["main_file"] == "app/streamlit_app.py"
    # Snowflake resolves the environment and the main file from the deployed
    # version root. Preserving the repository's app/ prefix leaves user_packages
    # empty and reports the main file missing as '/tmp/appRoot/streamlit_app.py',
    # so every artifact is mapped to that root.
    assert entity["artifacts"] == [
        {"src": "app/streamlit_app.py", "dest": "./"},
        {"src": "app/flakegraph_app/*.py", "dest": "flakegraph_app/"},
        {
            "src": "app/flakegraph_app/backends/*.py",
            "dest": "flakegraph_app/backends/",
        },
        {"src": "app/flakegraph_app/ui/*.py", "dest": "flakegraph_app/ui/"},
        {"src": "app/assets/*", "dest": "assets/"},
        {"src": "app/environment.yml", "dest": "./"},
        {"src": "configs/app-defaults.yaml", "dest": "configs/"},
        {"src": "configs/ontologies/general.yaml", "dest": "configs/ontologies/"},
    ]
    # The interpreter is pinned by Snowflake's default package set and is not a
    # package in its Anaconda channel; listing it fails the whole environment.
    assert not any(
        str(dependency).startswith("python") for dependency in environment["dependencies"]
    )
    assert "streamlit=1.52.2" in environment["dependencies"]


def test_kubernetes_backend_normalizes_queue_and_workload_status(tmp_path: Path) -> None:
    """Keep durable task progress and pod telemetry stable in a run workspace."""

    snapshot = _distributed_snapshot(
        {
            "run": {"id": "run-1", "graph_id": "graph-1", "status": "running"},
            "task_counts": [
                {
                    "stage": "preparation",
                    "status": "succeeded",
                    "count": 2,
                    "started_at": "2026-01-01T00:00:00Z",
                    "completed_at": "2026-01-01T00:00:05Z",
                },
                {"stage": "windows", "status": "running", "count": 1},
                {"stage": "windows", "status": "queued", "count": 3},
            ],
            "diagnostics": {"warnings": [{"message": "One worker is recovering"}]},
            "error": {
                "task_id": "run-1-windows-1",
                "error_type": "RuntimeError",
                "error_message": "provider unavailable",
            },
        },
        tmp_path / "config.yaml",
    )
    workload = _pod_status(
        {
            "metadata": {
                "name": "flakegraph-worker-1",
                "labels": {"app.kubernetes.io/component": "worker"},
            },
            "spec": {
                "nodeName": "gpu-node-1",
                "containers": [
                    {
                        "image": "ghcr.io/flakegraph/worker:1.0.0",
                        "args": ["worker", "distributed"],
                    }
                ],
            },
            "status": {
                "phase": "Running",
                "containerStatuses": [{"ready": True, "restartCount": 1}],
            },
        },
        {"flakegraph-worker-1": {"cpu": "850m", "memory": "12Gi"}},
    )

    assert snapshot.run_id == "run-1"
    assert snapshot.stages[0].elapsed_ms == 5_000
    assert [(stage.stage, stage.status) for stage in snapshot.stages] == [
        ("preparation", "completed"),
        ("windows", "running"),
    ]
    assert snapshot.warnings == ["One worker is recovering"]
    assert snapshot.error == ("Task run-1-windows-1 failed with RuntimeError: provider unavailable")
    assert workload.ready
    assert workload.node == "gpu-node-1"
    assert workload.cpu == "850m"
    assert workload.image == "ghcr.io/flakegraph/worker:1.0.0"


def test_kubernetes_snapshot_summarizes_java_failure_and_retains_details(
    tmp_path: Path,
) -> None:
    """Keep the Kubernetes cause visible without discarding its full traceback."""

    message = (
        "An error occurred while calling JavaSparkContext.\n"
        ": KubernetesClientException: Failure executing POST. "
        'Message: configmaps "spark-exec-conf-map" already exists. '
        "Received status: Status(code=409)\n"
        "at io.fabric8.kubernetes.client.dsl.internal.OperationSupport.waitForResult"
    )
    snapshot = _distributed_snapshot(
        {
            "run": {"id": "run-1", "graph_id": "graph-1", "status": "failed"},
            "task_counts": [],
            "error": {
                "task_id": "finalize-1",
                "error_type": "Py4JJavaError",
                "error_message": message,
            },
        },
        tmp_path / "config.yaml",
    )

    assert snapshot.error == (
        "Task finalize-1 failed with Py4JJavaError: "
        'configmaps "spark-exec-conf-map" already exists.'
    )
    assert snapshot.raw["error"]["error_message"] == message


def test_kubernetes_retry_requeues_only_failed_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Expose the durable retry command and refresh its resulting run status."""

    backend = KubernetesBackend(tmp_path, tmp_path / "state")
    commands: list[list[str]] = []
    owned_config = tmp_path / "state" / "runs" / "run-1" / "config.yaml"
    owned_config.parent.mkdir(parents=True)
    owned_config.write_text("runtime:\n  runtime: kubernetes\n", encoding="utf-8")

    def run_json(command: list[str], environment: object = None) -> dict[str, object]:
        del environment
        commands.append(command)
        return {
            "run": {"id": "run-1", "graph_id": "graph-1", "status": "queued"},
            "task_counts": [],
            "created_at": "2026-07-16T10:00:00Z",
            "updated_at": "2026-07-16T10:01:00Z",
        }

    monkeypatch.setattr(backend, "_run_json", run_json)

    snapshot = backend.retry("run-1", tmp_path / "unrelated-fleet-config.yaml")

    assert "retry" in backend.capabilities
    assert [command[4] for command in commands] == ["retry", "status"]
    assert [Path(command[command.index("--config") + 1]) for command in commands] == [
        owned_config,
        owned_config,
    ]
    assert snapshot.status == "queued"


def test_snowflake_status_uses_one_bounded_aggregate_query() -> None:
    """Avoid two warehouse round trips for every active-run progress refresh."""

    class Row:
        def __init__(self, values: dict[str, object]) -> None:
            self.values = values

        def as_dict(self) -> dict[str, object]:
            return self.values

    class Query:
        def collect(self) -> list[Row]:
            common: dict[str, object] = {
                "ID": "run-1",
                "GRAPH_ID": "graph-1",
                "STATUS": "RUNNING",
                "ERROR": None,
                "CREATED_AT": "2026-01-01T00:00:00Z",
                "UPDATED_AT": "2026-01-01T00:01:00Z",
            }
            return [
                Row(
                    {
                        **common,
                        "STAGE": "extract",
                        "FILE_STATUS": "DONE",
                        "FILE_COUNT": 8,
                    }
                ),
                Row(
                    {
                        **common,
                        "STAGE": "extract",
                        "FILE_STATUS": "CLAIMED",
                        "FILE_COUNT": 2,
                    }
                ),
            ]

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def sql(self, query: str, params: object = None) -> Query:
            self.calls.append((query, params))
            return Query()

    session = Session()

    snapshot = SnowflakeBackend(session).status("run-1")

    assert len(session.calls) == 1
    assert "LEFT JOIN KG_JOB_FILE" in session.calls[0][0]
    assert snapshot.status == "running"
    assert snapshot.documents_total == 10
    assert snapshot.documents_completed == 8
    assert snapshot.stages[0].status == "running"


def test_snowflake_graph_rename_is_a_parameterized_metadata_upsert() -> None:
    """Rename graph metadata without rewriting graph rows or interpolating input."""

    class Query:
        def collect(self) -> list[object]:
            return []

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def sql(self, query: str, params: object = None) -> Query:
            self.calls.append((query, params))
            return Query()

    session = Session()

    result = SnowflakeBackend(session).rename_graph(" graph-1 ", "  Research Graph  ")

    assert result == "Research Graph"
    # Renaming is authorised first, then written; the write itself stays a
    # single parameterised upsert that never touches graph rows.
    upserts = [call for call in session.calls if call[0].startswith("MERGE INTO KG_GRAPH")]
    assert len(upserts) == 1
    assert "KG_NODE" not in upserts[0][0]
    assert upserts[0][1] == ["graph-1", "Research Graph"]


def test_snowflake_app_service_spec_is_complete_and_credential_free(
    tmp_path: Path,
) -> None:
    """Carry app selections into SPCS without overriding its managed identity."""

    request = _snowflake_request(tmp_path)

    launch = build_spcs_launch(build_run_config(request))
    parsed = yaml.safe_load(launch.spec_yaml)
    container = parsed["spec"]["containers"][0]
    inline = json.loads(container["env"]["KG_CONFIG_JSON"])

    assert launch.compute_pool == "FLAKEGRAPH_POOL"
    assert launch.spec_stage == "@DB.GRAPH.SPECS"
    assert launch.service_identifier.startswith("DB.GRAPH.FLAKEGRAPH_APP_")
    assert "ASYNC = TRUE" in launch.execute_sql
    assert container["image"] == "/DB/GRAPH/IMAGES/flakegraph:test@sha256:" + "a" * 64
    assert inline["job"]["job_id"] == "job-1"
    assert inline["job"]["lease_owner"].startswith("flakegraph_app_")
    assert inline["ocr"]["provider"] == "snowflake_cortex"
    assert inline["llm"]["model"] == "mistral-7b"
    assert inline["graph"]["max_chunks_per_llm_call"] == 1
    assert inline["snowflake"]["authenticator"] == "oauth"
    assert inline["snowflake"]["oauth_token_path"] == "/snowflake/session/token"
    assert not ({"account", "host", "user", "password", "oauth_token"} & set(inline["snowflake"]))
    assert "secret-value" not in launch.spec_yaml


def test_snowflake_submit_stages_spec_before_launching_job(
    tmp_path: Path,
) -> None:
    """Make app submission a complete asynchronous SPCS control-plane action.

    The stage is listed through the session rather than through a replaced
    method, so submission exercises the same statement a warehouse would answer.
    """

    class Row:
        def __init__(self, values: dict[str, object]) -> None:
            self.values = values

        def as_dict(self) -> dict[str, object]:
            return dict(self.values)

    class Query:
        def __init__(self, session: Session, sql: str) -> None:
            self.session = session
            self.sql = sql

        def collect(self) -> list[object]:
            self.session.collected.append(self.sql)
            if self.sql.startswith("LIST "):
                return [
                    Row(
                        {
                            "name": "DB/GRAPH/DOCUMENTS/martial-arts.pdf",
                            "size": 42,
                            "md5": "abc",
                            "last_modified": "2026-07-16T10:00:00Z",
                        }
                    )
                ]
            return []

    class Writer:
        def mode(self, value: str) -> Writer:
            assert value == "overwrite"
            return self

        def save_as_table(self, name: str, *, table_type: str) -> None:
            assert name.startswith("KG_JOB_FILE_SUBMISSION_")
            # Stored procedures reject temporary tables; transient works in both.
            assert table_type == "transient"

    class Frame:
        write = Writer()

    class FileApi:
        def __init__(self) -> None:
            self.uploads: list[tuple[bytes, str]] = []

        def put_stream(self, stream: Any, destination: str, **kwargs: object) -> None:
            assert kwargs == {"auto_compress": False, "overwrite": True}
            self.uploads.append((stream.read(), destination))

    class Session:
        def __init__(self) -> None:
            self.collected: list[str] = []
            self.file = FileApi()

        def sql(self, sql: str, params: object = None) -> Query:
            del params
            return Query(self, sql)

        def create_dataframe(self, rows: Any, schema: object) -> Frame:
            assert list(rows)
            assert schema
            return Frame()

    expected = RunSnapshot(
        run_id="job-1",
        graph_id="graph-1",
        status="pending",
        started_at=None,
        updated_at=None,
    )

    class SubmittingBackend(SnowflakeBackend):
        """Report a known snapshot instead of reading back the queue it wrote."""

        def status(self, run_id: str, config_path: Path | None = None) -> RunSnapshot:
            del run_id, config_path
            return expected

    session = Session()
    backend = SubmittingBackend(session)

    result = backend.submit(_snowflake_request(tmp_path))

    assert result == expected
    assert any(sql.startswith("LIST @DB.GRAPH.DOCUMENTS/") for sql in session.collected)
    assert len(session.file.uploads) == 1
    assert session.file.uploads[0][1].startswith("@DB.GRAPH.SPECS/flakegraph-app-")
    assert b"KG_CONFIG_JSON" in session.file.uploads[0][0]
    assert session.collected[-1].startswith("EXECUTE JOB SERVICE")


def test_kubernetes_preflight_uses_coordinator_mode_and_not_local_provider_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep fleet dependencies and Secret values on the worker side of the boundary."""

    request = _request(tmp_path, SourceKind.LOCAL, {"path": str(tmp_path)})
    request = IngestionRequest(**{**request.__dict__, "runtime": RuntimeMode.KUBERNETES})
    observed: dict[str, Any] = {}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"ok": true, "checks": ["coordinator"], "errors": []}\n',
            stderr="",
        )

    monkeypatch.setattr("flakegraph_app.backends.kubernetes.subprocess.run", run)
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._fleet_preflight",
        lambda *_args: {"ok": True, "checks": ["fleet"], "errors": []},
    )

    result = KubernetesBackend(tmp_path, tmp_path / "state").preflight(request)

    assert "--orchestrator" in observed["command"]
    assert observed["environment"]["TEST_LLM_KEY"] == "configured-in-kubernetes"
    assert result == {"ok": True, "checks": ["coordinator", "fleet"], "errors": []}


def test_kubernetes_fleet_preflight_accepts_keda_workers_and_ready_local_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Treat scale-to-zero workers as healthy when KEDA and model serving are ready."""

    request = _fleet_request(tmp_path)
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._kubectl_json",
        _fleet_kubectl_fixture(profile_ocr="fallback", model_servers=1),
    )

    result = _fleet_preflight(request, "flakegraph", ClusterTarget())

    assert isinstance(result["checks"], list)
    assert isinstance(result["errors"], list)
    assert result["ok"] is True
    assert not result["errors"]
    assert "Chart-managed vLLM model servers are ready: 1/1" in result["checks"]


def _ontology_request(tmp_path: Path, profile: Mapping[str, object]) -> IngestionRequest:
    """Build a fleet request whose base profile references an ontology file."""

    # The reference must be relative and resolve beside the profile: absolute
    # paths are refused so a profile cannot name an arbitrary readable file.
    (tmp_path / "ontology.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump({"ontology": {"profile_path": "ontology.yaml"}}), encoding="utf-8"
    )
    request = _fleet_request(tmp_path)
    return IngestionRequest(**{**request.__dict__, "base_config_path": base})


def test_kubernetes_fleet_preflight_rejects_a_run_whose_ontology_no_worker_mounts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unmatched ontology leaves a run queued forever rather than failing it.

    The worker claim digest covers the ontology, so a run carrying one that the
    fleet does not mount is never claimed by anything. Nothing errors and nothing
    appears on the run page, which is precisely why preflight has to catch it.
    """

    request = _ontology_request(tmp_path, {"name": "general", "mode": "hybrid"})
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._kubectl_json",
        _fleet_kubectl_fixture(profile_ocr="fallback", model_servers=1),
    )

    result = _fleet_preflight(request, "flakegraph", ClusterTarget())

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert any("ontology is not deployed" in str(item) for item in errors)


def test_kubernetes_fleet_preflight_rejects_an_ontology_that_differs_from_the_fleet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Matching by name is not enough; changed ontology bytes change the graph."""

    request = _ontology_request(tmp_path, {"name": "general", "mode": "hybrid"})
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._kubectl_json",
        _fleet_kubectl_fixture(
            profile_ocr="fallback",
            model_servers=1,
            ontology={"name": "general", "mode": "strict"},
        ),
    )

    result = _fleet_preflight(request, "flakegraph", ClusterTarget())

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert any("differs from the one fleet workers mount" in str(item) for item in errors)


def test_kubernetes_fleet_preflight_accepts_a_matching_ontology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The run the operator actually wants must still pass."""

    profile: dict[str, object] = {"name": "general", "mode": "hybrid"}
    request = _ontology_request(tmp_path, profile)
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._kubectl_json",
        _fleet_kubectl_fixture(profile_ocr="fallback", model_servers=1, ontology=profile),
    )

    result = _fleet_preflight(request, "flakegraph", ClusterTarget())

    assert result["ok"] is True, result["errors"]


def test_kubernetes_fleet_preflight_rejects_missing_worker_service_account(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a fleet that cannot create autoscaled workers before accepting work."""

    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._kubectl_json",
        _fleet_kubectl_fixture(
            profile_ocr="fallback",
            model_servers=1,
            service_account_available=False,
        ),
    )

    result = _fleet_preflight(_fleet_request(tmp_path), "flakegraph", ClusterTarget())

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert any("references missing service account" in str(error) for error in errors)


def test_kubernetes_recovery_does_not_restart_healthy_shared_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave a healthy finalizer untouched when another graph currently owns it."""

    deployment = {
        "metadata": {
            "name": "flakegraph-finalize",
            "labels": {"app.kubernetes.io/component": "worker-finalize"},
        },
        "spec": {
            "replicas": 1,
            "template": {
                "spec": {
                    "serviceAccountName": "flakegraph-spark",
                    "nodeSelector": {"flakegraph.io/node-class": "nvidia-spark"},
                }
            },
        },
        "status": {"availableReplicas": 1},
    }

    def kubectl(arguments: list[str], *, target: object, raw: bool = False) -> object:
        assert raw is False
        resource = arguments[1]
        if resource == "deployments":
            return {"items": [deployment]}
        if resource == "serviceaccounts":
            return {"items": [{"metadata": {"name": "flakegraph-spark"}}]}
        if resource == "nodes":
            return {
                "items": [
                    {
                        "metadata": {"labels": {"flakegraph.io/node-class": "nvidia-spark"}},
                        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                    }
                ]
            }
        if resource == "pods":
            return {"items": []}
        raise AssertionError(f"Unexpected kubectl arguments: {arguments}")

    monkeypatch.setattr("flakegraph_app.backends.kubernetes._kubectl_json", kubectl)

    message = _recover_kubernetes_workers("flakegraph", {"worker-finalize"}, ClusterTarget())

    assert "worker-finalize" in message
    assert "waiting for shared capacity" in message


def test_kubernetes_recovery_does_not_interrupt_reconciling_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let Kubernetes replace an unavailable active pod without forcing a rollout."""

    deployment = {
        "metadata": {
            "name": "flakegraph-finalize",
            "labels": {"app.kubernetes.io/component": "worker-finalize"},
        },
        "spec": {
            "replicas": 1,
            "template": {
                "spec": {
                    "serviceAccountName": "flakegraph-spark",
                    "nodeSelector": {"flakegraph.io/node-class": "nvidia-spark"},
                }
            },
        },
        "status": {"availableReplicas": 0},
    }
    running_pod = {
        "metadata": {
            "labels": {"app.kubernetes.io/component": "worker-finalize"},
        },
        "status": {"phase": "Running", "containerStatuses": []},
    }
    calls: list[list[str]] = []

    def kubectl(arguments: list[str], *, target: object, raw: bool = False) -> object:
        if raw:
            calls.append(arguments)
            return ""
        resource = arguments[1]
        if resource == "deployments":
            return {"items": [deployment]}
        if resource == "serviceaccounts":
            return {"items": [{"metadata": {"name": "flakegraph-spark"}}]}
        if resource == "nodes":
            return {
                "items": [
                    {
                        "metadata": {"labels": {"flakegraph.io/node-class": "nvidia-spark"}},
                        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                    }
                ]
            }
        if resource == "pods":
            return {"items": [running_pod]}
        raise AssertionError(f"Unexpected kubectl arguments: {arguments}")

    monkeypatch.setattr("flakegraph_app.backends.kubernetes._kubectl_json", kubectl)

    message = _recover_kubernetes_workers("flakegraph", {"worker-finalize"}, ClusterTarget())

    assert "already reconciling" in message
    assert calls == []


def test_kubernetes_fleet_preflight_explains_profile_and_model_serving_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Block runs that the homogeneous worker profile cannot execute consistently."""

    request = _fleet_request(tmp_path)
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._kubectl_json",
        _fleet_kubectl_fixture(profile_ocr="builtin_text", model_servers=0),
    )

    result = _fleet_preflight(request, "flakegraph", ClusterTarget())

    assert isinstance(result["errors"], list)
    assert result["ok"] is False
    assert (
        "OCR provider does not match fleet workers: selected 'fallback', deployed 'builtin_text'"
        in result["errors"]
    )
    assert (
        "Selected vLLM endpoint is node-local, but the fleet has no ready model-serving StatefulSet"
    ) in result["errors"]


def test_kubernetes_fleet_preflight_requires_output_secret_on_finalizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a Snowflake destination that the finalizer cannot authenticate to."""

    request = _fleet_request(tmp_path)
    request = IngestionRequest(
        **{
            **request.__dict__,
            "output": OutputDestination(
                StorageKind.SNOWFLAKE,
                tmp_path / "workspace",
                SnowflakeOutput(
                    account="account",
                    user="user",
                    database="DB",
                    schema="GRAPH",
                    warehouse="WH",
                    bulk_stage="@DB.GRAPH.LOAD_STAGE",
                    credential_environment_variable="KG_SNOWFLAKE_PASSWORD",
                    credential_field="password",
                ),
            ),
        }
    )
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._kubectl_json",
        _fleet_kubectl_fixture(
            profile_ocr="fallback",
            model_servers=1,
            output_credential="KG_SNOWFLAKE_PASSWORD",
        ),
    )

    result = _fleet_preflight(request, "flakegraph", ClusterTarget())

    assert result["ok"] is True
    assert isinstance(result["checks"], list)
    assert "Finalizer receives output credential: KG_SNOWFLAKE_PASSWORD" in result["checks"]


def test_kubernetes_node_inventory_combines_hardware_metrics_and_model_placement() -> None:
    """Give operators one truthful node record from Kubernetes' separate APIs."""

    workloads = [
        _pod_status(
            {
                "metadata": {
                    "name": "extract-1",
                    "labels": {"app.kubernetes.io/component": "worker-extract"},
                },
                "spec": {
                    "nodeName": "spark-1",
                    "containers": [{"image": "flakegraph:1.0.0"}],
                },
                "status": {
                    "phase": "Running",
                    "containerStatuses": [{"ready": True, "restartCount": 0}],
                },
            },
            {},
        ),
        _pod_status(
            {
                "metadata": {
                    "name": "vllm-1",
                    "labels": {"app.kubernetes.io/component": "model-serving"},
                },
                "spec": {
                    "nodeName": "spark-1",
                    "containers": [
                        {
                            "image": "nvcr.io/nvidia/vllm:26.06-py3",
                            "args": [
                                "serve",
                                "unsloth/Qwen3.8-27B-NVFP4",
                                "--served-model-name",
                                "qwen-production",
                            ],
                        }
                    ],
                },
                "status": {
                    "phase": "Running",
                    "containerStatuses": [{"ready": True, "restartCount": 0}],
                },
            },
            {},
        ),
    ]
    node = _node_status(
        {
            "metadata": {
                "name": "spark-1",
                "labels": {
                    "flakegraph.io/node-class": "nvidia-spark",
                    "nvidia.com/gpu.product": "NVIDIA-GB10",
                },
            },
            "status": {
                "capacity": {"cpu": "20", "memory": "128Gi", "nvidia.com/gpu": "1"},
                "conditions": [{"type": "Ready", "status": "True"}],
            },
        },
        workloads,
        {
            "spark-1": {
                "cpu": "1240m",
                "cpu_percent": "6%",
                "memory": "42Gi",
                "memory_percent": "33%",
            }
        },
    )

    assert node.ready
    assert node.node_class == "nvidia-spark"
    assert node.gpu_count == 1
    assert node.gpu_model == "NVIDIA-GB10"
    assert node.worker_count == 1
    assert node.workload_count == 2
    assert node.model_server_ready
    assert node.model == "qwen-production"
    assert node.model_image == "nvcr.io/nvidia/vllm:26.06-py3"
    assert node.cpu_percent == "6%"


def test_kubernetes_recent_run_overview_uses_durable_task_counts(tmp_path: Path) -> None:
    """Translate the PostgreSQL-backed list contract without requiring task payloads."""

    snapshot = _distributed_overview_snapshot(
        {
            "id": "run-1",
            "graph_id": "graph-1",
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:05:00Z",
            "task_counts": [
                {"stage": "finalize_graph", "status": "queued", "count": 1},
                {"stage": "prepare_document", "status": "succeeded", "count": 8},
                {"stage": "prepare_document", "status": "running", "count": 2},
                {"stage": "compact_document", "status": "succeeded", "count": 7},
                {"stage": "compact_document", "status": "failed", "count": 1},
            ],
        },
        tmp_path / "fleet.yaml",
        None,
        tmp_path / "out" / "run-1",
    )

    assert snapshot.run_id == "run-1"
    assert snapshot.status == "running"
    assert snapshot.output_path == str(tmp_path / "out" / "run-1")
    assert [stage.stage for stage in snapshot.stages] == [
        "prepare_document",
        "compact_document",
        "finalize_graph",
    ]
    assert snapshot.stages[0].total == 10
    assert snapshot.stages[0].completed == 8
    assert snapshot.documents_total == 10
    assert snapshot.documents_completed == 7
    assert snapshot.documents_failed == 1


def test_kubernetes_finalization_uses_durable_inner_task_progress(tmp_path: Path) -> None:
    """Show real graph-finalization phases instead of one opaque queue task."""

    snapshot = _distributed_overview_snapshot(
        {
            "id": "run-1",
            "graph_id": "graph-1",
            "status": "running",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:05:00Z",
            "task_counts": [
                {
                    "stage": "finalize_graph",
                    "status": "running",
                    "count": 1,
                    "started_at": "2026-01-01T00:04:00Z",
                    "progress": {
                        "phase": "write_tables",
                        "phase_index": 10,
                        "phase_total": 12,
                        "completed": 7,
                        "total": 12,
                        "message": "Writing partitioned graph tables",
                    },
                }
            ],
        },
        tmp_path / "fleet.yaml",
        None,
        tmp_path / "out" / "run-1",
    )

    finalization = snapshot.stages[0]
    assert finalization.stage == "finalize_graph"
    assert finalization.status == "running"
    assert finalization.completed == 9
    assert finalization.total == 12
    assert finalization.message == "Writing partitioned graph tables · 7/12"


def test_run_history_tombstones_are_atomic_and_idempotent(tmp_path: Path) -> None:
    """Keep history removal durable without mutating graph or coordinator storage."""

    hide_run(tmp_path, "run-2")
    hide_run(tmp_path, "run-1")
    hide_run(tmp_path, "run-2")

    assert hidden_run_ids(tmp_path) == frozenset({"run-1", "run-2"})
    assert json.loads((tmp_path / "hidden-runs.json").read_text(encoding="utf-8")) == [
        "run-1",
        "run-2",
    ]


def test_local_history_removal_preserves_graph_artifacts(tmp_path: Path) -> None:
    """Remove app metadata only; an explicit history action must not erase graph data."""

    state_root = tmp_path / "state"
    output = tmp_path / "output"
    output.mkdir()
    artifact = output / "nodes.parquet"
    artifact.write_bytes(b"graph-data")
    run_directory = state_root / "runs" / "run-1"
    write_run_record(
        run_directory,
        {
            "run_id": "run-1",
            "graph_id": "graph-1",
            "runtime": "local",
            "status": "succeeded",
            "output_path": str(output),
        },
    )
    backend = LocalBackend(tmp_path, state_root)

    backend.forget("run-1")

    assert not run_directory.exists()
    assert artifact.read_bytes() == b"graph-data"


def test_sidebar_history_page_bounds_widgets_and_preserves_selection() -> None:
    """Keep long histories fast without making an older selected graph disappear."""

    runs = [
        RunSnapshot(
            run_id=f"run-{index}",
            graph_id=f"graph-{index}",
            status="succeeded",
            started_at=None,
            updated_at=None,
            storage_kind=StorageKind.LOCAL,
        )
        for index in range(30)
    ]

    visible = _history_page(runs, 15, "run-24")

    assert [run.run_id for run in visible[:15]] == [f"run-{index}" for index in range(15)]
    assert visible[-1].run_id == "run-24"
    assert len(visible) == 16


def test_bulk_history_selection_excludes_active_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never allow a stale checkbox value to remove work that is still active."""

    runs = [
        RunSnapshot(
            run_id="finished",
            graph_id="finished-graph",
            status="succeeded",
            started_at=None,
            updated_at=None,
            storage_kind=StorageKind.LOCAL,
        ),
        RunSnapshot(
            run_id="active",
            graph_id="active-graph",
            status="running",
            started_at=None,
            updated_at=None,
            storage_kind=StorageKind.LOCAL,
        ),
    ]
    monkeypatch.setattr(
        "flakegraph_app.ui.navigation.st.session_state",
        {
            "bulk_graph_selected_local_finished": True,
            "bulk_graph_selected_local_active": True,
        },
    )

    assert _selected_bulk_ids("local", runs) == ["finished"]


def test_bulk_removal_continues_after_one_backend_failure() -> None:
    """Remove independent entries while retaining exact failures for a retry.

    A runtime whose graphs are files this application does not own removes the
    run from history instead of deleting artifacts it did not write.
    """

    class Backend:
        capabilities = frozenset({"forget"})

        def __init__(self) -> None:
            self.calls: list[str] = []

        def forget(self, run_id: str, config_path: Path | None = None) -> None:
            del config_path
            self.calls.append(run_id)
            if run_id == "run-2":
                raise RuntimeError("temporary backend failure")

    runs = [
        RunSnapshot(
            run_id=f"run-{index}",
            graph_id=f"graph-{index}",
            status="succeeded",
            started_at=None,
            updated_at=None,
            storage_kind=StorageKind.LOCAL,
        )
        for index in range(1, 4)
    ]
    backend: Any = Backend()

    failures = _forget_runs(backend, runs)

    assert backend.calls == ["run-1", "run-2", "run-3"]
    assert failures == {"run-2": "temporary backend failure"}


def test_bulk_removal_deletes_the_graph_where_the_runtime_stores_it() -> None:
    """Delete rather than hide, so a removed graph leaves nothing behind.

    Removing a graph from the sidebar is the same act whether one or ten are
    selected; a bulk path that only hid them would strand exactly the data the
    single-graph path now clears.
    """

    class Backend:
        capabilities = frozenset({"forget", "delete_graph"})

        def __init__(self) -> None:
            self.deleted: list[str] = []

        def delete_graph(self, graph_id: str) -> dict[str, int]:
            self.deleted.append(graph_id)
            return {"KG_NODE": 5}

        def forget(self, run_id: str, config_path: Path | None = None) -> None:
            raise AssertionError("a stored graph must be deleted, not hidden")

    runs = [
        RunSnapshot(
            run_id=f"run-{index}",
            graph_id=f"graph-{index}",
            status="succeeded",
            started_at=None,
            updated_at=None,
            storage_kind=StorageKind.SNOWFLAKE,
        )
        for index in range(1, 3)
    ]
    backend: Any = Backend()

    assert _forget_runs(backend, runs) == {}
    assert backend.deleted == ["graph-1", "graph-2"]


def test_completed_runs_open_without_a_redundant_status_round_trip() -> None:
    """Use authoritative terminal list rows while refreshing all changeable states."""

    assert _requires_status_refresh("running") is True
    assert _requires_status_refresh("unknown") is True
    assert _requires_status_refresh("succeeded") is False
    assert _requires_status_refresh("failed") is True
    assert _requires_status_refresh("cancelled") is True


def test_run_recovery_control_requires_attention_diagnostics() -> None:
    """Offer recovery only for active work diagnosed as not advancing."""

    stalled = RunSnapshot(
        run_id="run-1",
        graph_id="graph-1",
        status="running",
        started_at=None,
        updated_at=None,
        raw={"diagnostics": {"state": "attention_required"}},
    )
    healthy = RunSnapshot(
        run_id="run-2",
        graph_id="graph-2",
        status="running",
        started_at=None,
        updated_at=None,
        raw={"diagnostics": {"state": "processing"}},
    )

    assert _needs_recovery(stalled) is True
    assert _needs_recovery(healthy) is False


def test_run_status_error_keeps_only_the_actionable_line() -> None:
    """Keep transient Kubernetes failures useful without rendering a traceback."""

    error = RuntimeError(
        'E0716 memcache.go: "Unhandled Error"\n'
        "Error from server (ServiceUnavailable): control plane unavailable"
    )

    assert concise_error(error) == (
        "Error from server (ServiceUnavailable): control plane unavailable"
    )


def test_failed_run_workspace_surfaces_error_and_retries_failed_work() -> None:
    """Render a durable failure with its diagnosis and an explicit retry action."""

    app = AppTest.from_string(
        """
from pathlib import Path
import streamlit as st
from flakegraph_app.models import RunSnapshot
from flakegraph_app.ui.run_workspace import _render_terminal_run

class Backend:
    capabilities = frozenset({"retry"})

    def retry(self, run_id, config_path=None):
        st.session_state["retried_run"] = run_id
        return snapshot

snapshot = RunSnapshot(
    run_id="run-1",
    graph_id="graph-1",
    status="failed",
    started_at=None,
    updated_at="2026-07-16T10:00:00Z",
    error="Task task-7 failed with RuntimeError: provider unavailable",
)
_render_terminal_run(Backend(), snapshot, Path("/tmp/config.yaml"))
"""
    ).run()

    assert [item.value for item in app.error] == [
        "Task task-7 failed with RuntimeError: provider unavailable"
    ]
    assert [item.label for item in app.button] == ["Retry failed work"]

    app.button[0].click().run()

    assert app.session_state["retried_run"] == "run-1"


def test_failed_run_workspace_hides_traceback_in_technical_details() -> None:
    """Keep complete distributed diagnostics available without a red error wall."""

    app = AppTest.from_string(
        """
from flakegraph_app.models import RunSnapshot
from flakegraph_app.ui.run_workspace import _render_terminal_run

class Backend:
    capabilities = frozenset()

snapshot = RunSnapshot(
    run_id="run-1",
    graph_id="graph-1",
    status="failed",
    started_at=None,
    updated_at="2026-07-16T10:00:00Z",
    error="Task task-7 failed with RuntimeError: provider unavailable",
    raw={"error": {
        "task_id": "task-7",
        "error_type": "RuntimeError",
        "error_message": "provider unavailable\\ntraceback line",
    }},
)
_render_terminal_run(Backend(), snapshot, None)
"""
    ).run()

    assert [item.value for item in app.error] == [
        "Task task-7 failed with RuntimeError: provider unavailable"
    ]
    assert [item.label for item in app.expander] == ["Technical details"]
    assert "traceback line" in app.code[0].value


def test_unchanged_uploads_are_written_only_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Avoid rewriting large upload bodies after unrelated form interactions."""

    class Upload:
        name = "document.pdf"
        size = 7
        file_id = "upload-1"

        def __init__(self) -> None:
            self.reads = 0

        def getvalue(self) -> bytes:
            self.reads += 1
            return b"content"

    upload = Upload()
    session: dict[str, object] = {}
    monkeypatch.setattr("flakegraph_app.ui.ingestion.st.session_state", session)

    first = _persist_uploads([upload], tmp_path / "uploads")
    second = _persist_uploads([upload], tmp_path / "uploads")

    assert first == second
    assert first[0].read_bytes() == b"content"
    assert upload.reads == 1


def test_gpu_utilization_is_optional_and_aggregated_per_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read accelerator load from model pods without failing CPU-only fleet views."""

    workloads = [
        WorkloadStatus(
            name="model-a",
            component="model-serving",
            phase="Running",
            node="node-a",
            ready=True,
            restarts=0,
        ),
        WorkloadStatus(
            name="worker-a",
            component="worker-extract",
            phase="Running",
            node="node-a",
            ready=True,
            restarts=0,
        ),
    ]
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._kubectl_json",
        lambda arguments, *, target, raw=False: "91\n93\n",
    )

    utilization, warning = _gpu_utilization_by_node("flakegraph", workloads, ClusterTarget())

    assert utilization == {"node-a": "92%"}
    assert warning is None


def test_node_assignments_query_only_running_tasks_on_selected_node(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Load bounded task identities lazily without fetching payloads or artifacts."""

    class _Cursor:
        def fetchall(self) -> list[tuple[object, ...]]:
            return [
                (
                    "worker-a",
                    "run-1",
                    "graph-1",
                    "task-1",
                    "extract_relation_window",
                    "window-9",
                    "2026-07-15T18:00:00Z",
                )
            ]

    class _Connection:
        def __enter__(self) -> _Connection:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, query: str, parameters: object) -> _Cursor:
            assert "task.payload_json" not in query
            assert parameters == (["worker-a"],)
            return _Cursor()

    monkeypatch.setenv("KG_DISTRIBUTED_DATABASE_URL", "postgresql://127.0.0.1/flakegraph")
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._kubectl_json",
        lambda arguments, *, target, raw=False: {
            "items": [
                {
                    "metadata": {
                        "name": "worker-a",
                        "labels": {"app.kubernetes.io/component": "worker-extract"},
                    },
                    "status": {"phase": "Running"},
                },
                {
                    "metadata": {
                        "name": "model-a",
                        "labels": {"app.kubernetes.io/component": "model-serving"},
                    },
                    "status": {"phase": "Running"},
                },
            ]
        },
    )
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes.connect",
        lambda *args, **kwargs: _Connection(),
    )

    assignments = KubernetesBackend(tmp_path).node_assignments("flakegraph", "node-a")

    assert len(assignments) == 1
    assert assignments[0].run_id == "run-1"
    assert assignments[0].graph_id == "graph-1"
    assert assignments[0].stage == "extract_relation_window"


def test_kubernetes_completed_run_reuses_exported_graph(tmp_path: Path) -> None:
    """Avoid a distributed export command when standard graph artifacts already exist."""

    output = tmp_path / "fleet-output"
    output.mkdir()
    pd.DataFrame([{"id": "node-1", "graph_id": "graph-1"}]).to_parquet(output / "nodes.parquet")
    pd.DataFrame(columns=["id", "source_node_id", "target_node_id"]).to_parquet(
        output / "edges.parquet"
    )
    backend = KubernetesBackend(tmp_path, tmp_path / "state")
    snapshot = RunSnapshot(
        run_id="run-1",
        graph_id="graph-1",
        status="succeeded",
        started_at=None,
        updated_at=None,
        output_path=str(output),
    )

    dataset = backend.load_run_graph(snapshot, tmp_path / "config.yaml")

    assert dataset.graph_id == "graph-1"


def test_kubernetes_artifact_export_resolves_finalizer_storage_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve only the shared-artifact values required for a local graph export."""

    def kubectl(
        arguments: list[str],
        *,
        target: object,
        raw: bool = False,
    ) -> dict[str, Any]:
        del raw
        if arguments[:2] == ["get", "deployments"]:
            return {
                "items": [
                    {
                        "metadata": {"labels": {"app.kubernetes.io/component": "worker-finalize"}},
                        "spec": {
                            "template": {
                                "spec": {
                                    "containers": [
                                        {
                                            "env": [
                                                {
                                                    "name": "KG_DISTRIBUTED_ARTIFACT_URI",
                                                    "value": "s3://graphs",
                                                },
                                                {
                                                    "name": (
                                                        "KG_DISTRIBUTED_ARTIFACT_ENDPOINT_URL"
                                                    ),
                                                    "value": "http://objects.storage.svc:8333",
                                                },
                                                {
                                                    "name": (
                                                        "KG_DISTRIBUTED_ARTIFACT_ACCESS_KEY_ID"
                                                    ),
                                                    "valueFrom": {
                                                        "secretKeyRef": {
                                                            "name": "artifacts",
                                                            "key": "access",
                                                        }
                                                    },
                                                },
                                            ]
                                        }
                                    ]
                                }
                            }
                        },
                    }
                ]
            }
        assert arguments[:3] == ["get", "secret", "artifacts"]
        return {"data": {"access": "dXNlcg=="}}

    monkeypatch.setattr("flakegraph_app.backends.kubernetes._kubectl_json", kubectl)

    environment = _fleet_artifact_environment("flakegraph", ClusterTarget())

    assert environment == {
        "KG_DISTRIBUTED_ARTIFACT_URI": "s3://graphs",
        "KG_DISTRIBUTED_ARTIFACT_ENDPOINT_URL": "http://objects.storage.svc:8333",
        "KG_DISTRIBUTED_ARTIFACT_ACCESS_KEY_ID": "user",
    }
    assert _cluster_service(environment["KG_DISTRIBUTED_ARTIFACT_ENDPOINT_URL"]) == (
        "objects",
        "storage",
        8333,
    )
    assert _cluster_service("https://objects.example.com") is None


def test_kubernetes_database_url_comes_from_deployed_worker_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discover the fleet database Secret mapping without a local shell export."""

    def kubectl(
        arguments: list[str],
        *,
        target: object,
        raw: bool = False,
    ) -> dict[str, Any]:
        del raw
        if arguments[:2] == ["get", "deployments"]:
            return {
                "items": [
                    {
                        "metadata": {"labels": {"app.kubernetes.io/component": "worker-prepare"}},
                        "spec": {
                            "template": {
                                "spec": {
                                    "containers": [
                                        {
                                            "env": [
                                                {
                                                    "name": "KG_DISTRIBUTED_DATABASE_URL",
                                                    "valueFrom": {
                                                        "secretKeyRef": {
                                                            "name": "database",
                                                            "key": "uri",
                                                        }
                                                    },
                                                }
                                            ]
                                        }
                                    ]
                                }
                            }
                        },
                    }
                ]
            }
        assert arguments[:3] == ["get", "secret", "database"]
        return {"data": {"uri": "cG9zdGdyZXNxbDovL2FwcEBkYXRhYmFzZS1ydzo1NDMyL2tn"}}

    monkeypatch.setattr("flakegraph_app.backends.kubernetes._kubectl_json", kubectl)

    assert (
        _fleet_database_url("flakegraph", ClusterTarget())
        == "postgresql://app@database-rw:5432/kg"
    )


def test_kubernetes_database_environment_forwards_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bridge internal Service DNS for one command and always reap the forward."""

    commands: list[list[str]] = []
    events: list[str | tuple[str, int]] = []
    process_options: list[dict[str, object]] = []

    class Forward:
        stdout = None

        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, timeout: int) -> int:
            events.append(("wait", timeout))
            return 0

        def kill(self) -> None:
            events.append("kill")

    def open_forward(command: list[str], **kwargs: object) -> Forward:
        """Capture the exact kubectl invocation and return a controllable process."""

        commands.append(command)
        process_options.append(kwargs)
        return Forward()

    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._fleet_database_url",
        lambda _namespace, _target: "postgresql://app:encoded%40value@database-rw:5432/kg",
    )
    monkeypatch.setattr("flakegraph_app.backends.kubernetes._available_local_port", lambda: 15432)
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._service_port_forward_target",
        lambda _service, _namespace, _target: "pod/database-primary-1",
    )
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes.subprocess.Popen",
        open_forward,
    )
    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._wait_for_port_forward",
        lambda _process, port: events.append(("ready", port)),
    )

    with _fleet_database_environment(
        "flakegraph", ClusterTarget(), {"PATH": "/bin"}
    ) as environment:
        assert environment["KG_DISTRIBUTED_DATABASE_URL"] == (
            "postgresql://app:encoded%40value@127.0.0.1:15432/kg"
        )

    assert commands[0][-2:] == ["pod/database-primary-1", "15432:5432"]
    # kubectl logs one line per forwarded connection on stdout, which would fill a
    # pipe nobody drains; stderr is captured so a startup failure can be reported.
    assert process_options[0]["stdout"] == subprocess.DEVNULL
    assert process_options[0]["stderr"] == subprocess.PIPE
    assert process_options[0]["env"] is not None
    assert events == [("ready", 15432), "terminate", ("wait", 5)]


def test_kubernetes_port_forward_uses_ready_service_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid selector-matched database replicas and unavailable nodes."""

    monkeypatch.setattr(
        "flakegraph_app.backends.kubernetes._kubectl_json",
        lambda arguments, *, target, raw=False: {
            "items": [
                {
                    "endpoints": [
                        {
                            "conditions": {"ready": False},
                            "targetRef": {"kind": "Pod", "name": "database-replica"},
                        },
                        {
                            "conditions": {"ready": True},
                            "targetRef": {"kind": "Pod", "name": "database-primary"},
                        },
                    ]
                }
            ]
        },
    )

    assert _service_port_forward_target("database-rw", "flakegraph", ClusterTarget()) == (
        "pod/database-primary"
    )


def test_kubernetes_database_service_and_cli_error_normalization() -> None:
    """Recognize only in-cluster database hosts and keep tracebacks out of the UI."""

    assert _database_cluster_service("postgresql://app@database-rw:5432/kg", "flakegraph") == (
        "database-rw",
        "flakegraph",
        5432,
    )
    assert _database_cluster_service("postgresql://app@database-rw.flakegraph/kg", "other") == (
        "database-rw",
        "flakegraph",
        5432,
    )
    assert _database_cluster_service(
        "postgresql://app@database-rw.flakegraph.svc.cluster.local/kg", "other"
    ) == ("database-rw", "flakegraph", 5432)
    assert (
        _database_cluster_service("postgresql://app@database.example.com/kg", "flakegraph") is None
    )
    assert _database_cluster_service("postgresql://localhost:99999/kg", "flakegraph") is None
    with pytest.raises(RuntimeError, match="invalid port"):
        _database_cluster_service("postgresql://app@database-rw:99999/kg", "flakegraph")
    with pytest.raises(RuntimeError, match="artifact endpoint.*invalid port"):
        _cluster_service("https://store.flakegraph.svc.cluster.local:notaport/bucket")
    assert _replace_url_host("postgresql://app:p%40ss@database-rw/kg", 15432) == (
        "postgresql://app:p%40ss@127.0.0.1:15432/kg"
    )
    assert (
        _command_failure_message(
            "Traceback (most recent call last):\nValueError: database is unavailable\n", ""
        )
        == "database is unavailable"
    )


def test_graph_canvas_falls_back_when_scipy_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an incompletely installed app usable while the app extra supplies SciPy."""

    def missing_scipy(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ModuleNotFoundError("No module named 'scipy'", name="scipy")

    monkeypatch.setattr("flakegraph_app.explorer.nx.spring_layout", missing_scipy)

    figure = build_graph_figure(
        [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
        [{"id": "ab", "source_node_id": "a", "target_node_id": "b"}],
    )

    assert len(figure.data) == 3


def test_snowflake_stage_location_rejects_sql_and_path_traversal() -> None:
    """Protect the only Snowflake identifiers interpolated into LIST and PUT SQL."""

    assert _stage_location("@DB.SCHEMA.KG_INPUT_STAGE", "incoming/2026") == (
        "@DB.SCHEMA.KG_INPUT_STAGE/incoming/2026"
    )
    assert _stage_location("DB.SCHEMA.KG_INPUT_STAGE/inbound", "run+1:50%") == (
        "@DB.SCHEMA.KG_INPUT_STAGE/inbound/run+1:50%"
    )
    with pytest.raises(ValueError, match="identifier"):
        _stage_location("KG_INPUT_STAGE; DROP TABLE KG_NODE")
    with pytest.raises(ValueError, match="unsafe path"):
        _stage_location("KG_INPUT_STAGE", "../private")
    with pytest.raises(ValueError, match="unsafe path"):
        _stage_location("KG_INPUT_STAGE", "batch 5")


def _request(
    tmp_path: Path,
    source_kind: SourceKind,
    source: dict[str, str],
) -> IngestionRequest:
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
            "sentence_transformers", model="sentence-transformers/all-MiniLM-L6-v2"
        ),
        output=OutputDestination(StorageKind.LOCAL, tmp_path / "out"),
    )


def _snowflake_request(tmp_path: Path) -> IngestionRequest:
    """Build a complete Snowflake app request with only non-secret runtime data."""

    profile = tmp_path / "snowflake-app-profile.yaml"
    profile.write_text("graph:\n  max_chunks_per_llm_call: 1\n", encoding="utf-8")
    return IngestionRequest(
        runtime=RuntimeMode.SNOWFLAKE,
        job_id="job-1",
        graph_id="graph-1",
        source_kind=SourceKind.SNOWFLAKE_STAGE,
        source={"stage": "@DB.GRAPH.DOCUMENTS", "prefix": "martial_arts"},
        ocr=ProviderSelection("snowflake_cortex"),
        llm=ProviderSelection("snowflake_cortex", model="mistral-7b"),
        embedding=ProviderSelection(
            "snowflake_cortex",
            model="snowflake-arctic-embed-m-v1.5",
        ),
        output=OutputDestination(
            StorageKind.SNOWFLAKE,
            tmp_path / "out",
            SnowflakeOutput(
                account="external-account",
                host="external-account.snowflakecomputing.com",
                user="external-user",
                database="DB",
                schema="GRAPH",
                warehouse="COMPUTE_WH",
                bulk_stage="@DB.GRAPH.BULK",
            ),
        ),
        runtime_options={
            "compute_pool": "FLAKEGRAPH_POOL",
            "image_repository": "DB.GRAPH.IMAGES",
            "image_name": "flakegraph:test",
            "image_digest": "sha256:" + "a" * 64,
            "service_spec_stage": "@DB.GRAPH.SPECS",
        },
        cache_provider="snowflake",
        base_config_path=profile,
    )


def test_snowflake_run_config_omits_blank_optional_runtime_values(tmp_path: Path) -> None:
    request = _snowflake_request(tmp_path)
    request = IngestionRequest(
        **{
            **request.__dict__,
            "runtime_options": {**request.runtime_options, "image_digest": "  "},
        }
    )

    config = build_run_config(request)

    assert "image_digest" not in config["snowflake"]


def _fleet_request(tmp_path: Path) -> IngestionRequest:
    """Build a representative Kubernetes request for fleet-contract tests."""

    request = _request(tmp_path, SourceKind.LOCAL, {"path": str(tmp_path)})
    return IngestionRequest(
        **{
            **request.__dict__,
            "runtime": RuntimeMode.KUBERNETES,
            "ocr": ProviderSelection("fallback"),
            "llm": ProviderSelection(
                "vllm_local",
                model="unsloth/Qwen3.8-27B-NVFP4",
                endpoint="http://localhost:8000/v1",
            ),
            "embedding": ProviderSelection(
                "sentence_transformers",
                model="sentence-transformers/all-MiniLM-L6-v2",
            ),
        }
    )


def _fleet_kubectl_fixture(
    *,
    profile_ocr: str,
    model_servers: int,
    output_credential: str | None = None,
    service_account_available: bool = True,
    ontology: Mapping[str, object] | None = None,
) -> object:
    """Return a deterministic kubectl adapter for fleet-preflight unit tests."""

    deployments = []
    scaled_objects = []
    for component in ("worker-prepare", "worker-extract", "worker-finalize"):
        name = f"flakegraph-{component}"
        deployments.append(
            {
                "metadata": {
                    "name": name,
                    "labels": {"app.kubernetes.io/component": component},
                },
                "spec": {
                    "replicas": 0,
                    "template": {
                        "spec": {
                            "serviceAccountName": "flakegraph",
                            "containers": [
                                {
                                    "image": "flakegraph:1.0.0",
                                    "env": (
                                        [{"name": output_credential}]
                                        if component == "worker-finalize" and output_credential
                                        else []
                                    ),
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "config",
                                    "configMap": {"name": "flakegraph-runtime"},
                                },
                                *(
                                    [
                                        {
                                            "name": "ontology",
                                            "configMap": {"name": "flakegraph-ontology"},
                                        }
                                    ]
                                    if ontology
                                    else []
                                ),
                            ],
                        }
                    },
                },
            }
        )
        scaled_objects.append(
            {
                "spec": {"scaleTargetRef": {"name": name}},
                "status": {"conditions": [{"type": "Ready", "status": "True"}]},
            }
        )

    def kubectl(arguments: list[str], *, target: object, raw: bool = False) -> object:
        if arguments == ["config", "current-context"] and raw:
            return "flakegraph-fleet\n"
        if arguments[:2] == ["get", "nodes"]:
            return {
                "items": [
                    {
                        "metadata": {"labels": {"kubernetes.io/arch": "arm64"}},
                        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                    }
                ]
            }
        resource = arguments[1]
        if resource == "deployments":
            response: object = {"items": deployments}
        elif resource == "scaledobjects.keda.sh":
            response = {"items": scaled_objects}
        elif resource == "configmap" and arguments[2] == "flakegraph-ontology":
            response = {"data": {"ontology.yaml": yaml.safe_dump(ontology or {})}}
        elif resource == "configmap":
            response = {
                "data": {
                    "config.yaml": yaml.safe_dump(
                        {
                            "ocr": {"provider": profile_ocr},
                            "llm": {
                                "provider": "vllm_local",
                                "model": "unsloth/Qwen3.8-27B-NVFP4",
                            },
                            "embedding": {
                                "provider": "sentence_transformers",
                                "model": "sentence-transformers/all-MiniLM-L6-v2",
                            },
                        }
                    )
                }
            }
        elif resource == "statefulsets":
            response = {
                "items": (
                    [
                        {
                            "metadata": {
                                "labels": {"app.kubernetes.io/component": "model-serving"}
                            },
                            "spec": {"replicas": model_servers},
                            "status": {"readyReplicas": model_servers},
                        }
                    ]
                    if model_servers
                    else []
                )
            }
        elif resource == "secrets":
            response = {
                "items": ([{"data": {output_credential: "encoded"}}] if output_credential else [])
            }
        elif resource == "serviceaccounts":
            response = {
                "items": (
                    [{"metadata": {"name": "flakegraph"}}] if service_account_available else []
                )
            }
        else:
            raise AssertionError(f"Unexpected kubectl arguments: {arguments}")
        return response

    return kubectl


def test_stage_upload_streams_bytes_rather_than_a_local_path() -> None:
    """Snowpark's put cannot read the app filesystem inside Snowflake.

    It fails there with a bare SystemError from open_stream, which reads as a
    Python bug rather than an unsupported operation. put_stream uploads the
    content directly and works in both places.
    """

    class _Result:
        status = "UPLOADED"
        target = "a.txt"

    class _FileOps:
        def __init__(self) -> None:
            self.streamed: list[str] = []

        def put(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("put cannot read the app filesystem in Snowflake")

        def put_stream(self, stream: object, location: str, **_kwargs: object) -> _Result:
            self.streamed.append(location)
            return _Result()

    class _StageRow:
        def as_dict(self) -> dict[str, str]:
            return {"name": "KG_INPUT_STAGE"}

    class _Sql:
        def collect(self) -> list[_StageRow]:
            return [_StageRow()]

    class _Session:
        def __init__(self) -> None:
            self.file = _FileOps()

        def sql(self, _statement: str) -> _Sql:
            return _Sql()

    session = _Session()
    backend = SnowflakeBackend(session)

    with tempfile.TemporaryDirectory() as directory:
        document = Path(directory) / "a.txt"
        document.write_text("content", encoding="utf-8")
        uploaded = backend.upload_files([document], "KG_INPUT_STAGE", "app/shared/run-1")

    assert uploaded == 1
    assert session.file.streamed == ["@KG_INPUT_STAGE/app/shared/run-1/a.txt"]


def test_uploading_to_a_missing_stage_names_the_problem() -> None:
    """Snowflake reports it as a SystemError from its storage client.

    That traceback names neither the stage nor the cause, so the check happens
    here, before the first byte moves.
    """

    class _Row:
        def __init__(self, name: str) -> None:
            self._name = name

        def as_dict(self) -> dict[str, str]:
            return {"name": self._name}

    class _Sql:
        def collect(self) -> list[_Row]:
            return [_Row("KG_DOCS"), _Row("KG_LOAD_STAGE")]

    class _Session:
        def sql(self, _statement: str) -> _Sql:
            return _Sql()

    backend = SnowflakeBackend(_Session())

    with tempfile.TemporaryDirectory() as directory:
        document = Path(directory) / "a.txt"
        document.write_text("content", encoding="utf-8")
        with pytest.raises(RuntimeError) as error:
            backend.upload_files([document], "KG_INPUT_STAGE", "app/shared/run-1")

    assert "KG_INPUT_STAGE does not exist" in str(error.value)
    assert "KG_DOCS" in str(error.value)


def test_a_single_put_result_is_not_split_into_its_fields() -> None:
    """PutResult is a NamedTuple, so a tuple check destructures one upload.

    Each field then reads as a separate result of unknown status, and a
    successful upload is reported as eight failures.
    """

    class PutResult(NamedTuple):
        source: str
        target: str
        source_size: int
        target_size: int
        source_compression: str
        target_compression: str
        status: str
        message: str

    result = PutResult("a.md", "a.md", 10, 10, "NONE", "NONE", "UPLOADED", "")

    assert _as_results(result) == [result]
    assert _as_results([result]) == [result]


def test_the_session_supplies_identity_instead_of_asking_for_it() -> None:
    """Deployed in Snowflake, account and user cannot be wrong — so do not ask.

    Retyping values the session already holds only creates opportunities for a
    typo in a field that has exactly one correct answer.
    """

    class _Row(tuple[Any, ...]):  # noqa: SLOT001 - Snowpark rows are tuple-like
        pass

    class _Sql:
        def collect(self) -> list[_Row]:
            return [
                _Row(
                    ("ACME", "ANALYST", "NUC_DB_DEV", "SBX", "WH", "ROLE"),
                )
            ]

    class _Session:
        def sql(self, _statement: str) -> _Sql:
            return _Sql()

    context = SnowflakeBackend(_Session()).current_context()

    assert context["account"] == "ACME"
    assert context["user"] == "ANALYST"
    assert context["database"] == "NUC_DB_DEV"
    assert context["role"] == "ROLE"


def test_discovery_failure_leaves_the_operator_able_to_proceed() -> None:
    """A role without SHOW privileges must not be trapped behind an empty list."""

    class _Session:
        def sql(self, _statement: str) -> object:
            raise RuntimeError("insufficient privileges")

    backend = SnowflakeBackend(_Session())

    assert backend.list_warehouses() == []
    assert backend.current_context() == {}


def test_a_stage_picker_prefers_the_stage_meant_for_the_job() -> None:
    """Falling back to whichever option sorts first picks the wrong stage.

    Alphabetically, the app's own artifact stage precedes the bulk-load stage, so
    an unset default would quietly load graph data into it.
    """

    class _Column:
        def __init__(self) -> None:
            self.index = -1

        def selectbox(self, _label: str, options: list[str], index: int, **_kw: object) -> str:
            self.index = index
            return options[index]

    column = _Column()
    chosen = _discovered_field(
        column,
        "Bulk-load stage",
        "",
        lambda: ["FLAKEGRAPH_APP_STAGE", "KG_DOCS", "KG_LOAD_STAGE", "KG_SERVICE_SPECS"],
        "help",
        prefer=("KG_LOAD_STAGE",),
    )

    assert chosen == "KG_LOAD_STAGE"


def test_a_chosen_stage_is_qualified_for_snowflake() -> None:
    """Pickers list bare names; SPCS and the bulk writer require the full form.

    Composing it from the selected namespace means a wrong identifier cannot be
    typed at all.
    """

    assert _qualified_stage("KG_SERVICE_SPECS", "NUC_DB_DEV", "SBX") == (
        "@NUC_DB_DEV.SBX.KG_SERVICE_SPECS"
    )
    # An already-qualified value is respected rather than qualified twice.
    assert _qualified_stage("@A.B.C", "DB", "SCHEMA") == "@A.B.C"
    assert _qualified_stage("", "DB", "SCHEMA") == ""


def test_submission_staging_avoids_temporary_tables() -> None:
    """Streamlit in Snowflake runs as a stored procedure.

    A temporary table is rejected there outright with "Unsupported statement type
    'temporary TABLE'", which stops ingestion at the moment of submission.
    """

    source = Path("app/flakegraph_app/backends/snowflake.py").read_text(encoding="utf-8")

    assert 'table_type="temporary"' not in source
    assert 'table_type="transient"' in source


def test_a_failed_submission_records_why() -> None:
    """A fixed message forces the reader to reproduce the failure to learn anything.

    The spec upload previously failed on a NamedTuple result and was recorded as
    the unqualified "SPCS job submission failed", which described every possible
    cause equally well.
    """

    source = Path("app/flakegraph_app/backends/snowflake.py").read_text(encoding="utf-8")

    assert "'SPCS job submission failed'" not in source
    assert 'f"SPCS job submission failed: {error}"' in source
    # The spec upload must use the shared helper rather than repeat the tuple
    # check that split one PutResult into its fields.
    assert "_validate_upload_results(_as_results(upload_result)" in source


def test_every_stage_the_worker_sees_is_qualified() -> None:
    """A bare stage name has no namespace to resolve against inside the container.

    The worker rejects it as an unsafe location, so the input stage needs the same
    qualification the spec and bulk stages already receive.
    """

    source = Path("app/flakegraph_app/ui/ingestion.py").read_text(encoding="utf-8")

    # The three stages that cross into the worker's configuration.
    assert source.count("_qualified_stage(") >= 4
    assert '"stage": _qualified_stage(' in source


def test_the_app_inlines_an_ontology_the_container_cannot_read(tmp_path: Path) -> None:
    """A profile path resolves on the author's filesystem, not inside the worker.

    Nine files failed with FileNotFoundError on the ontology because the CLI's
    specification renderer inlined it and the application's configuration builder
    did not. Both build container payloads; inlining in one leaves the other
    producing a run that cannot start.
    """

    ontology = tmp_path / "ontology.yaml"
    ontology.write_text("name: test\nentity_types: []\n", encoding="utf-8")
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        f"ontology:\n  profile_path: {ontology.name}\n",
        encoding="utf-8",
    )

    config = build_run_config(
        IngestionRequest(
            runtime=RuntimeMode.SNOWFLAKE,
            job_id="j",
            graph_id="g",
            source_kind=SourceKind.SNOWFLAKE_STAGE,
            source={"stage": "@D.S.KG_DOCS", "prefix": ""},
            ocr=ProviderSelection(provider="snowflake_cortex"),
            llm=ProviderSelection(provider="snowflake_cortex", model="m"),
            embedding=ProviderSelection(provider="snowflake_cortex", model="e", dimension=8),
            output=OutputDestination(kind=StorageKind.LOCAL, workspace_path=tmp_path / "out"),
            base_config_path=profile,
        )
    )

    assert config["ontology"]["profile"] == {"name": "test", "entity_types": []}
    assert config["ontology"]["profile_path"] is None


def test_a_snowflake_graph_carries_the_metrics_the_consumption_view_reads() -> None:
    """Cost transparency is only real if the deployed runtime supplies the numbers.

    Consumption is recorded into KG_GRAPH_METRICS by every runtime, but only the
    local workspace loader read it back. The Snowflake loader left graph_metrics
    empty, so the Consumption tab told users their graph "was built before
    consumption tracking" while the tokens and dollars sat in the table.
    """

    class Row:
        def __init__(self, values: dict[str, object]) -> None:
            self.values = values

        def as_dict(self) -> dict[str, object]:
            return self.values

    metrics = {"consumption": {"totals": {"billed_usd": 0.64, "calls": 114}}}

    class Query:
        def __init__(self, rows: list[object]) -> None:
            self.rows = rows

        def collect(self) -> list[object]:
            return self.rows

    class _Session:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def sql(self, query: str, params: object = None) -> Query:
            del params
            self.queries.append(query)
            if query.startswith("SELECT METRICS"):
                return Query([Row({"METRICS": metrics})])
            if "TABLE_NAME" in query or "UNION" in query:
                return Query([])
            return Query([])

    session = _Session()

    dataset = SnowflakeBackend(session).load_graph("current", "graph-1")

    assert any(query.startswith("SELECT METRICS") for query in session.queries)
    assert dataset.graph_metrics == metrics


class _ServiceSession:
    """Session stand-in that answers SHOW SERVICES and records every statement."""

    def __init__(self, service_rows: list[object]) -> None:
        self.service_rows = service_rows
        self.statements: list[str] = []

    def sql(self, query: str, params: object = None) -> Any:
        del params
        self.statements.append(query)
        rows = self.service_rows if query.startswith("SHOW SERVICES") else []

        class _Result:
            def collect(self) -> list[object]:
                return rows

        return _Result()


class _ServiceRow:
    def __init__(self, status: str) -> None:
        self.status = status

    def as_dict(self) -> dict[str, object]:
        return {"name": "FLAKEGRAPH_APP_ABC", "status": self.status, "is_job": "true"}


def test_a_finished_job_service_stops_blocking_the_next_submission() -> None:
    """Re-running a graph must not fail on the name its own last run left behind.

    The service name is derived from the job id, and Snowflake keeps a completed
    job service until it is dropped. Every retry of a graph therefore died with a
    bare "Object ... already exists" naming an internal service.
    """

    session = _ServiceSession([_ServiceRow("DONE")])

    SnowflakeBackend(session)._release_finished_job_service("DB.SCHEMA.FLAKEGRAPH_APP_ABC")

    assert session.statements == [
        "SHOW SERVICES LIKE 'FLAKEGRAPH_APP_ABC' IN SCHEMA DB.SCHEMA",
        "DROP SERVICE IF EXISTS DB.SCHEMA.FLAKEGRAPH_APP_ABC",
    ]


def test_a_running_job_service_is_reported_rather_than_destroyed() -> None:
    """Reusing the name of a live run would kill the work it is repeating."""

    session = _ServiceSession([_ServiceRow("RUNNING")])

    with pytest.raises(RuntimeError, match="already running"):
        SnowflakeBackend(session)._release_finished_job_service("DB.SCHEMA.FLAKEGRAPH_APP_ABC")

    assert not any(statement.startswith("DROP SERVICE") for statement in session.statements)


def test_a_graph_that_never_ran_drops_nothing() -> None:
    """A first submission has no service to release."""

    session = _ServiceSession([])

    SnowflakeBackend(session)._release_finished_job_service("DB.SCHEMA.FLAKEGRAPH_APP_ABC")

    assert session.statements == ["SHOW SERVICES LIKE 'FLAKEGRAPH_APP_ABC' IN SCHEMA DB.SCHEMA"]


def test_the_image_picker_offers_the_newest_push_first() -> None:
    """The default option must be the image most recently published.

    Sorted by name the first option is the oldest tag the repository ever held,
    and the picker offers option zero as its default. A run submitted from the
    app therefore launched flakegraph:0.1.0 — the first image ever built for the
    account — which failed on startup.
    """

    class Row:
        def __init__(self, created: str, tags: str) -> None:
            self.values: dict[str, object] = {
                "created_on": created,
                "image_name": "/DB/SCHEMA/REPO/flakegraph",
                "tags": tags,
            }

        def as_dict(self) -> dict[str, object]:
            return self.values

    class Query:
        def collect(self) -> list[object]:
            return [
                Row("2026-08-06T09:13:10+00:00", "0.1.0"),
                Row("2026-08-07T08:57:00+00:00", "0.1.6"),
                Row("2026-08-07T11:31:55+00:00", "0.1.7"),
            ]

    class _Session:
        def sql(self, query: str, params: object = None) -> Query:
            del query, params
            return Query()

    images = SnowflakeBackend(_Session()).list_images("DB.SCHEMA.REPO")

    assert list(images) == [
        "/DB/SCHEMA/REPO/flakegraph:0.1.7",
        "/DB/SCHEMA/REPO/flakegraph:0.1.6",
        "/DB/SCHEMA/REPO/flakegraph:0.1.0",
    ]


class _LostWorkerSession:
    """Session stand-in for a queued run whose job service has already stopped."""

    def __init__(
        self,
        service_status: str | None,
        job_status: str = "PENDING",
        quiet_for: int = 900,
    ) -> None:
        self.service_status = service_status
        self.job_status = job_status
        self.quiet_for = quiet_for
        self.statements: list[str] = []

    def sql(self, query: str, params: object = None) -> Any:
        del params
        self.statements.append(query)
        session = self

        class _Row:
            def __init__(self, values: dict[str, object]) -> None:
                self.values = values

            def as_dict(self) -> dict[str, object]:
                return self.values

        class _Result:
            def collect(self) -> list[object]:
                if query.startswith("SHOW SERVICES"):
                    if session.service_status is None:
                        return []
                    return [_Row({"name": "SVC", "status": session.service_status})]
                if query.startswith("SELECT J.ID"):
                    return [
                        _Row(
                            {
                                "ID": "run-1",
                                "GRAPH_ID": "graph-1",
                                "GRAPH_NAME": "graph-1",
                                "STATUS": session.job_status,
                                "ERROR": None,
                                "CREATED_AT": "2026-08-07T13:37:51",
                                "UPDATED_AT": "2026-08-07T13:37:51",
                                "PROGRESS": None,
                                "SECONDS_SINCE_UPDATE": session.quiet_for,
                                "STAGE": "queued",
                                "FILE_STATUS": "QUEUED",
                                "FILE_COUNT": 1,
                            }
                        )
                    ]
                return []

        return _Result()


def test_a_run_whose_worker_stopped_is_reported_failed_not_pending() -> None:
    """A dead worker writes nothing, so the app must read the service instead.

    Nothing advances a run except its SPCS job service. When that service dies —
    an image that cannot start, an exhausted container, a crash — no row changes
    and the app keeps reporting what it last saw: pending, zero of one, forever.
    """

    session = _LostWorkerSession("FAILED")

    snapshot = SnowflakeBackend(session).status("run-1")

    assert snapshot.status == "failed"
    assert snapshot.error is not None
    assert "stopped without processing" in snapshot.error
    assert "FAILED" in snapshot.error
    # The verdict is stored so the sidebar cannot disagree with the run page.
    assert any(
        statement.startswith("UPDATE KG_JOB SET STATUS = 'FAILED'")
        for statement in session.statements
    )


def test_a_run_whose_worker_is_alive_is_left_alone() -> None:
    """A slow run must never be declared dead while its worker is still working."""

    session = _LostWorkerSession("RUNNING")

    snapshot = SnowflakeBackend(session).status("run-1")

    assert snapshot.status == "pending"
    assert snapshot.error is None
    assert not any(statement.startswith("UPDATE KG_JOB") for statement in session.statements)


def test_an_unreadable_service_is_not_evidence_of_a_dead_worker() -> None:
    """Discovery needs privileges a viewer may not hold; absence is not death."""

    session = _LostWorkerSession(None)

    snapshot = SnowflakeBackend(session).status("run-1")

    assert snapshot.status == "pending"
    assert not any(statement.startswith("UPDATE KG_JOB") for statement in session.statements)


def test_a_busy_run_is_not_interrogated_about_its_worker() -> None:
    """Status stays one query for a healthy run; the check is for quiet runs only.

    The run page polls every few seconds. Asking Snowflake about the service on
    every poll would double the round trips for every run that is working fine.
    """

    session = _LostWorkerSession("FAILED", quiet_for=5)

    snapshot = SnowflakeBackend(session).status("run-1")

    assert snapshot.status == "pending"
    assert not any(statement.startswith("SHOW SERVICES") for statement in session.statements)


def test_a_stored_job_error_reads_as_a_sentence() -> None:
    """The cause is kept as an object, but a person is shown the explanation.

    Handed the raw column, the run page printed a JSON literal with the message
    quoted inside it.
    """

    assert _job_error_text('{"message": "The worker stopped.", "type": "worker_stopped"}') == (
        "The worker stopped."
    )
    # The job row records how many documents failed and nothing about how many
    # attempts each had. Claiming the retries were exhausted sent a reader
    # hunting for a cause when a run whose worker died leaves every document on
    # its first attempt of three, and one press of Retry finishes it.
    assert _job_error_text({"failed_files": 3}) == (
        "3 documents failed. Open the failures below for the reason."
    )
    assert _job_error_text({"failed_files": 1}) == (
        "1 document failed. Open the failures below for the reason."
    )
    assert _job_error_text(None) is None
    assert _job_error_text("") is None
    # Anything that is not the structured shape is still shown rather than hidden.
    assert _job_error_text("plain failure text") == "plain failure text"


def test_a_suspended_service_is_not_called_a_dead_worker() -> None:
    """Declaring a run dead is destructive, so it needs a state work cannot resume from.

    Reusing a finished service's name and declaring a run failed are different
    questions. SUSPENDED and DELETING free a name, but a run in one of them may
    still continue — and a wrong verdict marks a working run failed and stores it.
    """

    for state in ("SUSPENDED", "DELETING", "PENDING", "RUNNING"):
        session = _LostWorkerSession(state)

        snapshot = SnowflakeBackend(session).status("run-1")

        assert snapshot.status == "pending", state
        assert not any(
            statement.startswith("UPDATE KG_JOB") for statement in session.statements
        ), state


def test_one_concurrency_budget_reaches_every_model_calling_stage() -> None:
    """Extraction is not the only stage that waits on the provider.

    Entity resolution, community reports and description merges each spend the
    same account-level budget, and they run one after another, so each may use
    all of it. Setting only extraction leaves three stages at their old default.
    """

    assert provider_parallelism_settings(12) == {
        "extraction_parallelism": 12,
        "resolution_parallelism": 12,
        "community_report_parallelism": 12,
        "description_merge_parallelism": 12,
    }
    # A budget is a number of in-flight calls, so it cannot be zero or unbounded.
    assert provider_parallelism_settings(0)["extraction_parallelism"] == 1
    assert provider_parallelism_settings(9999)["extraction_parallelism"] == 64


def test_the_container_is_sized_from_the_node_it_is_billed_for() -> None:
    """A job service occupies a node alone and SPCS bills it whole.

    The shipped defaults asked for half a core and 3 GiB of a machine offering
    three cores and thirteen, so most of a paid-for node went unused.
    """

    resources = container_resources({"vcpu": 3, "memory_gib": 13})

    assert resources == {
        "service_cpu_request": "2",
        "service_cpu_limit": "3",
        "service_memory_request": "11Gi",
        "service_memory_limit": "11Gi",
    }


def test_an_unreadable_pool_leaves_the_container_defaults_alone() -> None:
    """Guessing at a node the app could not read would propose an unschedulable pod."""

    assert container_resources({}) == {}
    assert container_resources({"vcpu": 0, "memory_gib": 0}) == {}


def test_parallelism_is_configured_on_the_pipeline_not_the_container() -> None:
    """Concurrency belongs with the stages that spend it, not the SPCS object names."""

    with tempfile.TemporaryDirectory() as directory:
        base = _snowflake_request(Path(directory))
        request = dataclasses.replace(base, provider_parallelism=12)
        config = build_run_config(request)

    assert config["graph"]["extraction_parallelism"] == 12
    assert config["graph"]["resolution_parallelism"] == 12
    assert "provider_parallelism" not in config.get("snowflake", {})


def test_every_runtime_carries_the_operators_concurrency_budget() -> None:
    """Spend the chosen concurrency wherever the run executes.

    A stage waiting on a completion holds a socket rather than a core, so the
    budget describes the pipeline and not the machine hosting it. Applied to one
    runtime only, the same corpus runs at the library's conservative default
    everywhere else and the operator is given no sign of it.
    """

    with tempfile.TemporaryDirectory() as directory:
        base = _snowflake_request(Path(directory))
        configs = {
            runtime: build_run_config(
                dataclasses.replace(base, runtime=runtime, provider_parallelism=12)
            )
            for runtime in RuntimeMode
        }

    for runtime, config in configs.items():
        assert config["graph"]["extraction_parallelism"] == 12, runtime
        assert config["graph"]["resolution_parallelism"] == 12, runtime
        assert config["graph"]["community_report_parallelism"] == 12, runtime
        assert config["graph"]["description_merge_parallelism"] == 12, runtime
