from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from kg_processor import __version__
from kg_processor.adapters.jobs.snowflake import SnowflakeJobManager
from kg_processor.application.azure_openai_access import (
    AzureOpenAIAccessCheck,
    AzureOpenAIAccessReport,
)
from kg_processor.application.progress import ProgressEvent, ProgressSink
from kg_processor.application.snowflake_access import SnowflakeAccessCheck, SnowflakeAccessReport
from kg_processor.cli import _run_file_queue_worker, app
from kg_processor.config.settings import Settings
from kg_processor.domain.graph import GraphWriteBatch
from kg_processor.domain.jobs import JobFileClaim, JobFileResult

runner = CliRunner()


def test_cli_version_option_reports_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"FlakeGraph {__version__}"


def test_cli_config_print_redacts_api_keys(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  provider: openai_compatible
  endpoint: https://example.test/v1
  api_key: super-secret
snowflake:
  password: snowflake-password
  oauth_token: bearer-token
  private_key_path: /tmp/private-key.pem
azure_blob:
  connection_string: DefaultEndpointsProtocol=https;AccountKey=secret
  sas_token: sv=secret
generic_http_ocr:
  api_key: ocr-secret
  api_key_header: X-API-Key
  api_key_prefix: ""
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["config", "print", "--config", str(config)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["llm"]["api_key"] == "***"
    assert payload["snowflake"]["password"] == "***"
    assert payload["snowflake"]["oauth_token"] == "***"
    assert payload["snowflake"]["private_key_path"] == "***"
    assert payload["azure_blob"]["connection_string"] == "***"
    assert payload["azure_blob"]["sas_token"] == "***"
    assert payload["generic_http_ocr"]["api_key"] == "***"
    assert payload["generic_http_ocr"]["api_key_header"] == "X-API-Key"
    assert payload["generic_http_ocr"]["api_key_prefix"] == ""


def test_cli_config_providers_lists_supported_provider_catalog() -> None:
    result = runner.invoke(app, ["config", "providers", "--kind", "ocr"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    names = {provider["name"] for provider in payload}
    assert {provider["kind"] for provider in payload} == {"ocr"}
    assert {"mineru_internal", "mineru_api", "snowflake_cortex"} <= names
    assert "openai_compatible" not in names


def test_cli_config_providers_rejects_unknown_kind() -> None:
    result = runner.invoke(app, ["config", "providers", "--kind", "unknown"])

    assert result.exit_code != 0
    assert "Unsupported provider kind 'unknown'" in result.output


def test_cli_preflight_and_worker_roundtrip(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
files:
  input_path: {input_dir}
ocr:
  provider: builtin_text
llm:
  provider: fake
  model: fake
embedding:
  provider: hash
  dimension: 8
graph:
  chunk_token_size: 30
  chunk_token_overlap: 5
  gleaning_max_passes: 0
writer:
  output_path: {output_dir}
""",
        encoding="utf-8",
    )

    preflight = runner.invoke(app, ["preflight", "--config", str(config)])
    worker = runner.invoke(app, ["worker", "--config", str(config)])
    inspect = runner.invoke(app, ["inspect", "graph", "--output", str(output_dir)])

    assert preflight.exit_code == 0
    assert worker.exit_code == 0
    assert inspect.exit_code == 0
    report = json.loads(worker.stdout)
    inspection = json.loads(inspect.stdout)
    progress = [json.loads(line) for line in worker.stderr.splitlines() if line.strip()]
    assert report["files_processed"] == 1
    assert inspection["tables"]["nodes"] > 0
    assert inspection["trace_events"] > 0
    assert any(event["stage"] == "ocr" and event["status"] == "completed" for event in progress)
    assert any(event["stage"] == "write" and event["status"] == "completed" for event in progress)


def test_cli_compare_reports_stable_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    first_output = tmp_path / "out-first"
    second_output = tmp_path / "out-second"
    input_dir.mkdir()
    (input_dir / "sample.txt").write_text(
        "Alice Smith works at Acme Corp in Copenhagen.",
        encoding="utf-8",
    )
    first_config = _write_local_config(tmp_path / "first.yaml", input_dir, first_output)
    second_config = _write_local_config(tmp_path / "second.yaml", input_dir, second_output)

    first_worker = runner.invoke(app, ["worker", "--config", str(first_config)])
    second_worker = runner.invoke(app, ["worker", "--config", str(second_config)])
    compare = runner.invoke(
        app,
        ["inspect", "compare", "--left", str(first_output), "--right", str(second_output)],
    )

    assert first_worker.exit_code == 0
    assert second_worker.exit_code == 0
    assert compare.exit_code == 0
    comparison = json.loads(compare.stdout)
    assert comparison["ok"]
    assert any(check["name"] == "nodes_stable_identity" for check in comparison["checks"])


def test_cli_azure_openai_access_check_prints_secret_safe_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_access_check(**kwargs: object) -> AzureOpenAIAccessReport:
        assert kwargs["subscription"] == "sub"
        assert kwargs["resource_group"] == "rg"
        assert kwargs["account"] == "ai-account"
        assert kwargs["llm_deployment"] == "chat"
        assert kwargs["embedding_deployment"] == "embed"
        return AzureOpenAIAccessReport(
            ok=True,
            subscription="sub",
            resource_group="rg",
            account="ai-account",
            endpoint="https://example.cognitiveservices.azure.com/",
            checks=[
                AzureOpenAIAccessCheck(
                    name="api_key",
                    ok=True,
                    message="Azure OpenAI API key is readable",
                    details={"present": True},
                )
            ],
        )

    monkeypatch.setattr("kg_processor.cli.run_azure_openai_access_check", fake_access_check)

    result = runner.invoke(
        app,
        [
            "azure",
            "openai-access-check",
            "--subscription",
            "sub",
            "--resource-group",
            "rg",
            "--account",
            "ai-account",
            "--llm-deployment",
            "chat",
            "--embedding-deployment",
            "embed",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["checks"][0]["details"] == {"present": True}


def test_cli_snowflake_ddl_uses_configured_embedding_dimension(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
embedding:
  dimension: 42
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["snowflake", "ddl", "--config", str(config)])

    assert result.exit_code == 0
    assert "VECTOR(FLOAT, 42)" in result.stdout
    assert "CREATE TABLE IF NOT EXISTS KG_GRAPH_METRICS" in result.stdout


def test_file_queue_worker_drains_claimed_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings.load(
        overrides={
            "job": {
                "job_id": "job",
                "graph_id": "graph",
                "use_file_queue": True,
                "lease_owner": "worker-1",
                "file_batch_size": 1,
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )
    manager = _FakeQueueManager(
        [
            [JobFileClaim(job_id="job", graph_id="graph", file_id="file_1")],
            [JobFileClaim(job_id="job", graph_id="graph", file_id="file_2")],
            [],
        ]
    )

    def fake_build_pipeline(
        _settings: Settings,
        claimed_files: list[JobFileClaim] | None = None,
        progress_sink: ProgressSink | None = None,
    ) -> _FakeQueuePipeline:
        return _FakeQueuePipeline(claimed_files or [], progress_sink)

    monkeypatch.setattr("kg_processor.cli._build_pipeline", fake_build_pipeline)

    summary = _run_file_queue_worker(settings, cast(SnowflakeJobManager, manager))

    assert summary["claimed"] is True
    assert summary["drained"] is True
    assert summary["batches_processed"] == 2
    assert summary["files_processed"] == 2
    assert manager.claim_calls == 3
    assert manager.completed_file_ids == [["file_1"], ["file_2"]]
    assert manager.drained_reports == ["run_file_1", "run_file_2"]
    assert manager.progress_updates == [
        ("job", "graph", "worker-1", ["file_1"], "ocr:started"),
        ("job", "graph", "worker-1", ["file_2"], "ocr:started"),
    ]


def test_file_queue_worker_heartbeats_claimed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.load(
        overrides={
            "job": {
                "job_id": "job",
                "graph_id": "graph",
                "use_file_queue": True,
                "lease_owner": "worker-1",
                "lease_seconds": 6,
                "file_batch_size": 1,
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )
    manager = _FakeQueueManager(
        [
            [JobFileClaim(job_id="job", graph_id="graph", file_id="file_1")],
            [],
        ]
    )
    heartbeat_intervals: list[float] = []

    def fake_build_pipeline(
        _settings: Settings,
        claimed_files: list[JobFileClaim] | None = None,
        progress_sink: ProgressSink | None = None,
    ) -> _FakeQueuePipeline:
        return _FakeQueuePipeline(claimed_files or [], progress_sink)

    class ImmediateHeartbeat:
        def __init__(self, heartbeat: Callable[[], None], interval_seconds: float) -> None:
            self.heartbeat = heartbeat
            heartbeat_intervals.append(interval_seconds)

        def __enter__(self) -> ImmediateHeartbeat:
            self.heartbeat()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object,
        ) -> None:
            return None

    monkeypatch.setattr("kg_processor.cli._build_pipeline", fake_build_pipeline)
    monkeypatch.setattr("kg_processor.cli.LeaseHeartbeat", ImmediateHeartbeat)

    summary = _run_file_queue_worker(settings, cast(SnowflakeJobManager, manager))

    assert summary["files_processed"] == 1
    assert heartbeat_intervals == [2]
    assert manager.heartbeat_file_ids == [["file_1"]]


def _write_local_config(path: Path, input_dir: Path, output_dir: Path) -> Path:
    path.write_text(
        f"""
job:
  job_id: stable-test
  graph_id: stable-graph
files:
  input_path: {input_dir}
ocr:
  provider: builtin_text
llm:
  provider: fake
  model: fake
embedding:
  provider: hash
  dimension: 8
graph:
  chunk_token_size: 30
  chunk_token_overlap: 5
  gleaning_max_passes: 0
writer:
  output_path: {output_dir}
""",
        encoding="utf-8",
    )
    return path


class _FakeQueueManager:
    def __init__(self, claim_batches: list[list[JobFileClaim]]) -> None:
        self.claim_batches = claim_batches
        self.claim_calls = 0
        self.completed_file_ids: list[list[str]] = []
        self.drained_reports: list[str] = []
        self.heartbeat_file_ids: list[list[str]] = []
        self.progress_updates: list[tuple[str, str, str, list[str], str]] = []

    def claim_job_files(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> list[JobFileClaim]:
        self.claim_calls += 1
        return self.claim_batches.pop(0)

    def complete_job_files(
        self,
        _job_id: str,
        _graph_id: str,
        _worker_id: str,
        results: list[JobFileResult],
    ) -> None:
        self.completed_file_ids.append([result.file_id for result in results])

    def complete_job_if_file_queue_drained(
        self,
        _job_id: str,
        _graph_id: str,
        report: dict[str, object],
    ) -> None:
        self.drained_reports.append(str(report["run_id"]))

    def fail_job_files(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("fail_job_files should not be called")

    def heartbeat_job_files(
        self,
        _job_id: str,
        _graph_id: str,
        _worker_id: str,
        file_ids: list[str],
        _lease_seconds: int,
    ) -> None:
        self.heartbeat_file_ids.append(file_ids)

    def update_job_file_progress(
        self,
        job_id: str,
        graph_id: str,
        worker_id: str,
        file_ids: list[str],
        stage: str,
    ) -> None:
        self.progress_updates.append((job_id, graph_id, worker_id, file_ids, stage))


class _FakeQueuePipeline:
    def __init__(
        self,
        claimed_files: list[JobFileClaim],
        progress_sink: ProgressSink | None = None,
    ) -> None:
        self.claimed_files = claimed_files
        self.progress_sink = progress_sink

    def run(self) -> GraphWriteBatch:
        file_ids = [claim.file_id for claim in self.claimed_files]
        if self.progress_sink is not None:
            self.progress_sink.emit(
                ProgressEvent(
                    job_id="job",
                    graph_id="graph",
                    stage="ocr",
                    status="started",
                    file_id=file_ids[0],
                )
            )
        return GraphWriteBatch(
            graph_id="graph",
            write_scope="file_batch",
            reindex_file_ids=file_ids,
            documents=[],
            pages=[],
            chunks=[],
            nodes=[],
            edges=[],
            evidence=[],
            entity_sources=[],
            communities=[],
            community_findings=[],
            run_report={
                "job_id": "job",
                "graph_id": "graph",
                "run_id": f"run_{file_ids[0]}",
                "files_processed": len(file_ids),
            },
            graph_metrics={},
            extraction_trace=[],
        )

    def job_file_results(self, _batch: GraphWriteBatch) -> list[JobFileResult]:
        return [
            JobFileResult(file_id=claim.file_id, rows_written=1) for claim in self.claimed_files
        ]


def test_cli_snowflake_deployment_commands_render_from_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
runtime:
  runtime: spcs
job:
  job_id: job-123
  graph_id: graph-123
  use_lease: true
  lease_owner: worker-1
files:
  source: snowflake_stage
ocr:
  provider: snowflake_cortex
llm:
  provider: snowflake_cortex
  model: llama3.3-70b
embedding:
  provider: snowflake_cortex
  model: snowflake-arctic-embed-l-v2.0
  dimension: 1024
writer:
  provider: snowflake_bulk
cache:
  provider: snowflake
snowflake:
  account: EXAMPLE_ACCOUNT
  database: KG_DB
  schema: GRAPH
  role: KG_PROCESSOR_ROLE
  warehouse: KG_PROCESSOR_WH
  stage: "@KG_DB.GRAPH.KG_DOCS"
  bulk_stage: "@KG_DB.GRAPH.KG_LOAD_STAGE"
  image_repository: KG_DB.GRAPH.KG_IMAGES
  image_name: flakegraph:latest
  image_digest: sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  compute_pool: KG_PROCESSOR_CPU_POOL
  service_name: KG_PROCESSOR_JOB
  service_spec_stage: "@KG_DB.GRAPH.KG_SERVICE_SPECS"
""",
        encoding="utf-8",
    )

    spec = runner.invoke(app, ["snowflake", "service-spec", "--config", str(config)])
    image_reference = runner.invoke(
        app,
        ["snowflake", "image-reference", "--config", str(config)],
    )
    execute_sql = runner.invoke(app, ["snowflake", "execute-job-sql", "--config", str(config)])
    setup_sql = runner.invoke(app, ["snowflake", "setup-sql", "--config", str(config)])
    objects_sql = runner.invoke(app, ["snowflake", "objects-sql", "--config", str(config)])
    kubernetes = runner.invoke(
        app,
        [
            "snowflake",
            "kubernetes-job",
            "--config",
            str(config),
            "--image",
            "registry.example/flakegraph:latest",
            "--secret-name",
            "flakegraph-secrets",
            "--parallelism",
            "2",
            "--gpu-count",
            "1",
        ],
    )

    assert spec.exit_code == 0
    assert image_reference.exit_code == 0
    assert execute_sql.exit_code == 0
    assert setup_sql.exit_code == 0
    assert objects_sql.exit_code == 0
    assert kubernetes.exit_code == 0
    assert (
        "/KG_DB/GRAPH/KG_IMAGES/flakegraph:latest@"
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    ) in spec.stdout
    assert image_reference.stdout.strip() == (
        "/KG_DB/GRAPH/KG_IMAGES/flakegraph:latest@"
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    assert "KG_SNOWFLAKE_AUTHENTICATOR: oauth" in spec.stdout
    assert "EXECUTE JOB SERVICE" in execute_sql.stdout
    assert "FROM @KG_DB.GRAPH.KG_SERVICE_SPECS" in execute_sql.stdout
    assert "CREATE IMAGE REPOSITORY IF NOT EXISTS KG_DB.GRAPH.KG_IMAGES" in setup_sql.stdout
    assert "CREATE TABLE IF NOT EXISTS KG_NODE" in setup_sql.stdout
    assert "CREATE ROLE" not in objects_sql.stdout
    assert "GRANT " not in objects_sql.stdout
    assert "CREATE STAGE IF NOT EXISTS KG_DB.GRAPH.KG_DOCS" in objects_sql.stdout
    assert "CREATE TABLE IF NOT EXISTS KG_NODE" in objects_sql.stdout
    assert "kind: Job" in kubernetes.stdout
    assert "parallelism: 2" in kubernetes.stdout
    assert "name: flakegraph-secrets" in kubernetes.stdout
    assert "nvidia.com/gpu: '1'" in kubernetes.stdout


def test_cli_snowflake_access_check_outputs_report_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
snowflake:
  account: EXAMPLE_ACCOUNT
  database: KG_DB
  schema: GRAPH
""",
        encoding="utf-8",
    )

    def fake_access_check(*_args: object, **_kwargs: object) -> SnowflakeAccessReport:
        return SnowflakeAccessReport(
            ok=False,
            checks=[
                SnowflakeAccessCheck(
                    name="target_tables",
                    ok=False,
                    message="Some FlakeGraph tables are missing or not visible",
                    details={"missing": ["KG_NODE"]},
                )
            ],
        )

    monkeypatch.setattr("kg_processor.cli.run_snowflake_access_check", fake_access_check)

    result = runner.invoke(app, ["snowflake", "access-check", "--config", str(config)])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["checks"][0]["details"]["missing"] == ["KG_NODE"]
