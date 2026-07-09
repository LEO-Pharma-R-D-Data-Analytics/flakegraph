"""Command-line entry points for local, on-prem, and Snowflake-oriented runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from kg_processor import __version__
from kg_processor.adapters.jobs.snowflake import SnowflakeJobFileProgressSink, SnowflakeJobManager
from kg_processor.application.azure_openai_access import run_azure_openai_access_check
from kg_processor.application.inspect import compare_local_graph_artifacts, inspect_local_graph
from kg_processor.application.lease_heartbeat import LeaseHeartbeat, heartbeat_interval_seconds
from kg_processor.application.pipeline import KgProcessorPipeline
from kg_processor.application.progress import (
    CompositeProgressSink,
    JsonLineProgressSink,
    ProgressSink,
)
from kg_processor.application.redaction import redact_sensitive_data
from kg_processor.application.snowflake_access import run_snowflake_access_check
from kg_processor.application.snowflake_deployment import (
    render_execute_job_service_sql,
    render_kubernetes_job_yaml,
    render_snowflake_image_reference,
    render_snowflake_objects_sql,
    render_snowflake_setup_sql,
    render_spcs_service_spec_yaml,
)
from kg_processor.application.snowflake_schema import render_snowflake_schema_sql
from kg_processor.config.preflight import run_preflight
from kg_processor.config.provider_registry import (
    ProviderKind,
    provider_catalog,
    provider_catalog_for_kind,
    provider_kinds,
)
from kg_processor.config.settings import Settings
from kg_processor.domain.jobs import JobFileClaim
from kg_processor.factories import (
    build_cache,
    build_embedding_provider,
    build_file_source,
    build_job_manager,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)

APP_DISPLAY_NAME = "FlakeGraph"
app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
inspect_app = typer.Typer(no_args_is_help=True)
snowflake_app = typer.Typer(no_args_is_help=True)
azure_app = typer.Typer(no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(inspect_app, name="inspect")
app.add_typer(snowflake_app, name="snowflake")
app.add_typer(azure_app, name="azure")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{APP_DISPLAY_NAME} {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            help="Show the installed FlakeGraph version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Register top-level CLI options shared by all subcommands."""

    # The callback owns top-level options that should be available regardless of
    # which subcommand eventually runs. Keeping version handling here also makes
    # container verification independent of any provider configuration.
    _ = version


@config_app.command("print")
def print_config(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Print the resolved redacted configuration as JSON."""

    settings = Settings.load(config)
    _echo_json(redact_sensitive_data(settings.model_dump(mode="json", by_alias=True)))


@config_app.command("providers")
def print_providers(
    kind: Annotated[str | None, typer.Option("--kind")] = None,
) -> None:
    """Print the provider catalog, optionally filtered by provider kind."""

    if kind is None:
        _echo_json(provider_catalog())
        return
    if kind not in provider_kinds():
        supported = ", ".join(provider_kinds())
        raise typer.BadParameter(
            f"Unsupported provider kind '{kind}'. Supported kinds: {supported}"
        )
    _echo_json(provider_catalog_for_kind(_provider_kind(kind)))


@app.command()
def preflight(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Validate configuration and runtime prerequisites before a worker run."""

    settings = Settings.load(config)
    result = run_preflight(settings)
    _echo_json(result.model_dump(mode="json"))
    if not result.ok:
        raise typer.Exit(code=1)


@app.command()
def worker(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Run the FlakeGraph pipeline locally or as a leased Snowflake worker."""

    settings = Settings.load(config)
    job_manager = build_job_manager(settings)
    if job_manager is not None and settings.job.use_file_queue:
        _echo_json(_run_file_queue_worker(settings, job_manager))
        return
    elif job_manager is not None:
        if not settings.job.lease_owner:
            raise ValueError("job.use_lease requires job.lease_owner")
        claim = job_manager.claim_job(
            settings.job.job_id,
            settings.job.graph_id,
            settings.job.lease_owner,
            settings.job.lease_seconds,
            redact_sensitive_data(settings.model_dump(mode="json", by_alias=True)),
        )
        if not claim.claimed:
            _echo_json(claim.model_dump(mode="json"))
            return
    pipeline = _build_pipeline(settings)
    try:
        if job_manager is not None and settings.job.lease_owner:
            with _job_heartbeat(settings, job_manager):
                batch = pipeline.run()
        else:
            batch = pipeline.run()
    except Exception as exc:
        if job_manager is not None and settings.job.lease_owner:
            job_manager.fail_job(
                settings.job.job_id,
                settings.job.lease_owner,
                {"type": type(exc).__name__, "message": str(exc)},
            )
        raise
    if job_manager is not None and settings.job.lease_owner:
        job_manager.complete_job(settings.job.job_id, settings.job.lease_owner, batch.run_report)
    _echo_json(batch.run_report)


def _run_file_queue_worker(
    settings: Settings,
    job_manager: SnowflakeJobManager,
) -> dict[str, Any]:
    if not settings.job.lease_owner:
        raise ValueError("job.use_file_queue requires job.lease_owner")
    batches_processed = 0
    files_processed = 0
    last_run_report: dict[str, Any] | None = None
    while True:
        claimed_files = job_manager.claim_job_files(
            settings.job.job_id,
            settings.job.graph_id,
            settings.job.lease_owner,
            settings.job.lease_seconds,
            settings.job.file_batch_size,
        )
        if not claimed_files:
            return {
                "claimed": batches_processed > 0,
                "drained": True,
                "job_id": settings.job.job_id,
                "graph_id": settings.job.graph_id,
                "batches_processed": batches_processed,
                "files_processed": files_processed,
                "last_run_report": last_run_report,
            }
        pipeline = _build_pipeline(
            settings,
            claimed_files,
            _file_queue_progress_sink(settings, job_manager, claimed_files),
        )
        try:
            with _job_file_heartbeat(settings, job_manager, claimed_files):
                batch = pipeline.run()
        except Exception as exc:
            job_manager.fail_job_files(
                settings.job.job_id,
                settings.job.graph_id,
                settings.job.lease_owner,
                [claim.file_id for claim in claimed_files],
                {"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        job_manager.complete_job_files(
            settings.job.job_id,
            settings.job.graph_id,
            settings.job.lease_owner,
            pipeline.job_file_results(batch),
        )
        job_manager.complete_job_if_file_queue_drained(
            settings.job.job_id,
            settings.job.graph_id,
            batch.run_report,
        )
        batches_processed += 1
        files_processed += int(batch.run_report.get("files_processed", 0))
        last_run_report = batch.run_report


def _job_heartbeat(
    settings: Settings,
    job_manager: SnowflakeJobManager,
) -> LeaseHeartbeat:
    lease_owner = settings.job.lease_owner
    if not lease_owner:
        raise ValueError("job heartbeat requires job.lease_owner")
    return LeaseHeartbeat(
        lambda: job_manager.heartbeat_job(
            settings.job.job_id,
            lease_owner,
            settings.job.lease_seconds,
        ),
        heartbeat_interval_seconds(settings.job.lease_seconds),
    )


def _job_file_heartbeat(
    settings: Settings,
    job_manager: SnowflakeJobManager,
    claimed_files: list[JobFileClaim],
) -> LeaseHeartbeat:
    lease_owner = settings.job.lease_owner
    if not lease_owner:
        raise ValueError("file heartbeat requires job.lease_owner")
    file_ids = [claim.file_id for claim in claimed_files]
    return LeaseHeartbeat(
        lambda: job_manager.heartbeat_job_files(
            settings.job.job_id,
            settings.job.graph_id,
            lease_owner,
            file_ids,
            settings.job.lease_seconds,
        ),
        heartbeat_interval_seconds(settings.job.lease_seconds),
    )


def _build_pipeline(
    settings: Settings,
    claimed_files: list[JobFileClaim] | None = None,
    progress_sink: ProgressSink | None = None,
) -> KgProcessorPipeline:
    return KgProcessorPipeline(
        settings=settings,
        file_source=build_file_source(settings),
        ocr=build_ocr_provider(settings),
        llm=build_llm_provider(settings),
        embeddings=build_embedding_provider(settings),
        writer=build_writer(settings),
        cache=build_cache(settings),
        claimed_files=claimed_files,
        progress_sink=progress_sink or JsonLineProgressSink(),
    )


def _file_queue_progress_sink(
    settings: Settings,
    job_manager: SnowflakeJobManager,
    claimed_files: list[JobFileClaim],
) -> ProgressSink:
    lease_owner = settings.job.lease_owner
    if not lease_owner:
        raise ValueError("file progress persistence requires job.lease_owner")
    return CompositeProgressSink(
        [
            JsonLineProgressSink(),
            SnowflakeJobFileProgressSink(
                job_manager,
                settings.job.job_id,
                settings.job.graph_id,
                lease_owner,
                [claim.file_id for claim in claimed_files],
            ),
        ]
    )


@inspect_app.command("graph")
def inspect_graph(
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("out/kg"),
) -> None:
    """Inspect a local artifact directory and print a review report."""

    _echo_json(inspect_local_graph(output))


@inspect_app.command("compare")
def inspect_compare(
    left: Annotated[Path, typer.Option("--left")] = Path("out/kg-left"),
    right: Annotated[Path, typer.Option("--right")] = Path("out/kg-right"),
) -> None:
    """Compare two local artifact directories for deterministic parity."""

    result = compare_local_graph_artifacts(left, right)
    _echo_json(result)
    if not result["ok"]:
        raise typer.Exit(code=1)


@azure_app.command("openai-access-check")
def azure_openai_access_check(
    subscription: Annotated[str, typer.Option("--subscription")] = "",
    resource_group: Annotated[str, typer.Option("--resource-group")] = "",
    account: Annotated[str, typer.Option("--account")] = "",
    llm_deployment: Annotated[str | None, typer.Option("--llm-deployment")] = None,
    embedding_deployment: Annotated[str | None, typer.Option("--embedding-deployment")] = None,
) -> None:
    """Check current Azure CLI access to an Azure OpenAI account."""

    # Keep this command non-mutating and secret-safe. It confirms that the Azure
    # CLI session can see the account, deployments, and key material without
    # returning the key itself.
    _require_option(subscription, "--subscription")
    _require_option(resource_group, "--resource-group")
    _require_option(account, "--account")
    report = run_azure_openai_access_check(
        subscription=subscription,
        resource_group=resource_group,
        account=account,
        llm_deployment=llm_deployment,
        embedding_deployment=embedding_deployment,
    )
    _echo_json(report.model_dump(mode="json"))
    if not report.ok:
        raise typer.Exit(code=1)


@snowflake_app.command("ddl")
def snowflake_ddl(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    embedding_dim: Annotated[int | None, typer.Option("--embedding-dim")] = None,
) -> None:
    """Render canonical Snowflake table DDL."""

    settings = Settings.load(config)
    dim = embedding_dim or settings.embedding.dimension
    typer.echo(render_snowflake_schema_sql(dim).strip())


@snowflake_app.command("service-spec")
def snowflake_service_spec(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    config_path: Annotated[str, typer.Option("--config-path")] = "configs/snowflake-cortex.yaml",
) -> None:
    """Render a Snowpark Container Services job spec."""

    settings = Settings.load(config)
    typer.echo(render_spcs_service_spec_yaml(settings, config_path).strip())


@snowflake_app.command("image-reference")
def snowflake_image_reference(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Render the fully qualified Snowflake image repository reference."""

    settings = Settings.load(config)
    typer.echo(render_snowflake_image_reference(settings))


@snowflake_app.command("execute-job-sql")
def snowflake_execute_job_sql(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    spec_file: Annotated[str, typer.Option("--spec-file")] = "flakegraph-job.yaml",
    async_execution: Annotated[bool, typer.Option("--async")] = False,
) -> None:
    """Render SQL for executing the SPCS job service."""

    settings = Settings.load(config)
    typer.echo(render_execute_job_service_sql(settings, spec_file, async_execution).strip())


@snowflake_app.command("kubernetes-job")
def snowflake_kubernetes_job(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    image: Annotated[str, typer.Option("--image")] = "flakegraph:latest",
    config_path: Annotated[
        str,
        typer.Option("--config-path"),
    ] = "configs/onprem-azure-blob-vllm-mineru-oss.yaml",
    secret_name: Annotated[str | None, typer.Option("--secret-name")] = None,
    parallelism: Annotated[int, typer.Option("--parallelism")] = 1,
    completions: Annotated[int | None, typer.Option("--completions")] = None,
    gpu_count: Annotated[int | None, typer.Option("--gpu-count")] = None,
) -> None:
    """Render an on-prem Kubernetes Job manifest using the same settings model."""

    settings = Settings.load(config)
    typer.echo(
        render_kubernetes_job_yaml(
            settings,
            image=image,
            config_path=config_path,
            secret_name=secret_name,
            parallelism=parallelism,
            completions=completions,
            gpu_count=gpu_count,
        ).strip()
    )


@snowflake_app.command("setup-sql")
def snowflake_setup_sql(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    role_name: Annotated[str, typer.Option("--role-name")] = "KG_PROCESSOR_ROLE",
) -> None:
    """Render admin-oriented Snowflake setup SQL including grants."""

    settings = Settings.load(config)
    typer.echo(render_snowflake_setup_sql(settings, role_name).strip())


@snowflake_app.command("objects-sql")
def snowflake_objects_sql(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Render schema-object Snowflake setup SQL for lower-privilege execution."""

    settings = Settings.load(config)
    typer.echo(render_snowflake_objects_sql(settings).strip())


@snowflake_app.command("access-check")
def snowflake_access_check(
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    skip_cortex: Annotated[bool, typer.Option("--skip-cortex")] = False,
) -> None:
    """Check current Snowflake access to required objects and optional Cortex calls."""

    settings = Settings.load(config)
    report = run_snowflake_access_check(settings, check_cortex=not skip_cortex)
    _echo_json(report.model_dump(mode="json"))
    if not report.ok:
        raise typer.Exit(code=1)


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


def _require_option(value: str, option_name: str) -> None:
    if not value.strip():
        raise typer.BadParameter(f"{option_name} is required")


def _provider_kind(value: str) -> ProviderKind:
    if value == "file_source":
        return "file_source"
    if value == "ocr":
        return "ocr"
    if value == "llm":
        return "llm"
    if value == "embedding":
        return "embedding"
    if value == "writer":
        return "writer"
    if value == "cache":
        return "cache"
    raise ValueError(f"Unsupported provider kind: {value}")
