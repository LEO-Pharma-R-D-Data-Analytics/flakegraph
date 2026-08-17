"""Ingestion configuration and submission page."""

from __future__ import annotations

import os
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import streamlit as st
import yaml
from flakegraph_app.backends.base import ControlPlaneBackend
from flakegraph_app.configuration import (
    build_run_config,
    container_resources,
    effective_request_fingerprint,
    load_base_config,
    redacted_config,
    resolve_profile_path,
)
from flakegraph_app.models import (
    DEFAULT_PROVIDER_PARALLELISM,
    IngestionRequest,
    OutputDestination,
    RuntimeMode,
    SnowflakeOutput,
    SourceKind,
    StorageKind,
)
from flakegraph_app.providers import (
    CORTEX_EMBEDDING_MODELS_BY_WIDTH,
    EMBEDDING_PROVIDERS,
    LLM_PROVIDERS,
    OCR_PROVIDERS,
    cortex_embedding_model_for_width,
)
from flakegraph_app.sources import SUPPORTED_SUFFIXES
from flakegraph_app.ui.cache_state import (
    account_context,
    account_options,
    graph_embedding_width,
    invalidate_run_history,
)
from flakegraph_app.ui.shared import (
    format_bytes,
    provider_controls,
    secret_text_input,
    seeded_widget,
)
from flakegraph_app.ui.theme import page_heading
from flakegraph_app.viewer import current_viewer, viewer_stage_prefix


class _StageUploadBackend(Protocol):
    """Narrow capability required only by Snowflake's staged upload control."""

    def upload_files(self, paths: Sequence[Path], stage: str, prefix: str = "uploads") -> int:
        """Upload local paths into one Snowflake stage prefix."""
        ...


class _SessionState(Protocol):
    """Structural state operations used by security-sensitive form helpers."""

    def get(self, key: str, default: Any = None, /) -> Any:
        """Return a stored value or its default."""
        ...

    def __setitem__(self, key: str, value: Any) -> None:
        """Store one session-scoped value."""
        ...

    def __delitem__(self, key: str) -> None:
        """Remove a session value."""
        ...


def render_ingestion(
    backend: ControlPlaneBackend,
    runtime: RuntimeMode,
    repository_root: Path,
) -> None:
    """Render source selection, provider controls, preflight, and submission."""

    page_heading(
        "Ingestion",
        "Build a graph",
        "Choose a corpus and processing adapters, validate the run, then submit it "
        "to the selected runtime.",
    )
    request = _request_controls(backend, runtime, repository_root)
    if request is None:
        return

    try:
        effective_config = build_run_config(request)
    except ValueError as exc:
        st.error(str(exc))
        return
    request_fingerprint = effective_request_fingerprint(request, effective_config)
    with st.expander("Effective configuration"):
        st.code(
            yaml.safe_dump(redacted_config(effective_config), sort_keys=False),
            language="yaml",
        )

    left, right = st.columns([1, 1])
    if left.button(
        "Run preflight",
        type="secondary",
        width="stretch",
        help=(
            "Validate source access, provider configuration, credentials, runtime "
            "capacity, and output prerequisites without starting ingestion."
        ),
    ):
        with st.spinner("Checking configuration and providers"):
            try:
                result = backend.preflight(request)
            except Exception as exc:
                result = {"ok": False, "errors": [str(exc)]}
            _record_preflight(
                cast(_SessionState, st.session_state),
                request_fingerprint,
                result,
            )
    preflight = _preflight_for_request(
        cast(_SessionState, st.session_state),
        request_fingerprint,
    )
    if preflight:
        if preflight.get("ok"):
            st.success("Preflight passed. The run is ready to submit.")
        else:
            errors = preflight.get("errors", ["Preflight failed"])
            messages = (
                errors
                if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes))
                else [errors]
            )
            for error in messages:
                st.error(str(error))
    if right.button(
        "Start ingestion",
        type="primary",
        width="stretch",
        disabled=not bool(preflight and preflight.get("ok")),
        help=(
            "Submit the validated run. Processing continues in the selected runtime "
            "and can be followed from its graph entry in the sidebar."
        ),
    ):
        with st.spinner("Submitting graph run"):
            try:
                latest_fingerprint = effective_request_fingerprint(request)
                if (
                    _approved_preflight(
                        cast(_SessionState, st.session_state),
                        latest_fingerprint,
                    )
                    is None
                ):
                    raise ValueError(
                        "The effective request changed after preflight. Run preflight again."
                    )
                snapshot = backend.submit(request)
            except Exception as exc:
                st.error(str(exc))
            else:
                invalidate_run_history()
                st.session_state["selected_run_id"] = snapshot.run_id
                st.session_state["active_page"] = "run"
                st.session_state["preflight"] = None
                _release_submitted_identity(cast(_SessionState, st.session_state))
                st.rerun()


def _request_controls(
    backend: ControlPlaneBackend,
    runtime: RuntimeMode,
    repository_root: Path,
) -> IngestionRequest | None:
    """Collect a valid request while keeping source and provider concerns separate."""

    default_id = _default_run_id()
    identity_columns = st.columns(2)
    job_id = identity_columns[0].text_input(
        "Run ID",
        value=default_id,
        key="ingest_job_id",
        help="Unique execution identifier used for status, retries, logs, and audit records.",
    )
    graph_id = identity_columns[1].text_input(
        "Graph ID",
        value=default_id,
        key="ingest_graph_id",
        help="Stable name of the knowledge graph produced by this run.",
    )
    if not _safe_id(job_id) or not _safe_id(graph_id):
        st.error("Run and graph IDs may contain letters, numbers, dots, underscores, and hyphens.")
        return None

    st.subheader("Source")
    source_kind, source = _source_controls(backend, runtime, repository_root, job_id)
    if source is None:
        return None

    default_profile = _default_profile(runtime, repository_root)
    st.subheader("Processing")
    provider_columns = st.columns(3)
    defaults = _provider_defaults(runtime, default_profile)
    # The destination's vector width is a hard constraint on the embedding model,
    # so it is read before the control is drawn rather than enforced afterwards.
    stored_width = graph_embedding_width(runtime.value, _backend=backend)
    _synchronize_runtime_provider_state(
        cast(_SessionState, st.session_state),
        runtime,
        defaults,
        embedding_width=stored_width,
    )
    # Workers claim only runs whose semantic configuration equals their own, so
    # the models the fleet actually runs are the only ones this form can submit.
    fleet = _fleet_models(backend)
    with provider_columns[0]:
        ocr = provider_controls("OCR", "ocr", OCR_PROVIDERS, defaults["ocr"])
    with provider_columns[1]:
        llm = provider_controls(
            "LLM",
            "llm",
            LLM_PROVIDERS,
            defaults["llm"],
            deployed_model=fleet.get("llm"),
        )
    with provider_columns[2]:
        embedding = provider_controls(
            "Embeddings",
            "embedding",
            EMBEDDING_PROVIDERS,
            defaults["embedding"],
            required_dimension=stored_width,
            deployed_model=fleet.get("embedding"),
        )

    st.subheader("Destination")
    output = _output_controls(
        runtime, repository_root, graph_id, job_id, default_profile, backend
    )
    if output is None:
        return None

    with st.expander("Run settings"):
        base_config_text = st.text_input(
            "Base configuration",
            value=str(default_profile) if default_profile.exists() else "",
            help=(
                "Advanced graph, ontology, distributed-store, and Snowflake settings "
                "are inherited from this profile, except where a control on this "
                "page sets the same value. It must be a file the application ships "
                "or the repository provides."
            ),
        )
        include_globs = st.text_input(
            "Include patterns",
            value="**/*",
            help=(
                "Comma-separated glob patterns evaluated inside the selected source; "
                "unsupported file types are still ignored."
            ),
        )
        cache_provider = st.selectbox(
            "Cache",
            ["local", "none", "snowflake"],
            index=(
                2
                if runtime == RuntimeMode.SNOWFLAKE
                else (1 if runtime == RuntimeMode.KUBERNETES else 0)
            ),
            help=(
                "Reuse expensive OCR and model results when inputs and processing "
                "settings are unchanged."
            ),
        )
        # Every stage that calls a model is bounded by how many calls the provider
        # will serve at once, not by the machine's cores: a window waiting on a
        # completion holds a socket, not a CPU. One control governs all of them,
        # in every runtime, and each stage applies it in turn.
        provider_parallelism = int(
            st.number_input(
                "Concurrent provider calls",
                min_value=1,
                max_value=64,
                value=DEFAULT_PROVIDER_PARALLELISM,
                step=1,
                help=(
                    "How many model calls each processing stage may have in flight. "
                    "Extraction, entity resolution, community reports and description "
                    "merges all use this. Raising it shortens a large corpus roughly "
                    "in proportion until the provider's own limit is reached; lower it "
                    "if runs start reporting provider throttling."
                ),
            )
        )
        runtime_options = _snowflake_runtime_controls(runtime, default_profile, backend)
    base_path: Path | None = None
    if base_config_text.strip():
        try:
            base_path = resolve_profile_path(
                base_config_text,
                repository_root,
                Path(__file__).resolve().parents[2],
            )
        except ValueError as exc:
            st.error(str(exc))
            return None
    return IngestionRequest(
        runtime=runtime,
        job_id=job_id,
        graph_id=graph_id,
        source_kind=source_kind,
        source=source,
        ocr=ocr,
        llm=llm,
        embedding=embedding,
        output=output,
        base_config_path=base_path,
        include_globs=tuple(item.strip() for item in include_globs.split(",") if item.strip()),
        cache_provider=cache_provider,
        provider_parallelism=provider_parallelism,
        runtime_options=runtime_options,
    )


_QUALIFIED_STAGE_PARTS = 3


def _snowflake_runtime_controls(
    runtime: RuntimeMode,
    profile: Path,
    backend: ControlPlaneBackend | None = None,
) -> dict[str, str]:
    """Collect the SPCS objects required to turn a queued run into a job service.

    Every one of these names an object that must already exist in the account, so
    each is offered as a choice rather than typed. A mistyped pool or tag fails
    only once the service has been created and immediately dies.
    """

    if runtime != RuntimeMode.SNOWFLAKE:
        return {}
    defaults = _snowflake_defaults(profile)
    # Deployed in Snowflake the profile carries no namespace, so the session's own
    # database and schema are what qualify the stage.
    for key, value in account_context(runtime.value, _backend=backend).items():
        defaults.setdefault(key, value)
    st.markdown("**Snowpark Container Services**")
    object_columns = st.columns(2)
    compute_pool = _discovered_field(
        object_columns[0],
        "Compute pool",
        defaults.get("compute_pool", ""),
        _options(backend, runtime, "list_compute_pools"),
        "Compute pool that runs the FlakeGraph worker.",
    )
    image_repository = _discovered_field(
        object_columns[1],
        "Image repository",
        defaults.get("image_repository", ""),
        _options(backend, runtime, "list_image_repositories"),
        "Snowflake image repository containing the FlakeGraph image.",
    )
    image_columns = st.columns(2)
    image_name = _discovered_field(
        image_columns[0],
        "Image name",
        defaults.get("image_name", ""),
        (
            (lambda: list_images(image_repository))
            if (list_images := getattr(backend, "list_images", None)) and image_repository
            else None
        ),
        "Image and tag to execute, chosen from what has been pushed.",
    )
    image_digest = image_columns[1].text_input(
        "Image digest",
        value=defaults.get("image_digest", ""),
        help="Optional sha256 digest that pins the exact image revision.",
    )
    service_spec_stage = _discovered_field(
        st,
        "Service specification stage",
        defaults.get("service_spec_stage", ""),
        _options(backend, runtime, "list_stages"),
        "Internal stage where the app uploads the generated SPCS job specification.",
        prefer=("KG_SERVICE_SPECS",),
    )
    required = {
        "Compute pool": compute_pool,
        "Image repository": image_repository,
        "Image name": image_name,
        "Service specification stage": service_spec_stage,
    }
    missing = [label for label, value in required.items() if not value.strip()]
    if missing:
        st.warning("Snowflake runtime requires: " + ", ".join(missing))
    options: dict[str, Any] = {
        "compute_pool": compute_pool.strip(),
        "image_repository": image_repository.strip(),
        "image_name": image_name.strip(),
        # The picker lists bare stage names, but SPCS requires the fully
        # qualified form. Qualifying here keeps the operator choosing a stage
        # rather than composing an identifier.
        "service_spec_stage": _qualified_stage(
            service_spec_stage, defaults.get("database", ""), defaults.get("schema", "")
        ),
    }
    if image_digest.strip():
        options["image_digest"] = image_digest.strip()
    # A job service has its node to itself and SPCS bills the node either way, so
    # the container is sized from the pool rather than from a fixed default that
    # fits the smallest machine anyone might use.
    capacity = getattr(backend, "compute_pool_capacity", None)
    if callable(capacity) and compute_pool.strip():
        with suppress(Exception):
            options.update(container_resources(capacity(compute_pool.strip())))
    return options


def _output_controls(
    runtime: RuntimeMode,
    repository_root: Path,
    graph_id: str,
    job_id: str,
    profile: Path,
    backend: ControlPlaneBackend | None = None,
) -> OutputDestination | None:
    """Render a runtime-independent durable graph destination.

    Kubernetes changes where work executes, not where the graph must live. A
    Snowflake destination therefore uses the same bulk writer for local and fleet
    runs, while the fleet resolves credentials from namespace Secrets.
    """

    default_kind = StorageKind.SNOWFLAKE if runtime == RuntimeMode.SNOWFLAKE else StorageKind.LOCAL
    available_kinds = (
        [StorageKind.SNOWFLAKE] if runtime == RuntimeMode.SNOWFLAKE else list(StorageKind)
    )
    kind = (
        st.segmented_control(
            "Output storage",
            available_kinds,
            default=default_kind,
            format_func=lambda value: value.value,
            key="output_storage_kind",
            help=(
                "Choose where finalized graph tables and run reports are published. "
                "This is independent of where processing executes."
            ),
        )
        or default_kind
    )
    workspace_path = repository_root / "out" / "app" / graph_id / job_id
    if kind == StorageKind.LOCAL:
        path = st.text_input(
            "Output directory",
            value=str(workspace_path),
            help="Writes portable Parquet and JSON graph artifacts to this directory.",
        )
        return OutputDestination(kind=kind, workspace_path=Path(path).expanduser())

    defaults = _snowflake_defaults(profile)
    # Deployed in Snowflake the session already knows the account, user and
    # current namespace. Retyping them invites errors in fields that cannot be
    # wrong, so they are read from the session and the rest becomes a choice
    # between things that exist.
    context = dict(account_context(runtime.value, _backend=backend))
    for key, value in context.items():
        defaults.setdefault(key, value)

    account, user, host = _identity_controls(context, defaults)

    namespace = st.columns(3)
    database = _discovered_field(
        namespace[0],
        "Database",
        defaults.get("database", ""),
        _options(backend, runtime, "list_databases"),
        "Existing database that will own the graph tables.",
    )
    schema = _discovered_field(
        namespace[1],
        "Schema",
        defaults.get("schema", ""),
        _options(backend, runtime, "list_schemas", database),
        "Existing schema in which FlakeGraph publishes canonical graph tables.",
    )
    warehouse = _discovered_field(
        namespace[2],
        "Warehouse",
        defaults.get("warehouse", ""),
        _options(backend, runtime, "list_warehouses"),
        "Warehouse used for bulk loading and graph-table merge operations.",
    )
    load = st.columns(2)
    role = _discovered_field(
        load[0],
        "Role",
        defaults.get("role", ""),
        _options(backend, runtime, "list_roles"),
        "Snowflake role with access to the target namespace and load stage.",
    )
    bulk_stage = _discovered_field(
        load[1],
        "Bulk-load stage",
        defaults.get("bulk_stage", ""),
        _options(backend, runtime, "list_stages"),
        "An existing internal stage used for temporary Parquet load files.",
        prefer=("KG_LOAD_STAGE", "KG_BULK_STAGE"),
    )
    credential_name: str | None = None
    credential_field: str | None = None
    authenticator: str | None = None
    if runtime == RuntimeMode.SNOWFLAKE:
        # Deployed in Snowflake there is nothing to choose: the app and its SPCS
        # worker both hold a Snowflake-managed OAuth token, so the run stores no
        # credential and no authentication control is offered. The effective
        # configuration the operator approves has to say exactly that.
        authenticator = "oauth"
        st.caption(
            "The deployed app and its SPCS worker use Snowflake-managed OAuth; "
            "no password or token is stored in the run."
        )
    else:
        authentication = st.selectbox(
            "Authentication",
            ["Password", "OAuth token", "External browser"],
            index=0,
            help="External-browser authentication is available only for local runs.",
        )
        if authentication == "Password":
            credential_name = secret_text_input(
                "Password environment variable",
                key=seeded_widget(
                    cast(_SessionState, st.session_state),
                    "snowflake_output_password_environment",
                    "KG_SNOWFLAKE_PASSWORD",
                ),
                help_text="Enter the variable name, not the password value.",
            )
            credential_field = "password"
        elif authentication == "OAuth token":
            credential_name = secret_text_input(
                "OAuth token environment variable",
                key=seeded_widget(
                    cast(_SessionState, st.session_state),
                    "snowflake_output_oauth_environment",
                    "KG_SNOWFLAKE_OAUTH_TOKEN",
                ),
                help_text="Enter the variable name, not the token value.",
            )
            credential_field = "oauth_token"
            authenticator = "oauth"
        else:
            authenticator = "externalbrowser"
            if runtime == RuntimeMode.KUBERNETES:
                st.error("External-browser authentication cannot run inside Kubernetes workers.")
                return None

    required = {
        "Database": database,
        "Schema": schema,
        "Warehouse": warehouse,
        "Bulk-load stage": bulk_stage,
    }
    if runtime != RuntimeMode.SNOWFLAKE:
        required = {"Account": account, "User": user, **required}
    missing = [label for label, value in required.items() if not value.strip()]
    if missing:
        st.info(f"Complete the Snowflake destination to continue: {', '.join(missing)}")
        return None
    return OutputDestination(
        kind=kind,
        workspace_path=workspace_path,
        snowflake=SnowflakeOutput(
            account=account.strip() or "active-session",
            host=host.strip() or None,
            user=user.strip() or "active-session",
            authenticator=authenticator,
            database=database.strip(),
            schema=schema.strip(),
            role=role.strip() or None,
            warehouse=warehouse.strip(),
            bulk_stage=_qualified_stage(bulk_stage, database, schema),
            credential_environment_variable=credential_name.strip() if credential_name else None,
            credential_field=credential_field,
        ),
    )


def _snowflake_defaults(profile: Path) -> dict[str, str]:
    """Read non-secret Snowflake coordinates from the selected base profile."""

    config = load_base_config(profile) if profile.is_file() else {}
    raw = config.get("snowflake")
    if not isinstance(raw, Mapping):
        return {}
    allowed = {
        "account",
        "host",
        "user",
        "database",
        "schema",
        "role",
        "warehouse",
        "bulk_stage",
        "compute_pool",
        "image_repository",
        "image_name",
        "image_digest",
        "service_spec_stage",
    }
    return {key: str(raw[key]) for key in allowed if raw.get(key)}


def _source_controls(  # noqa: PLR0912, PLR0915 - each branch is one source-specific form.
    backend: ControlPlaneBackend,
    runtime: RuntimeMode,
    repository_root: Path,
    job_id: str,
) -> tuple[SourceKind, Mapping[str, Any] | None]:
    """Render only source families supported by the selected backend."""

    available = [SourceKind.UPLOAD]
    if "local" in backend.capabilities:
        available.append(SourceKind.LOCAL)
    if "azure_blob" in backend.capabilities:
        available.append(SourceKind.AZURE_BLOB)
    if "s3" in backend.capabilities:
        available.append(SourceKind.S3)
    if "snowflake_stage" in backend.capabilities:
        available.append(SourceKind.SNOWFLAKE_STAGE)
    source_kind = (
        st.segmented_control(
            "Input source",
            available,
            default=available[0],
            format_func=lambda value: value.value,
            help=(
                "Choose where source documents are discovered. Processing and output "
                "destinations are configured independently."
            ),
        )
        or available[0]
    )
    source: dict[str, Any]
    if source_kind == SourceKind.UPLOAD:
        uploads = st.file_uploader(
            "Documents",
            accept_multiple_files=True,
            type=sorted(suffix.lstrip(".") for suffix in SUPPORTED_SUFFIXES),
            help="Upload one or more supported documents directly into this run workspace.",
        )
        if not uploads:
            st.info("Add one or more documents to continue.")
            return source_kind, None
        upload_root = repository_root / ".flakegraph" / "app" / "uploads" / job_id
        try:
            paths = _persist_uploads(uploads, upload_root)
        except ValueError as exc:
            st.error(str(exc))
            return source_kind, None
        if runtime == RuntimeMode.SNOWFLAKE:
            stage_columns = st.columns(2)
            # Chosen from what exists rather than typed. The previous default
            # named a stage this schema never had, and Snowflake reports an
            # upload to a missing stage as a SystemError from its storage client
            # that names neither the stage nor the problem.
            stages = list((_options(backend, runtime, "list_stages") or list)())
            if stages:
                preferred = "KG_DOCS" if "KG_DOCS" in stages else stages[0]
                stage = stage_columns[0].selectbox(
                    "Internal stage",
                    stages,
                    index=stages.index(preferred),
                    help="Snowflake internal stage that receives the uploaded documents.",
                )
            else:
                stage = stage_columns[0].text_input(
                    "Internal stage",
                    value="KG_DOCS",
                    help="Snowflake internal stage that receives the uploaded documents.",
                )
            # Derived from the viewer rather than typed. The app lists a stage
            # with the owner's privileges, so a prefix anyone may choose lets one
            # viewer enumerate and ingest another's uploaded documents.
            prefix = viewer_stage_prefix(current_viewer(), job_id)
            stage_columns[1].text_input(
                "Stage prefix",
                value=prefix,
                disabled=True,
                help=(
                    "Folder within the internal stage. It is derived from your "
                    "Snowflake identity so uploads stay separated per person."
                ),
            )
            upload_state_key = f"stage_upload_{job_id}"
            upload_identity = (_upload_signature(uploads), stage, prefix)
            if st.button(
                "Upload to stage",
                help="Upload the selected files to the configured Snowflake internal stage.",
            ):
                upload_backend = cast(_StageUploadBackend, backend)
                with st.spinner("Uploading documents to Snowflake"):
                    try:
                        uploaded = upload_backend.upload_files(paths, stage, prefix)
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.session_state[upload_state_key] = {
                            "identity": upload_identity,
                            "count": uploaded,
                        }
            upload_state = st.session_state.get(upload_state_key)
            if not isinstance(upload_state, Mapping) or (
                upload_state.get("identity") != upload_identity or not upload_state.get("count")
            ):
                st.info("Upload the selected files to the internal stage before preflight.")
                return source_kind, None
            # The worker resolves this stage from inside the container, where a
            # bare name has no namespace to resolve against and is rejected as an
            # unsafe location.
            namespace = dict(account_context(runtime.value, _backend=backend))
            return SourceKind.SNOWFLAKE_STAGE, {
                "stage": _qualified_stage(
                    stage, namespace.get("database", ""), namespace.get("schema", "")
                ),
                "prefix": prefix,
            }
        return source_kind, {"path": str(upload_root)}
    if source_kind == SourceKind.LOCAL:
        default_input_path = os.environ.get(
            "FLAKEGRAPH_APP_INPUT_PATH",
            str(repository_root / "data" / "martial_arts" / "files"),
        )
        path = st.text_input(
            "File or directory",
            value=default_input_path,
            help=("Defaults to FLAKEGRAPH_APP_INPUT_PATH when that environment variable is set."),
        )
        source = {"kind": "local", "path": path}
    elif source_kind == SourceKind.AZURE_BLOB:
        columns = st.columns(2)
        account_url = columns[0].text_input(
            "Account URL",
            placeholder="https://account.blob.core.windows.net",
            help="Azure Storage account URL containing the source container.",
        )
        container = columns[1].text_input(
            "Container",
            help="Blob container to scan for supported documents.",
        )
        prefix = st.text_input(
            "Prefix",
            value="",
            help="Optional virtual folder prefix used to limit document discovery.",
        )
        source = {
            "kind": "azure_blob",
            "account_url": account_url,
            "container": container,
            "prefix": prefix,
        }
    elif source_kind == SourceKind.S3:
        columns = st.columns(2)
        bucket = columns[0].text_input(
            "Bucket",
            help="S3 or S3-compatible bucket containing source documents.",
        )
        prefix = columns[1].text_input(
            "Prefix",
            value="",
            help="Optional object-key prefix used to limit document discovery.",
        )
        endpoint = st.text_input(
            "Endpoint",
            placeholder="Optional for MinIO or another S3-compatible store",
            help="Leave empty for AWS S3; set for MinIO or another compatible service.",
        )
        region = st.text_input(
            "Region",
            value="",
            help="Optional cloud region used to address the bucket.",
        )
        source = {
            "kind": "s3",
            "bucket": bucket,
            "prefix": prefix,
            "endpoint_url": endpoint,
            "region": region,
        }
    else:
        columns = st.columns(2)
        stage = columns[0].text_input(
            "Stage",
            value="KG_INPUT_STAGE",
            help="Snowflake stage containing source documents.",
        )
        prefix = columns[1].text_input(
            "Prefix",
            value="",
            help="Optional folder prefix inside the Snowflake stage.",
        )
        # The worker resolves this stage from inside the container, where a bare
        # name has no namespace to resolve against and is rejected as an unsafe
        # location. Qualifying it here lets an operator name the stage they see
        # in Snowsight rather than compose an identifier.
        namespace = dict(account_context(runtime.value, _backend=backend))
        source = {
            "kind": "snowflake_stage",
            "stage": _qualified_stage(
                stage, namespace.get("database", ""), namespace.get("schema", "")
            ),
            "prefix": prefix,
        }

    source_identity = tuple(sorted((str(key), str(value)) for key, value in source.items()))
    source_cache_key = f"source_objects_{runtime.value.lower()}"
    if st.button(
        "List source",
        icon=":material/refresh:",
        help=(
            "List supported documents at this source before preflight. This reads "
            "object metadata only and does not begin processing."
        ),
    ):
        try:
            with st.spinner("Listing source objects"):
                st.session_state[source_cache_key] = {
                    "identity": source_identity,
                    "objects": backend.list_source_objects(source),
                }
        except Exception as exc:
            st.error(str(exc))
    cached_source = st.session_state.get(source_cache_key)
    objects = (
        cached_source.get("objects", [])
        if isinstance(cached_source, Mapping) and cached_source.get("identity") == source_identity
        else []
    )
    if objects:
        st.caption(f"{len(objects):,} supported objects shown")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Name": item.name,
                        "Size": format_bytes(item.size_bytes),
                        "Modified": item.modified_at,
                    }
                    for item in objects
                ]
            ),
            width="stretch",
            hide_index=True,
            height=min(310, 38 + len(objects) * 35),
        )
    normalized = dict(source)
    normalized.pop("kind", None)
    if source_kind == SourceKind.LOCAL:
        normalized = {"path": source["path"]}
    return source_kind, normalized


def _save_uploads(uploads: list[Any], destination: Path) -> list[Path]:
    """Store uploads under an app-owned directory using sanitized base names."""

    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    seen: set[str] = set()
    for upload in uploads:
        name = Path(upload.name).name
        if name in seen:
            raise ValueError(f"Duplicate uploaded file name: {name}")
        seen.add(name)
        path = destination / name
        path.write_bytes(upload.getvalue())
        paths.append(path)
    return paths


def _persist_uploads(uploads: list[Any], destination: Path) -> list[Path]:
    """Write a Streamlit upload selection once and reuse its durable paths.

    Uploaded files remain stable across widget reruns. Without this manifest,
    changing an OCR or LLM field rewrote every uploaded byte even though the source
    selection had not changed. Streamlit's upload ``file_id`` distinguishes a true
    replacement from an unchanged object with the same name and size.
    """

    signature = _upload_signature(uploads)
    cache_key = f"persisted_uploads_{destination.expanduser().resolve()}"
    paths = [destination / Path(upload.name).name for upload in uploads]
    if st.session_state.get(cache_key) == signature and all(path.is_file() for path in paths):
        return paths
    paths = _save_uploads(uploads, destination)
    st.session_state[cache_key] = signature
    return paths


def _upload_signature(uploads: list[Any]) -> tuple[tuple[str, int, str], ...]:
    """Return a non-secret identity for one ordered Streamlit upload selection."""

    return tuple(
        (
            Path(upload.name).name,
            int(getattr(upload, "size", 0) or 0),
            str(getattr(upload, "file_id", "") or ""),
        )
        for upload in uploads
    )


def _provider_defaults(runtime: RuntimeMode, profile: Path | None = None) -> dict[str, str]:
    """Choose defaults that match the active runtime's reviewed base profile."""

    if runtime == RuntimeMode.SNOWFLAKE:
        return {
            "ocr": "snowflake_cortex",
            "llm": "snowflake_cortex",
            "embedding": "snowflake_cortex",
        }
    if runtime == RuntimeMode.KUBERNETES:
        configured = load_base_config(profile) if profile and profile.is_file() else {}
        return {
            "ocr": _configured_provider(configured, "ocr", "fallback"),
            "llm": _configured_provider(configured, "llm", "vllm_local"),
            "embedding": _configured_provider(
                configured,
                "embedding",
                "sentence_transformers",
            ),
        }
    return {"ocr": "fallback", "llm": "ollama", "embedding": "sentence_transformers"}


def _configured_provider(
    config: Mapping[str, Any],
    section: str,
    fallback: str,
) -> str:
    """Read one provider name from a base profile with a stable fallback."""

    value = config.get(section)
    if not isinstance(value, Mapping) or not value.get("provider"):
        return fallback
    return str(value["provider"])


def _default_profile(runtime: RuntimeMode, root: Path) -> Path:
    """Return the domain-neutral quality profile shared by every app runtime.

    The ingestion form overlays runtime-specific providers and destinations, so
    selecting a runtime must never import a benchmark graph ID, source path,
    ontology, or relaxed quality gate into an unrelated user corpus.
    """

    _ = runtime
    return root / "configs" / "app-defaults.yaml"


def _safe_id(value: str) -> bool:
    """Keep identifiers valid as file names, run keys, and operator labels."""

    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value))


def _default_run_id(
    now: datetime | None = None,
    entropy: str | None = None,
) -> str:
    """Return a readable run ID with enough entropy for concurrent sessions.

    Wall-clock seconds are useful to operators but are not a uniqueness boundary.
    A 64-bit random suffix prevents two sessions or reruns in the same second from
    targeting the same durable run directory or Snowflake job identifier.
    """

    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = entropy or secrets.token_hex(8)
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", suffix):
        raise ValueError("Run ID entropy must contain 8-64 URL-safe characters")
    return f"graph-{timestamp}-{suffix}"


def _release_submitted_identity(state: _SessionState) -> None:
    """Free the run and graph identity fields once a run owns them.

    A submitted identity belongs to that run. Leaving it in the form makes the
    next graph re-submit against the same job, whose durable record is inserted
    only when it does not already exist: the earlier status and configuration
    survive while new files join its queue and a second worker starts against
    settings nobody reviewed. Clearing the widget keys restores the generated
    default for the next run.
    """

    for key in ("ingest_job_id", "ingest_graph_id"):
        with suppress(KeyError):
            del state[key]


def _record_preflight(
    state: _SessionState,
    fingerprint: str,
    result: Mapping[str, object],
) -> None:
    """Store a preflight result together with the exact request it reviewed."""

    state["preflight"] = {
        "fingerprint": fingerprint,
        "result": dict(result),
    }


def _approved_preflight(
    state: _SessionState,
    fingerprint: str,
) -> Mapping[str, object] | None:
    """Return a passing result only when it belongs to the current request."""

    result = _preflight_for_request(state, fingerprint)
    return result if result is not None and result.get("ok") is True else None


def _preflight_for_request(
    state: _SessionState,
    fingerprint: str,
) -> Mapping[str, object] | None:
    """Return any preflight result bound to the current request for display."""

    stored = state.get("preflight")
    if not isinstance(stored, Mapping) or stored.get("fingerprint") != fingerprint:
        return None
    result = stored.get("result")
    return cast(Mapping[str, object], result) if isinstance(result, Mapping) else None


def _synchronize_runtime_provider_state(
    state: _SessionState,
    runtime: RuntimeMode,
    defaults: Mapping[str, str],
    *,
    embedding_width: int | None = None,
) -> None:
    """Reset provider widgets when the execution runtime changes.

    Provider state is useful across ordinary reruns in one runtime. It is not
    portable across runtime boundaries, where a local endpoint or credential
    reference could otherwise remain selected for Snowflake Cortex.
    """

    runtime_key = runtime.value.lower()
    previous = state.get("_ingestion_provider_runtime")
    state["_ingestion_provider_runtime"] = runtime_key
    _adopt_stored_embedding_width(state, embedding_width)
    if previous is None or previous == runtime_key:
        return
    for kind in ("ocr", "llm", "embedding"):
        for suffix in ("model", "endpoint", "credential", "dimension"):
            with suppress(KeyError):
                del state[f"{kind}_{suffix}"]
        with suppress(KeyError):
            del state[f"_{kind}_fields_provider"]
        # Removed rather than reassigned: the selector's own default is already
        # this runtime's adapter, and writing the same value into the widget's
        # key is what Streamlit reports as a default that disagrees with state.
        with suppress(KeyError):
            del state[f"{kind}_provider"]


def _adopt_stored_embedding_width(state: _SessionState, width: int | None) -> None:
    """Move a stale embedding selection onto the width the destination stores.

    A model whose vectors the graph tables cannot hold produces a run that
    preflight refuses and that no operator can submit, so a selection left over
    from another destination is corrected once rather than reported forever.
    """

    key = "_ingestion_embedding_width"
    if width is None or state.get(key) == width:
        return
    state[key] = width
    # The width is what the destination can store, so it applies whatever model
    # produces the vectors. Only the model default is specific to the adapters
    # this application knows the widths of.
    state["embedding_dimension"] = width
    model = cortex_embedding_model_for_width(width)
    chosen = str(state.get("embedding_model") or "")
    if model and (not chosen or chosen in CORTEX_EMBEDDING_MODELS_BY_WIDTH.values()):
        state["embedding_model"] = model


def _options(
    backend: ControlPlaneBackend | None,
    runtime: RuntimeMode,
    method: str,
    argument: str = "",
) -> Callable[[], list[str]] | None:
    """Return a discovery listing as a no-argument callable, read at most rarely.

    Selectboxes are rebuilt on every rerun, so reading these directly cost one
    Snowflake round trip per control per interaction and left the page blank in
    between. The values are account metadata, identical for every viewer under
    owner's rights, which is what makes the shared cache behind this correct.
    """

    if getattr(backend, method, None) is None:
        return None
    return lambda: account_options(runtime.value, method, argument, _backend=backend)


def _discovered_field(
    column: Any,
    label: str,
    current: str,
    lister: Any,
    help_text: str,
    prefer: Sequence[str] = (),
) -> str:
    """Offer a choice between values that exist, falling back to free text.

    Discovery depends on privileges the current role may not hold. An empty list
    must therefore leave the operator able to type a value rather than trapped
    behind a control with nothing in it.
    """

    options: list[str] = []
    if callable(lister):
        try:
            options = [str(item) for item in lister() if str(item)]
        except Exception:
            options = []
    if not options:
        return str(column.text_input(label, value=current, help=help_text))
    if current and current not in options:
        options = [current, *options]
    # Falling back to whichever option sorts first would quietly pick, say, the
    # app's own artifact stage to bulk-load a graph into.
    chosen = current or next((name for name in prefer if name in options), "")
    index = options.index(chosen) if chosen in options else 0
    return str(column.selectbox(label, options, index=index, help=help_text))


def _identity_controls(
    context: Mapping[str, str],
    defaults: Mapping[str, str],
) -> tuple[str, str, str]:
    """Read identity from the session where there is one, ask for it where there is not."""

    if context:
        # Deliberately not CURRENT_USER(): inside Streamlit in Snowflake that
        # names the account that owns the application object, so every viewer
        # would be told they are publishing as somebody else.
        viewer = current_viewer()
        who = viewer.user_name or context.get("user", "the current user")
        st.caption(
            f"Publishing as **{who}** on "
            f"**{context.get('account', 'this account')}** using Snowflake-managed "
            "OAuth. No credential is stored in the run."
        )
        return context.get("account", ""), context.get("user", ""), ""

    st.caption(
        "The graph is bulk-loaded into canonical Snowflake tables. Credentials stay in "
        "an environment variable locally or a Kubernetes Secret on a fleet."
    )
    identity = st.columns(3)
    return (
        identity[0].text_input(
            "Account",
            value=defaults.get("account", ""),
            help="Snowflake account identifier, such as organization-account.",
        ),
        identity[1].text_input(
            "User",
            value=defaults.get("user", ""),
            help="Snowflake principal used to publish the finalized graph.",
        ),
        identity[2].text_input(
            "Host",
            value=defaults.get("host", ""),
            help="Optional explicit Snowflake hostname; normally inferred from the account.",
        ),
    )


def _qualified_stage(stage: str, database: str, schema: str) -> str:
    """Return a stage as @DATABASE.SCHEMA.STAGE, qualifying a bare name.

    Pickers offer the bare names Snowflake reports, while SPCS and the bulk
    writer both require the qualified form. Composing it here means a wrong
    identifier cannot be typed at all.
    """

    normalized = stage.strip().lstrip("@")
    if not normalized:
        return ""
    if normalized.count(".") == _QUALIFIED_STAGE_PARTS - 1:
        return f"@{normalized}"
    if database and schema:
        return f"@{database}.{schema}.{normalized}"
    return f"@{normalized}"


def _fleet_models(backend: ControlPlaneBackend) -> dict[str, str]:
    """Read the LLM and embedding models the deployed fleet runs.

    Only the Kubernetes runtime has a fleet to read, and a backend that cannot
    answer is not an error: these values seed form defaults, and preflight
    remains the authority on whether a run actually matches its workers.
    """

    reader = getattr(backend, "fleet_profile", None)
    if reader is None:
        return {}
    try:
        profile = reader()
    except Exception:
        return {}
    models: dict[str, str] = {}
    for section in ("llm", "embedding"):
        value = profile.get(section) if isinstance(profile, Mapping) else None
        if isinstance(value, Mapping):
            model = value.get("model")
            if isinstance(model, str) and model.strip():
                models[section] = model
    return models
