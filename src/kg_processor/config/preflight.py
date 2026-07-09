"""Preflight checks for provider availability and account/object readiness.

The worker should fail before expensive OCR/LLM calls when credentials, paths,
Snowflake objects, dimensions, or provider binaries are not ready. These checks
mirror factory selection so configuration errors are caught early and plainly.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from collections.abc import Callable
from os import W_OK, access
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from kg_processor.adapters.files.common import is_supported_file
from kg_processor.adapters.files.manifest import manifest_candidate_paths
from kg_processor.adapters.snowflake import validate_stage_location
from kg_processor.config.provider_registry import ProviderKind, provider_names
from kg_processor.config.settings import Settings

_OCR_SUPPORTED_SUFFIXES = {
    "builtin_text": {
        ".txt",
        ".md",
        ".markdown",
        ".pdf",
        ".docx",
        ".html",
        ".htm",
        ".pptx",
        ".xlsx",
    },
    "tesseract_internal": {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    },
    "mineru_api": {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    },
    "mineru_internal": {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    },
}
_TESSERACT_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_SPCS_FILE_SOURCES = {"stage", "snowflake_stage"}
_SPCS_WRITERS = {"snowflake_direct", "snowflake_bulk"}
_SYSTEM_COMPUTE_POOL_PREFIX = "SYSTEM_COMPUTE_POOL"
_PageWindow = tuple[int | None, int | None]


class PreflightResult(BaseModel):
    """Collected preflight checks and errors for a settings object."""

    ok: bool
    checks: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def require(self, condition: bool, ok_message: str, error_message: str) -> None:
        """Record either a passing check message or an error."""

        if condition:
            self.checks.append(ok_message)
        else:
            self.errors.append(error_message)
            self.ok = False


def run_preflight(settings: Settings) -> PreflightResult:
    """Validate settings before provider construction or expensive processing."""

    result = PreflightResult(ok=True)

    _validate_supported_providers(settings, result)
    if settings.files.source == "local":
        result.require(
            settings.files.input_path.exists(),
            f"Input path exists: {settings.files.input_path}",
            f"Input path does not exist: {settings.files.input_path}",
        )
    if settings.files.source == "manifest":
        result.require(
            bool(settings.files.manifest_path and settings.files.manifest_path.is_file()),
            f"Manifest file exists: {settings.files.manifest_path}",
            f"Manifest file source requires an existing files.manifest_path: "
            f"{settings.files.manifest_path}",
        )
    _validate_azure_blob_file_source(settings, result)
    _validate_snowflake_provider_selection(settings, result)
    if settings.writer.provider == "local_artifacts":
        parent = _nearest_existing_parent(settings.writer.output_path)
        result.require(
            parent.is_dir() and access(parent, W_OK),
            f"Output path can be created below: {parent}",
            f"Output path cannot be created below: {parent}",
        )
    if settings.cache.provider == "local":
        parent = _nearest_existing_parent(settings.cache.path)
        result.require(
            parent.is_dir() and access(parent, W_OK),
            f"Cache path can be created below: {parent}",
            f"Cache path cannot be created below: {parent}",
        )
    _validate_local_ocr_provider_selection(settings, result)
    _validate_ocr_page_range(settings, result)
    _validate_local_ocr_file_types(settings, result)
    _validate_llm_provider_selection(settings, result)
    _validate_embedding_provider_selection(settings, result)
    if settings.runtime.runtime == "spcs":
        required = [
            settings.snowflake.account or settings.snowflake.host,
            settings.snowflake.database,
            settings.snowflake.schema_name,
            settings.snowflake.role,
            settings.snowflake.warehouse,
            settings.snowflake.compute_pool,
            settings.snowflake.image_repository,
            settings.snowflake.service_spec_stage,
        ]
        result.require(
            all(required),
            "Snowflake runtime fields are configured",
            "Snowflake runtime requires account/host, database, schema, role, warehouse, "
            "compute pool, image repository, and service spec stage",
        )
        if settings.snowflake.service_spec_stage:
            _require_valid_stage(settings.snowflake.service_spec_stage, result)
        _validate_spcs_compute_pool(settings, result)
        _validate_spcs_runtime_provider_selection(settings, result)
    if settings.job.use_lease:
        result.require(
            bool(settings.job.lease_owner),
            "Job lease owner is configured",
            "job.use_lease requires job.lease_owner",
        )
        _require_snowflake_target(settings, "job lease", result)
    if settings.job.use_file_queue:
        result.require(
            bool(settings.job.lease_owner),
            "Job file queue worker id is configured",
            "job.use_file_queue requires job.lease_owner",
        )
        result.require(
            settings.job.file_batch_size > 0,
            f"Job file queue batch size is configured: {settings.job.file_batch_size}",
            "job.use_file_queue requires a positive job.file_batch_size",
        )
        _require_snowflake_target(settings, "job file queue", result)
    _validate_no_secret_files(settings, result)
    return result


def _validate_supported_providers(settings: Settings, result: PreflightResult) -> None:
    selected: dict[ProviderKind, str] = {
        "file_source": settings.files.source,
        "ocr": settings.ocr.provider,
        "llm": settings.llm.provider,
        "embedding": settings.embedding.provider,
        "writer": settings.writer.provider,
        "cache": settings.cache.provider,
    }
    for kind, name in selected.items():
        supported = provider_names(kind)
        result.require(
            name in supported,
            f"{kind} provider is supported: {name}",
            f"Unsupported {kind} provider '{name}'. Supported providers: "
            f"{', '.join(sorted(supported))}",
        )


def _validate_azure_blob_file_source(settings: Settings, result: PreflightResult) -> None:
    if settings.files.source != "azure_blob":
        return
    result.require(
        bool(settings.azure_blob.container),
        "Azure Blob container is configured",
        "azure_blob file source requires azure_blob.container",
    )
    result.require(
        bool(
            settings.azure_blob.connection_string
            or (settings.azure_blob.account_url and settings.azure_blob.sas_token)
        ),
        "Azure Blob credentials are configured",
        "azure_blob file source requires connection_string or account_url+sas_token",
    )
    parent = _nearest_existing_parent(settings.azure_blob.download_path)
    result.require(
        parent.is_dir() and access(parent, W_OK),
        f"Azure Blob download path can be created below: {parent}",
        f"Azure Blob download path cannot be created below: {parent}",
    )


def _validate_local_ocr_provider_selection(settings: Settings, result: PreflightResult) -> None:
    if settings.ocr.provider == "mineru_internal":
        result.require(
            shutil.which(settings.ocr.mineru_command) is not None,
            f"MinerU command is available: {settings.ocr.mineru_command}",
            f"MinerU command is not available: {settings.ocr.mineru_command}",
        )
        if settings.ocr.model_cache_dir is not None:
            parent = _nearest_existing_parent(settings.ocr.model_cache_dir)
            result.require(
                parent.is_dir() and access(parent, W_OK),
                f"MinerU model/cache path can be created below: {parent}",
                f"MinerU model/cache path cannot be created below: {parent}",
            )
        if settings.ocr.mineru_api_url:
            result.require(
                _is_http_url(settings.ocr.mineru_api_url),
                "MinerU internal API endpoint is a valid HTTP URL",
                "mineru_internal api_url must be an http(s) URL",
            )
        if settings.ocr.mineru_server_url:
            result.require(
                _is_http_url(settings.ocr.mineru_server_url),
                "MinerU internal server URL is a valid HTTP URL",
                "mineru_internal server_url must be an http(s) URL",
            )
    if settings.ocr.provider == "mineru_api":
        result.require(
            bool(settings.ocr.mineru_api_url),
            "MinerU API endpoint is configured",
            "mineru_api OCR requires mineru_api_url",
        )
        if settings.ocr.mineru_api_url:
            result.require(
                _is_http_url(settings.ocr.mineru_api_url),
                "MinerU API endpoint is a valid HTTP URL",
                "mineru_api OCR endpoint must be an http(s) URL",
            )
    if settings.ocr.provider == "generic_http":
        result.require(
            bool(settings.generic_http_ocr.endpoint),
            "Generic HTTP OCR endpoint is configured",
            "generic_http OCR requires generic_http_ocr.endpoint",
        )
        if settings.generic_http_ocr.endpoint:
            result.require(
                _is_http_url(settings.generic_http_ocr.endpoint),
                "Generic HTTP OCR endpoint is a valid HTTP URL",
                "generic_http OCR endpoint must be an http(s) URL",
            )
    if settings.ocr.provider == "tesseract_internal":
        result.require(
            shutil.which(settings.ocr.tesseract_command) is not None,
            f"Tesseract command is available: {settings.ocr.tesseract_command}",
            f"Tesseract command is not available: {settings.ocr.tesseract_command}",
        )
        result.require(
            shutil.which(settings.ocr.tesseract_pdf_renderer_command) is not None,
            (
                "Tesseract PDF renderer command is available: "
                f"{settings.ocr.tesseract_pdf_renderer_command}"
            ),
            (
                "Tesseract PDF renderer command is not available: "
                f"{settings.ocr.tesseract_pdf_renderer_command}"
            ),
        )


def _validate_ocr_page_range(settings: Settings, result: PreflightResult) -> None:
    """Validate OCR page-window syntax before a provider starts parsing documents.

    Each OCR backend mirrors a different upstream API. Builtin text extraction
    and Tesseract expose one-based page numbers because users think in document
    pages. MinerU and Snowflake Cortex expose zero-based page ids because their
    APIs use index-style page filters. Keeping that distinction visible here
    prevents quiet off-by-one surprises in long container jobs.
    """

    page_range = settings.ocr.page_range
    if page_range is not None and not page_range.strip():
        page_range = None

    if settings.ocr.provider == "builtin_text" and page_range:
        _require_page_range(
            result,
            lambda: _parse_page_windows(
                page_range,
                provider="builtin_text",
                minimum=1,
                numbering_message="builtin_text page_range uses one-based page numbers",
                allow_multiple=True,
                allow_open_ranges=False,
            ),
            "builtin_text OCR page_range is valid",
        )
        return

    if settings.ocr.provider == "tesseract_internal" and page_range:
        windows = _require_page_range(
            result,
            lambda: _parse_page_windows(
                page_range,
                provider="tesseract_internal",
                minimum=1,
                numbering_message="tesseract_internal page_range uses one-based page numbers",
                allow_multiple=False,
                allow_open_ranges=True,
            ),
            "tesseract_internal OCR page_range is valid",
        )
        if windows is not None:
            _validate_tesseract_image_page_range(settings, windows, result)
        return

    if settings.ocr.provider == "mineru_internal":
        _validate_mineru_page_window(settings, result, page_range)
        return

    if settings.ocr.provider == "snowflake_cortex" and page_range:
        _require_page_range(
            result,
            lambda: _parse_page_windows(
                page_range,
                provider="snowflake_cortex",
                minimum=0,
                numbering_message="snowflake_cortex page_range uses zero-based page ids",
                allow_multiple=True,
                allow_open_ranges=True,
            ),
            "snowflake_cortex OCR page_range is valid",
        )


def _validate_mineru_page_window(
    settings: Settings,
    result: PreflightResult,
    page_range: str | None,
) -> None:
    start = settings.ocr.mineru_start_page_id
    end = settings.ocr.mineru_end_page_id

    # MinerU command-line options take precedence over the portable page_range
    # field, so validate the explicit API fields first and do not reinterpret
    # page_range when either of those ids is configured.
    if start is not None or end is not None:
        result.require(
            start is None or end is None or end >= start,
            "mineru_internal OCR explicit page window is valid",
            "Invalid mineru_internal page window: mineru_end_page_id is before "
            "mineru_start_page_id",
        )
        return

    if not page_range:
        return

    _require_page_range(
        result,
        lambda: _parse_page_windows(
            page_range,
            provider="mineru_internal",
            minimum=0,
            numbering_message="mineru_internal page_range uses zero-based page ids",
            allow_multiple=False,
            allow_open_ranges=True,
        ),
        "mineru_internal OCR page_range is valid",
    )


def _validate_tesseract_image_page_range(
    settings: Settings,
    windows: list[_PageWindow],
    result: PreflightResult,
) -> None:
    # A standalone image represents exactly one OCR page. Tesseract can page
    # slice PDFs, but an image filter that excludes page 1 would produce no
    # document content and should fail while config is still cheap to fix.
    candidates = _local_or_manifest_candidate_paths(settings)
    if not any(path.suffix.lower() in _TESSERACT_IMAGE_SUFFIXES for path in candidates):
        return
    includes_page_one = any(
        (start is None or start <= 1) and (end is None or end >= 1) for start, end in windows
    )
    result.require(
        includes_page_one,
        "tesseract_internal image OCR page_range includes page 1",
        "tesseract_internal image OCR only supports page 1; adjust ocr.page_range",
    )


def _require_page_range(
    result: PreflightResult,
    parser: Callable[[], list[_PageWindow]],
    ok_message: str,
) -> list[_PageWindow] | None:
    try:
        windows = parser()
    except ValueError as exc:
        result.errors.append(str(exc))
        result.ok = False
        return None
    result.checks.append(ok_message)
    return windows


def _parse_page_windows(
    page_range: str,
    *,
    provider: str,
    minimum: int,
    numbering_message: str,
    allow_multiple: bool,
    allow_open_ranges: bool,
) -> list[_PageWindow]:
    raw = page_range.strip()
    if not raw:
        return []
    if "," in raw and not allow_multiple:
        raise ValueError(
            f"Invalid {provider} page_range '{raw}': comma-separated ranges are not supported"
        )

    windows: list[_PageWindow] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"Invalid {provider} page_range '{raw}': empty range segment")
        if "-" not in token:
            page = _parse_page_index(token, provider, minimum, numbering_message)
            windows.append((page, page))
            continue

        start_raw, end_raw = token.split("-", 1)
        if not allow_open_ranges and (not start_raw.strip() or not end_raw.strip()):
            raise ValueError(
                f"Invalid {provider} page_range '{raw}': open-ended ranges are not supported"
            )
        if not start_raw.strip() and not end_raw.strip():
            raise ValueError(f"Invalid {provider} page_range '{raw}': empty range segment")
        start = (
            _parse_page_index(start_raw.strip(), provider, minimum, numbering_message)
            if start_raw.strip()
            else None
        )
        end = (
            _parse_page_index(end_raw.strip(), provider, minimum, numbering_message)
            if end_raw.strip()
            else None
        )
        if start is not None and end is not None and end < start:
            raise ValueError(
                f"Invalid {provider} page_range '{raw}': end page is before start page"
            )
        windows.append((start, end))
    return windows


def _parse_page_index(
    value: str,
    provider: str,
    minimum: int,
    numbering_message: str,
) -> int:
    try:
        page = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {provider} page number '{value}' in page_range. {numbering_message}."
        ) from exc
    if page < minimum:
        raise ValueError(
            f"Invalid {provider} page number '{value}' in page_range. {numbering_message}."
        )
    return page


def _validate_local_ocr_file_types(settings: Settings, result: PreflightResult) -> None:
    supported_suffixes = _OCR_SUPPORTED_SUFFIXES.get(settings.ocr.provider)
    if supported_suffixes is None or settings.files.source not in {"local", "manifest"}:
        return
    candidates = _local_or_manifest_candidate_paths(settings)
    if not candidates:
        return
    unsupported = sorted(
        {
            path.suffix.lower()
            for path in candidates
            if path.suffix.lower() not in supported_suffixes
        }
    )
    result.require(
        not unsupported,
        f"{settings.ocr.provider} OCR supports all configured local file types",
        (
            f"{settings.ocr.provider} OCR does not support configured file suffixes: "
            f"{', '.join(unsupported)}"
        ),
    )


def _local_or_manifest_candidate_paths(settings: Settings) -> list[Path]:
    if settings.files.source == "manifest" and settings.files.manifest_path:
        if not settings.files.manifest_path.is_file():
            return []
        return manifest_candidate_paths(settings.files.manifest_path, settings.files.include_globs)
    if settings.files.source != "local":
        return []
    input_path = settings.files.input_path
    if input_path.is_file():
        return [input_path] if is_supported_file(input_path) else []
    candidates: dict[Path, None] = {}
    for pattern in settings.files.include_globs:
        for path in input_path.glob(pattern):
            if path.is_file() and is_supported_file(path):
                candidates[path] = None
    return sorted(candidates)


def _validate_llm_provider_selection(settings: Settings, result: PreflightResult) -> None:
    if settings.llm.provider in {"openai_compatible", "azure_openai", "vllm_local"}:
        result.require(
            bool(settings.llm.model),
            f"{settings.llm.provider} LLM model is configured",
            f"{settings.llm.provider} LLM requires llm.model",
        )
    if settings.llm.provider in {"openai_compatible", "azure_openai"}:
        result.require(
            bool(settings.llm.endpoint and settings.llm.api_key),
            f"{settings.llm.provider} LLM endpoint and API key are configured",
            f"{settings.llm.provider} LLM requires endpoint and API key",
        )
    if settings.llm.provider == "azure_openai":
        result.require(
            bool(settings.llm.api_version),
            "azure_openai LLM API version is configured",
            "azure_openai LLM requires llm.api_version",
        )
    if settings.llm.provider == "vllm_local":
        result.require(
            bool(settings.llm.endpoint),
            "vllm_local LLM endpoint is configured",
            "vllm_local LLM requires endpoint",
        )
    if settings.llm.provider == "snowflake_cortex":
        result.require(
            bool(settings.llm.model),
            "Snowflake Cortex LLM model is configured",
            "snowflake_cortex LLM requires llm.model",
        )


def _validate_embedding_provider_selection(
    settings: Settings,
    result: PreflightResult,
) -> None:
    if settings.embedding.provider in {
        "openai_compatible",
        "azure_openai",
        "sentence_transformers",
        "snowflake_cortex",
    }:
        result.require(
            bool(settings.embedding.model),
            f"{settings.embedding.provider} embedding model is configured",
            f"{settings.embedding.provider} embedding requires embedding.model",
        )
    if settings.embedding.provider in {"openai_compatible", "azure_openai"}:
        result.require(
            bool(settings.embedding.endpoint and settings.embedding.api_key),
            f"{settings.embedding.provider} embedding endpoint and API key are configured",
            f"{settings.embedding.provider} embedding requires endpoint and API key",
        )
    if settings.embedding.provider == "azure_openai":
        result.require(
            bool(settings.embedding.api_version),
            "azure_openai embedding API version is configured",
            "azure_openai embedding requires embedding.api_version",
        )
    if settings.embedding.provider == "sentence_transformers":
        result.require(
            importlib.util.find_spec("sentence_transformers") is not None,
            "sentence_transformers package is installed",
            "sentence_transformers embeddings require the local-embeddings extra",
        )
        if _is_cuda_device(settings.embedding.device):
            result.require(
                _torch_cuda_device_available(settings.embedding.device or "cuda"),
                f"CUDA embedding device is available: {settings.embedding.device}",
                (
                    "sentence_transformers embedding device "
                    f"{settings.embedding.device} requires CUDA but CUDA is not available"
                ),
            )


def _validate_no_secret_files(settings: Settings, result: PreflightResult) -> None:
    paths: list[Path] = [settings.files.input_path, settings.writer.output_path]
    for path in paths:
        if ".env" in path.parts:
            result.errors.append(f"Runtime path must not point inside a secret file path: {path}")
            result.ok = False


def _nearest_existing_parent(path: Path) -> Path:
    current = path if path.exists() else path.parent
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_cuda_device(device: str | None) -> bool:
    if not device:
        return False
    return device.strip().lower() == "cuda" or device.strip().lower().startswith("cuda:")


def _torch_cuda_device_available(device: str) -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return False
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available) or not bool(is_available()):
        return False
    index = _cuda_device_index(device)
    if index is None:
        return True
    device_count = getattr(cuda, "device_count", None)
    return callable(device_count) and int(device_count()) > index


def _cuda_device_index(device: str) -> int | None:
    normalized = device.strip().lower()
    if normalized == "cuda":
        return None
    _prefix, _separator, suffix = normalized.partition(":")
    if not suffix.isdigit():
        return None
    return int(suffix)


def _validate_snowflake_provider_selection(
    settings: Settings,
    result: PreflightResult,
) -> None:
    if settings.files.source in {"stage", "snowflake_stage"}:
        result.require(
            bool(settings.snowflake.stage),
            "Snowflake stage is configured",
            "snowflake_stage file source requires snowflake.stage",
        )
        if settings.snowflake.stage:
            _require_valid_stage(settings.snowflake.stage, result)
        _require_snowflake_target(settings, "snowflake_stage file source", result)
    if settings.ocr.provider == "snowflake_cortex":
        result.require(
            settings.files.source in {"stage", "snowflake_stage"},
            "Snowflake Cortex OCR will read staged files",
            "snowflake_cortex OCR requires files.source=snowflake_stage",
        )
        _require_snowflake_target(settings, "snowflake_cortex OCR", result)
    if settings.llm.provider == "snowflake_cortex":
        _require_snowflake_target(settings, "snowflake_cortex LLM", result)
    if settings.embedding.provider == "snowflake_cortex":
        _require_snowflake_target(settings, "snowflake_cortex embedding", result)
    if settings.cache.provider == "snowflake":
        _require_snowflake_target(settings, "snowflake cache", result)
    if settings.writer.provider == "snowflake_direct":
        _require_snowflake_target(settings, "snowflake_direct writer", result)
    if settings.writer.provider == "snowflake_bulk":
        _require_snowflake_target(settings, "snowflake_bulk writer", result)
        result.require(
            bool(settings.snowflake.bulk_stage),
            "Snowflake bulk load stage is configured",
            "snowflake_bulk writer requires snowflake.bulk_stage",
        )
        if settings.snowflake.bulk_stage:
            _require_valid_stage(settings.snowflake.bulk_stage, result)


def _validate_spcs_runtime_provider_selection(
    settings: Settings,
    result: PreflightResult,
) -> None:
    """Keep SPCS jobs on the Snowflake-native provider set.

    The local/open-source profile is intentionally complete, but it should not
    leak into Snowpark Container Services. SPCS containers are ephemeral and run
    inside the Snowflake boundary, so staged input, Cortex providers, Snowflake
    cache tables, and Snowflake writers are the reviewable production contract.
    """

    result.require(
        settings.files.source in _SPCS_FILE_SOURCES,
        "SPCS runtime uses staged Snowflake input",
        "SPCS runtime requires files.source=snowflake_stage",
    )
    result.require(
        settings.ocr.provider == "snowflake_cortex",
        "SPCS runtime uses Snowflake Cortex OCR",
        "SPCS runtime requires ocr.provider=snowflake_cortex",
    )
    result.require(
        settings.llm.provider == "snowflake_cortex",
        "SPCS runtime uses Snowflake Cortex LLM",
        "SPCS runtime requires llm.provider=snowflake_cortex",
    )
    result.require(
        settings.embedding.provider == "snowflake_cortex",
        "SPCS runtime uses Snowflake Cortex embeddings",
        "SPCS runtime requires embedding.provider=snowflake_cortex",
    )
    result.require(
        settings.writer.provider in _SPCS_WRITERS,
        "SPCS runtime uses a Snowflake graph writer",
        "SPCS runtime requires writer.provider=snowflake_direct or snowflake_bulk",
    )
    result.require(
        settings.cache.provider == "snowflake",
        "SPCS runtime uses Snowflake cache tables",
        "SPCS runtime requires cache.provider=snowflake",
    )


def _validate_spcs_compute_pool(settings: Settings, result: PreflightResult) -> None:
    pool = settings.snowflake.compute_pool or ""
    result.require(
        not pool.upper().startswith(_SYSTEM_COMPUTE_POOL_PREFIX),
        "SPCS runtime uses a dedicated service-compatible compute pool",
        "SPCS runtime requires a dedicated compute pool; Snowflake SYSTEM_COMPUTE_POOL_* "
        "pools reject FlakeGraph EXECUTE JOB SERVICE workloads",
    )


def _require_snowflake_target(
    settings: Settings,
    label: str,
    result: PreflightResult,
) -> None:
    result.require(
        bool(
            (settings.snowflake.account or settings.snowflake.host)
            and settings.snowflake.database
            and settings.snowflake.schema_name
        ),
        f"{label} Snowflake target is configured",
        f"{label} requires account/host, database, and schema",
    )


def _require_valid_stage(stage: str, result: PreflightResult) -> None:
    try:
        validate_stage_location(stage)
    except ValueError as exc:
        result.errors.append(str(exc))
        result.ok = False
    else:
        result.checks.append(f"Snowflake stage location is valid: {stage}")
