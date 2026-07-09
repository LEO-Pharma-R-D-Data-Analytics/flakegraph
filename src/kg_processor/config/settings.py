"""Typed configuration model for the full local/on-prem/SPCS runtime.

Defaults intentionally describe the production local-open-source profile rather
than a test fallback. Smoke tests opt into fake/hash/builtin providers
explicitly, which keeps real runs from silently changing providers.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from kg_processor.config.provider_registry import ProviderKind, provider_names

_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")
_ENV_PLACEHOLDER_ONLY_RE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")
_MISSING_ENV_PLACEHOLDER = object()


class RuntimeSettings(BaseModel):
    """Selects the deployment runtime profile without changing provider contracts."""

    runtime: Literal["local", "onprem", "spcs"] = "local"


class JobSettings(BaseModel):
    """Controls job identity, leasing, and file queue batching for resumable runs."""

    job_id: str = "local-job"
    graph_id: str = "local-graph"
    use_lease: bool = False
    use_file_queue: bool = False
    lease_owner: str | None = None
    lease_seconds: int = 900
    file_batch_size: int = 100

    @field_validator("lease_seconds")
    @classmethod
    def lease_seconds_must_be_positive(cls, value: int) -> int:
        """Reject non-positive lease windows before a worker can stall the queue."""

        if value <= 0:
            raise ValueError("lease_seconds must be positive")
        return value

    @field_validator("file_batch_size")
    @classmethod
    def file_batch_size_must_be_positive(cls, value: int) -> int:
        """Keep file claim batches meaningful for both local and Snowflake queues."""

        if value <= 0:
            raise ValueError("file_batch_size must be positive")
        return value


class FileSettings(BaseModel):
    """Describes where source documents are discovered before OCR begins."""

    source: Literal["local", "manifest", "stage", "snowflake_stage", "azure_blob"] = "local"
    input_path: Path = Path("data/samples")
    manifest_path: Path | None = None
    stage_prefix: str | None = None
    include_globs: list[str] = Field(default_factory=lambda: ["**/*"])

    @field_validator("source", mode="before")
    @classmethod
    def source_must_be_registered(cls, value: object) -> object:
        """Validate file-source names against the shared provider catalog."""

        return _validate_provider_name(value, "file_source")


class AzureBlobSettings(BaseModel):
    """Configures optional Azure Blob input discovery and local download staging."""

    account_url: str | None = None
    connection_string: str | None = None
    container: str | None = None
    prefix: str | None = None
    sas_token: str | None = None
    download_path: Path = Path("out/azure-blob")


class OcrSettings(BaseModel):
    """Configures OCR provider selection plus provider-specific execution knobs."""

    provider: str = "mineru_internal"
    language: str | None = None
    page_range: str | None = None
    timeout_seconds: int = 900
    model_cache_dir: Path | None = None
    mineru_command: str = "mineru"
    mineru_method: Literal["auto", "txt", "ocr"] = "auto"
    mineru_backend: str | None = None
    mineru_effort: str | None = None
    mineru_api_url: str | None = None
    mineru_api_key: str | None = None
    mineru_server_url: str | None = None
    mineru_start_page_id: int | None = None
    mineru_end_page_id: int | None = None
    mineru_formula: bool | None = None
    mineru_table: bool | None = None
    mineru_image_analysis: bool | None = None
    mineru_client_side_output_generation: bool = False
    tesseract_command: str = "tesseract"
    tesseract_pdf_renderer_command: str = "pdftoppm"
    tesseract_dpi: int = 300
    snowflake_parse_mode: Literal["OCR", "LAYOUT"] = "OCR"
    snowflake_extract_images: bool = False
    snowflake_page_split: bool = True

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate OCR provider names before factory construction."""

        return _validate_provider_name(value, "ocr")

    @field_validator("mineru_start_page_id", "mineru_end_page_id")
    @classmethod
    def mineru_page_ids_must_be_non_negative(cls, value: int | None) -> int | None:
        """Keep MinerU page slicing compatible with its zero-based API contract."""

        if value is not None and value < 0:
            raise ValueError("MinerU page ids use zero-based indexes and must be non-negative")
        return value

    @field_validator("tesseract_dpi")
    @classmethod
    def tesseract_dpi_must_be_positive(cls, value: int) -> int:
        """Require a render DPI that can be passed safely to pdftoppm/Tesseract."""

        if value <= 0:
            raise ValueError("tesseract_dpi must be positive")
        return value

    @field_validator("snowflake_parse_mode", mode="before")
    @classmethod
    def snowflake_parse_mode_uppercase(cls, value: object) -> object:
        """Normalize Snowflake Cortex parse modes to the literals accepted by SQL."""

        return value.upper() if isinstance(value, str) else value


class GenericHttpOcrSettings(BaseModel):
    """Maps an arbitrary HTTP OCR response into the normalized document schema."""

    endpoint: str | None = None
    api_key: str | None = None
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    file_field: str = "file"
    result_path: str | None = None
    pages_path: str = "pages"
    page_number_path: str = "page_number"
    markdown_path: str = "markdown"
    raw_text_path: str = "raw_text"
    detected_language_path: str = "detected_language"
    blocks_path: str = "blocks"
    block_id_path: str = "id"
    block_kind_path: str = "kind"
    block_text_path: str = "text"
    block_bbox_path: str = "bbox"
    block_confidence_path: str | None = "confidence"
    block_metadata_path: str | None = "metadata"
    assets_path: str = "assets"
    asset_id_path: str = "id"
    asset_kind_path: str = "kind"
    asset_uri_path: str = "uri"
    asset_page_number_path: str = "page_number"
    asset_confidence_path: str | None = "confidence"
    asset_metadata_path: str | None = "metadata"
    warnings_path: str = "warnings"
    error_path: str = "error"
    status_path: str = "status"


class LlmSettings(BaseModel):
    """Selects the graph-extraction LLM endpoint and model."""

    provider: str = "openai_compatible"
    endpoint: str | None = None
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    api_version: str = "2025-01-01-preview"
    timeout_seconds: int = 120

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate LLM provider names before adapter construction."""

        return _validate_provider_name(value, "llm")


class EmbeddingSettings(BaseModel):
    """Selects the embedding backend used for chunks, nodes, edges, and communities."""

    provider: str = "sentence_transformers"
    endpoint: str | None = None
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_key: str | None = None
    api_version: str = "2025-01-01-preview"
    dimension: int = 384
    batch_size: int = 32
    device: str | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate embedding provider names before adapter construction."""

        return _validate_provider_name(value, "embedding")

    @field_validator("dimension")
    @classmethod
    def dimension_must_be_positive(cls, value: int) -> int:
        """Require a vector dimension that can produce valid table rows."""

        if value <= 0:
            raise ValueError("embedding dimension must be positive")
        return value

    @field_validator("batch_size")
    @classmethod
    def batch_size_must_be_positive(cls, value: int) -> int:
        """Require a positive provider batch size so embeddings always make progress."""

        if value <= 0:
            raise ValueError("embedding batch_size must be positive")
        return value


class GraphSettings(BaseModel):
    """Tuning parameters for chunking, extraction, filtering, merging, and quality gates."""

    chunk_token_size: int = 1000
    chunk_token_overlap: int = 200
    max_chunks_per_llm_call: int = 3
    max_entities_per_batch: int = 100
    max_relations_per_batch: int = 100
    gleaning_max_passes: int = 1
    gleaning_saturation_threshold: int = 10
    min_entity_confidence: float = 0.0
    min_relation_confidence: float = 0.0
    min_entity_name_length: int = 2
    require_relation_endpoint_grounding: bool = True
    relation_weight_max: float = 10.0
    min_community_size: int = 2
    description_merge_min_observations: int = 2
    description_merge_max_descriptions: int = 8
    description_merge_max_evidence: int = 5
    fail_on_quality_error: bool = True
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "PRODUCT",
            "EVENT",
            "CONCEPT",
            "DATE",
            "QUANTITY",
        ]
    )
    relation_types: list[str] | None = None
    entity_blocklist: list[str] = Field(
        default_factory=lambda: [
            "chapter",
            "section",
            "introduction",
            "conclusion",
            "paragraph",
            "document",
            "page",
        ]
    )

    @field_validator("chunk_token_size")
    @classmethod
    def chunk_token_size_must_be_positive(cls, value: int) -> int:
        """Require non-empty chunks before any LLM extraction is scheduled."""

        if value <= 0:
            raise ValueError("chunk_token_size must be positive")
        return value

    @field_validator("chunk_token_overlap")
    @classmethod
    def overlap_smaller_than_chunk(cls, value: int, info: Any) -> int:
        """Prevent overlap settings that would duplicate or block chunk advancement."""

        if value < 0:
            raise ValueError("chunk_token_overlap must be non-negative")
        chunk_size = info.data.get("chunk_token_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_token_overlap must be smaller than chunk_token_size")
        return value

    @field_validator(
        "max_chunks_per_llm_call",
        "max_entities_per_batch",
        "max_relations_per_batch",
        "gleaning_saturation_threshold",
        "min_entity_name_length",
        "min_community_size",
    )
    @classmethod
    def graph_positive_integer_limits(cls, value: int) -> int:
        """Validate graph limits that are used as loop bounds or SQL batch sizes."""

        if value <= 0:
            raise ValueError("graph integer limits must be positive")
        return value

    @field_validator("gleaning_max_passes")
    @classmethod
    def gleaning_max_passes_must_be_non_negative(cls, value: int) -> int:
        """Allow disabling gleaning with zero while rejecting impossible negatives."""

        if value < 0:
            raise ValueError("gleaning_max_passes must be non-negative")
        return value

    @field_validator("relation_weight_max")
    @classmethod
    def relation_weight_max_must_be_positive(cls, value: float) -> float:
        """Keep relation weights in a bounded positive scale for downstream ranking."""

        if value <= 0:
            raise ValueError("relation_weight_max must be positive")
        return value

    @field_validator("min_entity_confidence", "min_relation_confidence")
    @classmethod
    def confidence_thresholds_must_be_probability(cls, value: float) -> float:
        """Ensure confidence thresholds stay on the normalized 0..1 scale."""

        if value < 0.0 or value > 1.0:
            raise ValueError("confidence thresholds must be between 0 and 1")
        return value

    @field_validator(
        "description_merge_min_observations",
        "description_merge_max_descriptions",
        "description_merge_max_evidence",
    )
    @classmethod
    def description_merge_limits_must_be_positive(cls, value: int) -> int:
        """Require merge limits that still allow descriptions to be synthesized."""

        if value <= 0:
            raise ValueError("description merge limits must be positive")
        return value


class WriterSettings(BaseModel):
    """Selects where the assembled graph artifacts are persisted."""

    provider: str = "local_artifacts"
    output_path: Path = Path("out/kg")

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate writer provider names before adapter construction."""

        return _validate_provider_name(value, "writer")


class CacheSettings(BaseModel):
    """Controls optional OCR and extraction result caching."""

    provider: Literal["none", "local", "snowflake"] = "none"
    path: Path = Path("out/cache")

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate cache provider names before adapter construction."""

        return _validate_provider_name(value, "cache")


class SnowflakeSettings(BaseModel):
    """Holds Snowflake connection, stage, image, and SPCS deployment settings."""

    account: str | None = None
    host: str | None = None
    user: str | None = None
    password: str | None = None
    authenticator: str | None = None
    private_key_path: Path | None = None
    oauth_token: str | None = None
    oauth_token_path: Path | None = Path("/snowflake/session/token")
    database: str | None = None
    schema_name: str | None = Field(default=None, alias="schema")
    role: str | None = None
    warehouse: str | None = None
    stage: str | None = None
    bulk_stage: str | None = None
    image_repository: str | None = None
    image_name: str = "flakegraph:latest"
    image_digest: str | None = None
    compute_pool: str | None = None
    compute_pool_instance_family: str | None = None
    compute_pool_min_nodes: int = 1
    compute_pool_max_nodes: int = 1
    service_name: str = "KG_PROCESSOR_JOB"
    service_spec_stage: str | None = None
    service_cpu_request: str = "1"
    service_cpu_limit: str = "2"
    service_memory_request: str = "8Gi"
    service_memory_limit: str = "16Gi"
    service_gpu_count: int = 0
    bulk_target_file_size_mb: int = 128

    @field_validator(
        "compute_pool_min_nodes",
        "compute_pool_max_nodes",
        "bulk_target_file_size_mb",
    )
    @classmethod
    def snowflake_positive_integer_settings(cls, value: int) -> int:
        """Validate Snowflake sizing knobs that are emitted into SQL or service specs."""

        if value <= 0:
            raise ValueError("Snowflake positive integer settings must be positive")
        return value

    @field_validator("image_digest")
    @classmethod
    def image_digest_must_be_sha256(cls, value: str | None) -> str | None:
        """Accept only immutable OCI SHA-256 image digests when one is supplied."""

        if value is None:
            return None
        normalized = value.strip().lower()
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", normalized):
            raise ValueError("image_digest must be a sha256:<64 hex chars> OCI digest")
        return normalized

    @field_validator("service_gpu_count")
    @classmethod
    def service_gpu_count_must_be_non_negative(cls, value: int) -> int:
        """Allow CPU-only services while rejecting negative GPU requests."""

        if value < 0:
            raise ValueError("service_gpu_count must be non-negative")
        return value

    @model_validator(mode="after")
    def compute_pool_max_must_cover_min(self) -> SnowflakeSettings:
        """Ensure rendered compute pools can satisfy their configured minimum size."""

        if self.compute_pool_max_nodes < self.compute_pool_min_nodes:
            raise ValueError("compute_pool_max_nodes must be greater than or equal to min nodes")
        return self


class Settings(BaseModel):
    """Top-level configuration object consumed by factories, CLI commands, and workers."""

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    job: JobSettings = Field(default_factory=JobSettings)
    files: FileSettings = Field(default_factory=FileSettings)
    azure_blob: AzureBlobSettings = Field(default_factory=AzureBlobSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    generic_http_ocr: GenericHttpOcrSettings = Field(default_factory=GenericHttpOcrSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    writer: WriterSettings = Field(default_factory=WriterSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    snowflake: SnowflakeSettings = Field(default_factory=SnowflakeSettings)

    @classmethod
    def load(
        cls,
        config_file: Path | None = None,
        env: dict[str, str] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> Settings:
        """Load YAML, environment variables, and explicit overrides into settings."""

        env = env if env is not None else dict(os.environ)
        selected_file = config_file or (
            Path(env["KG_CONFIG_FILE"]) if env.get("KG_CONFIG_FILE") else None
        )
        data: dict[str, Any] = {}
        if selected_file:
            data = _deep_update(data, _read_yaml(selected_file, env))
        data = _deep_update(data, _from_env(env))
        if overrides:
            data = _deep_update(data, overrides)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Invalid FlakeGraph configuration: {exc}") from exc


def _read_yaml(path: Path, env: dict[str, str]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    interpolated = _interpolate_env(loaded, env)
    if not isinstance(interpolated, dict):
        raise ValueError(f"Configuration file must contain a mapping after interpolation: {path}")
    return interpolated


def _interpolate_env(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            interpolated = _interpolate_env(item, env)
            if interpolated is _MISSING_ENV_PLACEHOLDER:
                continue
            result[key] = interpolated
        return result
    if isinstance(value, list):
        list_result: list[Any] = []
        for item in value:
            interpolated = _interpolate_env(item, env)
            if interpolated is _MISSING_ENV_PLACEHOLDER:
                continue
            list_result.append(interpolated)
        return list_result
    if isinstance(value, str):
        whole_placeholder = _ENV_PLACEHOLDER_ONLY_RE.fullmatch(value)
        if whole_placeholder and not env.get(whole_placeholder.group(1)):
            return _MISSING_ENV_PLACEHOLDER
        return _ENV_PLACEHOLDER_RE.sub(lambda match: env.get(match.group(1), ""), value)
    return value


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _from_env(env: dict[str, str]) -> dict[str, Any]:
    # Keep env parsing as data rather than scattered conditionals. This makes
    # redaction, config printing, preflight, and tests reason over the same
    # supported environment-variable surface.
    data = _ai_backend_profile_from_env(env)
    alias_mapping: dict[str, tuple[str, str, Any]] = {
        # These names come from the original container-image spec and from
        # Snowpark Container Services itself. They are parsed before the
        # KG_SNOWFLAKE_* names so the FlakeGraph-specific env surface can
        # deliberately override ambient platform values during tests or jobs.
        "SNOWFLAKE_ACCOUNT": ("snowflake", "account", str),
        "SNOWFLAKE_HOST": ("snowflake", "host", str),
        "SNOWFLAKE_USER": ("snowflake", "user", str),
        "SNOWFLAKE_PASSWORD": ("snowflake", "password", str),
        "SNOWFLAKE_AUTH": ("snowflake", "authenticator", _snowflake_auth_alias),
        "SNOWFLAKE_AUTHENTICATOR": ("snowflake", "authenticator", _snowflake_auth_alias),
        "SNOWFLAKE_PRIVATE_KEY_PATH": ("snowflake", "private_key_path", Path),
        "SNOWFLAKE_OAUTH_TOKEN": ("snowflake", "oauth_token", str),
        "SNOWFLAKE_OAUTH_TOKEN_PATH": ("snowflake", "oauth_token_path", Path),
        "SNOWFLAKE_DATABASE": ("snowflake", "database", str),
        "SNOWFLAKE_SCHEMA": ("snowflake", "schema", str),
        "SNOWFLAKE_ROLE": ("snowflake", "role", str),
        "SNOWFLAKE_WAREHOUSE": ("snowflake", "warehouse", str),
        "KG_STAGE": ("snowflake", "stage", str),
        "KG_BULK_STAGE": ("snowflake", "bulk_stage", str),
        "KG_BLOB_ACCOUNT_URL": ("azure_blob", "account_url", str),
        "KG_BLOB_CONNECTION_STRING": ("azure_blob", "connection_string", str),
        "KG_BLOB_CONTAINER": ("azure_blob", "container", str),
        "KG_BLOB_PREFIX": ("azure_blob", "prefix", str),
        "KG_BLOB_SAS_TOKEN": ("azure_blob", "sas_token", str),
        "KG_BLOB_DOWNLOAD_PATH": ("azure_blob", "download_path", Path),
        "KG_OCR_ENGINE": ("ocr", "provider", _ocr_provider_alias),
    }
    mapping: dict[str, tuple[str, str, Any]] = {
        "KG_RUNTIME": ("runtime", "runtime", str),
        "KG_JOB_ID": ("job", "job_id", str),
        "KG_GRAPH_ID": ("job", "graph_id", str),
        "KG_JOB_USE_LEASE": ("job", "use_lease", _str_to_bool),
        "KG_JOB_USE_FILE_QUEUE": ("job", "use_file_queue", _str_to_bool),
        "KG_WORKER_ID": ("job", "lease_owner", str),
        "KG_JOB_LEASE_OWNER": ("job", "lease_owner", str),
        "KG_JOB_LEASE_SECONDS": ("job", "lease_seconds", int),
        "KG_LEASE_MINUTES": ("job", "lease_seconds", _minutes_to_seconds),
        "KG_BATCH_FILES": ("job", "file_batch_size", int),
        "KG_FILE_BATCH_SIZE": ("job", "file_batch_size", int),
        "KG_FILE_SOURCE": ("files", "source", _file_source_alias),
        "KG_INPUT_PATH": ("files", "input_path", Path),
        "KG_MANIFEST_PATH": ("files", "manifest_path", Path),
        "KG_STAGE_PREFIX": ("files", "stage_prefix", str),
        "KG_AZURE_BLOB_ACCOUNT_URL": ("azure_blob", "account_url", str),
        "KG_AZURE_BLOB_CONNECTION_STRING": ("azure_blob", "connection_string", str),
        "KG_AZURE_BLOB_CONTAINER": ("azure_blob", "container", str),
        "KG_AZURE_BLOB_PREFIX": ("azure_blob", "prefix", str),
        "KG_AZURE_BLOB_SAS_TOKEN": ("azure_blob", "sas_token", str),
        "KG_AZURE_BLOB_DOWNLOAD_PATH": ("azure_blob", "download_path", Path),
        "KG_OCR_PROVIDER": ("ocr", "provider", str),
        "KG_OCR_LANGUAGE": ("ocr", "language", str),
        "KG_OCR_PAGE_RANGE": ("ocr", "page_range", str),
        "KG_OCR_MODEL_CACHE_DIR": ("ocr", "model_cache_dir", Path),
        "KG_MINERU_COMMAND": ("ocr", "mineru_command", str),
        "KG_MINERU_METHOD": ("ocr", "mineru_method", str),
        "KG_MINERU_BACKEND": ("ocr", "mineru_backend", str),
        "KG_MINERU_EFFORT": ("ocr", "mineru_effort", str),
        "KG_MINERU_API_URL": ("ocr", "mineru_api_url", str),
        "KG_MINERU_API_KEY": ("ocr", "mineru_api_key", str),
        "KG_MINERU_SERVER_URL": ("ocr", "mineru_server_url", str),
        "KG_MINERU_START_PAGE_ID": ("ocr", "mineru_start_page_id", int),
        "KG_MINERU_END_PAGE_ID": ("ocr", "mineru_end_page_id", int),
        "KG_MINERU_FORMULA": ("ocr", "mineru_formula", _str_to_bool),
        "KG_MINERU_TABLE": ("ocr", "mineru_table", _str_to_bool),
        "KG_MINERU_IMAGE_ANALYSIS": ("ocr", "mineru_image_analysis", _str_to_bool),
        "KG_MINERU_CLIENT_SIDE_OUTPUT": (
            "ocr",
            "mineru_client_side_output_generation",
            _str_to_bool,
        ),
        "KG_TESSERACT_COMMAND": ("ocr", "tesseract_command", str),
        "KG_TESSERACT_PDF_RENDERER_COMMAND": (
            "ocr",
            "tesseract_pdf_renderer_command",
            str,
        ),
        "KG_TESSERACT_DPI": ("ocr", "tesseract_dpi", int),
        "KG_SNOWFLAKE_PARSE_MODE": ("ocr", "snowflake_parse_mode", str),
        "KG_SNOWFLAKE_EXTRACT_IMAGES": ("ocr", "snowflake_extract_images", _str_to_bool),
        "KG_SNOWFLAKE_PAGE_SPLIT": ("ocr", "snowflake_page_split", _str_to_bool),
        "KG_GENERIC_HTTP_OCR_ENDPOINT": ("generic_http_ocr", "endpoint", str),
        "KG_GENERIC_HTTP_OCR_API_KEY": ("generic_http_ocr", "api_key", str),
        "KG_GENERIC_HTTP_OCR_API_KEY_HEADER": (
            "generic_http_ocr",
            "api_key_header",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_API_KEY_PREFIX": (
            "generic_http_ocr",
            "api_key_prefix",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_FILE_FIELD": ("generic_http_ocr", "file_field", str),
        "KG_GENERIC_HTTP_OCR_RESULT_PATH": ("generic_http_ocr", "result_path", str),
        "KG_GENERIC_HTTP_OCR_PAGES_PATH": ("generic_http_ocr", "pages_path", str),
        "KG_GENERIC_HTTP_OCR_PAGE_NUMBER_PATH": (
            "generic_http_ocr",
            "page_number_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_MARKDOWN_PATH": ("generic_http_ocr", "markdown_path", str),
        "KG_GENERIC_HTTP_OCR_RAW_TEXT_PATH": ("generic_http_ocr", "raw_text_path", str),
        "KG_GENERIC_HTTP_OCR_LANGUAGE_PATH": (
            "generic_http_ocr",
            "detected_language_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_BLOCKS_PATH": ("generic_http_ocr", "blocks_path", str),
        "KG_GENERIC_HTTP_OCR_BLOCK_ID_PATH": ("generic_http_ocr", "block_id_path", str),
        "KG_GENERIC_HTTP_OCR_BLOCK_KIND_PATH": (
            "generic_http_ocr",
            "block_kind_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_BLOCK_TEXT_PATH": (
            "generic_http_ocr",
            "block_text_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_BLOCK_BBOX_PATH": (
            "generic_http_ocr",
            "block_bbox_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_BLOCK_CONFIDENCE_PATH": (
            "generic_http_ocr",
            "block_confidence_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_BLOCK_METADATA_PATH": (
            "generic_http_ocr",
            "block_metadata_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_ASSETS_PATH": ("generic_http_ocr", "assets_path", str),
        "KG_GENERIC_HTTP_OCR_ASSET_ID_PATH": ("generic_http_ocr", "asset_id_path", str),
        "KG_GENERIC_HTTP_OCR_ASSET_KIND_PATH": (
            "generic_http_ocr",
            "asset_kind_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_ASSET_URI_PATH": ("generic_http_ocr", "asset_uri_path", str),
        "KG_GENERIC_HTTP_OCR_ASSET_PAGE_NUMBER_PATH": (
            "generic_http_ocr",
            "asset_page_number_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_ASSET_CONFIDENCE_PATH": (
            "generic_http_ocr",
            "asset_confidence_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_ASSET_METADATA_PATH": (
            "generic_http_ocr",
            "asset_metadata_path",
            str,
        ),
        "KG_GENERIC_HTTP_OCR_WARNINGS_PATH": ("generic_http_ocr", "warnings_path", str),
        "KG_GENERIC_HTTP_OCR_ERROR_PATH": ("generic_http_ocr", "error_path", str),
        "KG_GENERIC_HTTP_OCR_STATUS_PATH": ("generic_http_ocr", "status_path", str),
        "KG_LLM_PROVIDER": ("llm", "provider", str),
        "KG_LLM_ENDPOINT": ("llm", "endpoint", str),
        "KG_LLM_MODEL": ("llm", "model", str),
        "KG_LLM_API_KEY": ("llm", "api_key", str),
        "KG_LLM_API_VERSION": ("llm", "api_version", str),
        "KG_EMBED_PROVIDER": ("embedding", "provider", str),
        "KG_EMBED_ENDPOINT": ("embedding", "endpoint", str),
        "KG_EMBED_MODEL": ("embedding", "model", str),
        "KG_EMBED_API_KEY": ("embedding", "api_key", str),
        "KG_EMBED_API_VERSION": ("embedding", "api_version", str),
        "KG_EMBED_DIM": ("embedding", "dimension", int),
        "KG_EMBED_BATCH_SIZE": ("embedding", "batch_size", int),
        "KG_EMBED_DEVICE": ("embedding", "device", str),
        "KG_GRAPH_MAX_CHUNKS_PER_LLM_CALL": ("graph", "max_chunks_per_llm_call", int),
        "KG_GRAPH_MAX_ENTITIES_PER_BATCH": ("graph", "max_entities_per_batch", int),
        "KG_GRAPH_MAX_RELATIONS_PER_BATCH": ("graph", "max_relations_per_batch", int),
        "KG_GRAPH_GLEANING_MAX_PASSES": ("graph", "gleaning_max_passes", int),
        "KG_GRAPH_GLEANING_SATURATION_THRESHOLD": (
            "graph",
            "gleaning_saturation_threshold",
            int,
        ),
        "KG_GRAPH_MIN_ENTITY_CONFIDENCE": ("graph", "min_entity_confidence", float),
        "KG_GRAPH_MIN_RELATION_CONFIDENCE": ("graph", "min_relation_confidence", float),
        "KG_GRAPH_MIN_ENTITY_NAME_LENGTH": ("graph", "min_entity_name_length", int),
        "KG_GRAPH_REQUIRE_RELATION_ENDPOINT_GROUNDING": (
            "graph",
            "require_relation_endpoint_grounding",
            _str_to_bool,
        ),
        "KG_GRAPH_ENTITY_BLOCKLIST": ("graph", "entity_blocklist", _str_to_list),
        "KG_GRAPH_DESCRIPTION_MERGE_MIN_OBSERVATIONS": (
            "graph",
            "description_merge_min_observations",
            int,
        ),
        "KG_GRAPH_DESCRIPTION_MERGE_MAX_DESCRIPTIONS": (
            "graph",
            "description_merge_max_descriptions",
            int,
        ),
        "KG_GRAPH_DESCRIPTION_MERGE_MAX_EVIDENCE": (
            "graph",
            "description_merge_max_evidence",
            int,
        ),
        "KG_GRAPH_FAIL_ON_QUALITY_ERROR": ("graph", "fail_on_quality_error", _str_to_bool),
        "KG_OUTPUT_PATH": ("writer", "output_path", Path),
        "KG_WRITER": ("writer", "provider", str),
        "KG_CACHE_PROVIDER": ("cache", "provider", str),
        "KG_CACHE_PATH": ("cache", "path", Path),
        "KG_SNOWFLAKE_ACCOUNT": ("snowflake", "account", str),
        "KG_SNOWFLAKE_HOST": ("snowflake", "host", str),
        "KG_SNOWFLAKE_USER": ("snowflake", "user", str),
        "KG_SNOWFLAKE_PASSWORD": ("snowflake", "password", str),
        "KG_SNOWFLAKE_AUTHENTICATOR": (
            "snowflake",
            "authenticator",
            _snowflake_auth_alias,
        ),
        "KG_SNOWFLAKE_PRIVATE_KEY_PATH": ("snowflake", "private_key_path", Path),
        "KG_SNOWFLAKE_OAUTH_TOKEN": ("snowflake", "oauth_token", str),
        "KG_SNOWFLAKE_OAUTH_TOKEN_PATH": ("snowflake", "oauth_token_path", Path),
        "KG_SNOWFLAKE_DATABASE": ("snowflake", "database", str),
        "KG_SNOWFLAKE_SCHEMA": ("snowflake", "schema", str),
        "KG_SNOWFLAKE_ROLE": ("snowflake", "role", str),
        "KG_SNOWFLAKE_WAREHOUSE": ("snowflake", "warehouse", str),
        "KG_SNOWFLAKE_STAGE": ("snowflake", "stage", str),
        "KG_SNOWFLAKE_BULK_STAGE": ("snowflake", "bulk_stage", str),
        "KG_BULK_TARGET_FILE_MB": ("snowflake", "bulk_target_file_size_mb", int),
        "KG_SNOWFLAKE_BULK_TARGET_FILE_MB": (
            "snowflake",
            "bulk_target_file_size_mb",
            int,
        ),
        "KG_SNOWFLAKE_IMAGE_REPOSITORY": ("snowflake", "image_repository", str),
        "KG_SNOWFLAKE_IMAGE_NAME": ("snowflake", "image_name", str),
        "KG_SNOWFLAKE_IMAGE_DIGEST": ("snowflake", "image_digest", str),
        "KG_SNOWFLAKE_COMPUTE_POOL": ("snowflake", "compute_pool", str),
        "KG_SNOWFLAKE_COMPUTE_POOL_INSTANCE_FAMILY": (
            "snowflake",
            "compute_pool_instance_family",
            str,
        ),
        "KG_SNOWFLAKE_COMPUTE_POOL_MIN_NODES": (
            "snowflake",
            "compute_pool_min_nodes",
            int,
        ),
        "KG_SNOWFLAKE_COMPUTE_POOL_MAX_NODES": (
            "snowflake",
            "compute_pool_max_nodes",
            int,
        ),
        "KG_SNOWFLAKE_SERVICE_NAME": ("snowflake", "service_name", str),
        "KG_SNOWFLAKE_SERVICE_SPEC_STAGE": ("snowflake", "service_spec_stage", str),
        "KG_SPCS_CPU_REQUEST": ("snowflake", "service_cpu_request", str),
        "KG_SPCS_CPU_LIMIT": ("snowflake", "service_cpu_limit", str),
        "KG_SPCS_MEMORY_REQUEST": ("snowflake", "service_memory_request", str),
        "KG_SPCS_MEMORY_LIMIT": ("snowflake", "service_memory_limit", str),
        "KG_SPCS_GPU_COUNT": ("snowflake", "service_gpu_count", int),
    }
    _apply_env_mapping(data, alias_mapping, env)
    _apply_env_mapping(data, mapping, env)
    return data


def _apply_env_mapping(
    data: dict[str, Any],
    mapping: dict[str, tuple[str, str, Any]],
    env: dict[str, str],
) -> None:
    for env_name, (group, key, caster) in mapping.items():
        if env_name not in env or env[env_name] == "":
            continue
        data.setdefault(group, {})[key] = caster(env[env_name])


def _ai_backend_profile_from_env(env: dict[str, str]) -> dict[str, Any]:
    raw_backend = env.get("KG_AI_BACKEND")
    if not raw_backend:
        return {}
    backend = raw_backend.strip().lower()
    if backend == "oss":
        return {
            "ocr": {"provider": "mineru_internal"},
            "llm": {"provider": "openai_compatible"},
            "embedding": {"provider": "sentence_transformers"},
        }
    if backend == "cortex":
        return {
            "ocr": {"provider": "snowflake_cortex"},
            "llm": {"provider": "snowflake_cortex"},
            "embedding": {"provider": "snowflake_cortex"},
        }
    raise ValueError("KG_AI_BACKEND must be one of: oss, cortex")


def _file_source_alias(value: str) -> str:
    normalized = value.strip().lower()
    return {
        "mount": "local",
        "mounted": "local",
        "blob_sdk": "azure_blob",
        "azure_blob": "azure_blob",
        "stage": "snowflake_stage",
    }.get(normalized, value)


def _ocr_provider_alias(value: str) -> str:
    normalized = value.strip().lower()
    return {
        "mineru": "mineru_internal",
        "mineru_internal": "mineru_internal",
        "mineru_api": "mineru_api",
        "tesseract": "tesseract_internal",
        "tesseract_internal": "tesseract_internal",
        "cortex": "snowflake_cortex",
        "snowflake_cortex": "snowflake_cortex",
    }.get(normalized, value)


def _snowflake_auth_alias(value: str) -> str:
    normalized = value.strip().lower()
    return {
        "oauth_file": "oauth",
        "azure_oauth": "oauth",
        "oauth": "oauth",
        "keypair": "SNOWFLAKE_JWT",
        "key_pair": "SNOWFLAKE_JWT",
        "snowflake_jwt": "SNOWFLAKE_JWT",
        "password": "snowflake",
        "snowflake": "snowflake",
    }.get(normalized, value)


def _str_to_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean environment value, got: {value}")


def _str_to_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _minutes_to_seconds(value: str) -> int:
    minutes = int(value)
    if minutes <= 0:
        raise ValueError("Expected positive minute value")
    return minutes * 60


def _validate_provider_name(value: object, kind: ProviderKind) -> object:
    if not isinstance(value, str):
        return value
    supported = provider_names(kind)
    if value not in supported:
        # This is deliberately a configuration-load error, not just a preflight
        # error. Unknown provider names should never survive long enough for a
        # factory or runtime path to guess at intent.
        raise ValueError(
            f"Unsupported {kind} provider '{value}'. Supported providers: "
            f"{', '.join(sorted(supported))}"
        )
    return value
