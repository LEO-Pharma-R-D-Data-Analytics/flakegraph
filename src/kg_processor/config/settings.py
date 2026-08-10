"""Typed configuration model for the full local/on-prem/SPCS runtime.

Defaults intentionally describe the production local-open-source profile rather
than a test fallback. Smoke tests opt into fake/hash/builtin providers
explicitly, which keeps real runs from silently changing providers.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from kg_processor.config.provider_registry import ProviderKind, provider_names
from kg_processor.domain.consumption import RateCard, rate_card_from_mapping
from kg_processor.ports.llm import DEFAULT_LLM_TIMEOUT_SECONDS

_ENV_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_PLACEHOLDER_ONLY_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_MISSING_ENV_PLACEHOLDER = object()
DistributedStageName = Literal[
    "prepare_document",
    "extract_document_context",
    "extract_entity_window",
    "compact_entity_inventory",
    "extract_relation_window",
    "compact_document",
    "finalize_graph",
]


def _default_distributed_worker_stages() -> list[DistributedStageName]:
    """Return every distributed stage for a general-purpose worker process."""

    return [
        "prepare_document",
        "extract_document_context",
        "extract_entity_window",
        "compact_entity_inventory",
        "extract_relation_window",
        "compact_document",
        "finalize_graph",
    ]


class _SettingsModel(BaseModel):
    """Reject misspelled configuration keys at every nested settings boundary.

    Field names are accepted alongside their configuration aliases so a dump of
    these settings validates back into them. Distributed execution relies on that
    round trip: settings are serialized on the driver and rebuilt on each Spark
    executor, and a dump emits field names while ``extra="forbid"`` rejects
    anything the model does not declare.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RuntimeSettings(_SettingsModel):
    """Selects the deployment runtime profile without changing provider contracts."""

    runtime: Literal["local", "onprem", "spcs", "kubernetes"] = "local"


class JobSettings(_SettingsModel):
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


class DistributedSettings(_SettingsModel):
    """Configure durable multi-process execution independently of Kubernetes.

    PostgreSQL provides transactional task leases and durable storage with one
    operational dependency.
    The application layer still targets task/artifact ports, so another backend can
    be introduced without changing graph stages or provider implementations.
    """

    database_url: str | None = None
    worker_id: str | None = None
    worker_stages: list[DistributedStageName] = Field(
        default_factory=_default_distributed_worker_stages
    )
    # Live workers renew ownership in the background, so this window controls
    # crash detection rather than maximum task duration. Five minutes tolerates
    # brief coordination outages while returning work from a dead node promptly.
    lease_seconds: int = 300
    poll_interval_seconds: float = 1.0
    retry_delay_seconds: float = 15.0
    max_attempts: int = 3
    artifact_compression_level: int = Field(default=6, ge=0, le=9)
    max_artifact_bytes: int = 512 * 1024 * 1024
    artifact_uri: str | None = None
    artifact_endpoint_url: str | None = None
    artifact_access_key_id: str | None = None
    artifact_secret_access_key: str | None = None
    artifact_region: str = "us-east-1"
    finalization_engine: Literal["auto", "local", "spark"] = "auto"
    spark_master: str = "local[*]"
    spark_image: str | None = None
    spark_namespace: str = "default"
    spark_service_account: str = "flakegraph-spark"
    spark_executor_pod_template: str = "/etc/flakegraph-spark/executor-pod-template.yaml"
    # Kubernetes Secret holding the provider credentials. Executors are given a
    # reference to it rather than the values, which Spark would otherwise write
    # into every executor Pod spec in clear text.
    spark_provider_secret: str = "flakegraph-providers"
    spark_executor_instances: int = 4
    spark_executor_cores: int = 4
    spark_executor_memory: str = "8g"
    spark_executor_memory_overhead: str = "8g"
    spark_shuffle_partitions: int = 0

    @field_validator(
        "database_url",
        "worker_id",
        "artifact_uri",
        "artifact_endpoint_url",
        "artifact_access_key_id",
        "artifact_secret_access_key",
        "spark_image",
    )
    @classmethod
    def optional_text_must_not_be_blank(cls, value: str | None) -> str | None:
        """Reject present-but-empty values that would fail later and less clearly."""

        if value is not None and not value.strip():
            raise ValueError("distributed text settings must not be blank")
        return value

    @field_validator("worker_stages")
    @classmethod
    def worker_stages_must_not_be_empty(
        cls, value: list[DistributedStageName]
    ) -> list[DistributedStageName]:
        """Ensure a worker can claim at least one kind of useful work."""

        if not value:
            raise ValueError("distributed.worker_stages must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("distributed.worker_stages must not contain duplicates")
        return value

    @field_validator(
        "lease_seconds",
        "max_attempts",
        "max_artifact_bytes",
        "spark_executor_instances",
        "spark_executor_cores",
    )
    @classmethod
    def positive_integer_controls(cls, value: int) -> int:
        """Keep leases, retries, and payload bounds meaningful at runtime."""

        if value <= 0:
            raise ValueError("distributed integer controls must be positive")
        return value

    @field_validator("poll_interval_seconds", "retry_delay_seconds")
    @classmethod
    def non_negative_timing_controls(cls, value: float) -> float:
        """Allow immediate test retries while rejecting nonsensical negative delays."""

        if value < 0:
            raise ValueError("distributed timing controls must not be negative")
        return value

    @field_validator("spark_shuffle_partitions")
    @classmethod
    def shuffle_partitions_must_not_be_negative(cls, value: int) -> int:
        """Use zero for adaptive sizing while rejecting impossible negatives."""

        if value < 0:
            raise ValueError("distributed.spark_shuffle_partitions must not be negative")
        return value

    @field_validator(
        "artifact_region",
        "spark_master",
        "spark_namespace",
        "spark_service_account",
        "spark_executor_pod_template",
        "spark_executor_memory",
        "spark_executor_memory_overhead",
    )
    @classmethod
    def required_distributed_text_must_not_be_blank(cls, value: str) -> str:
        """Reject blank execution values before starting a remote data engine."""

        if not value.strip():
            raise ValueError("required distributed text settings must not be blank")
        return value

    @model_validator(mode="after")
    def scalable_finalization_requires_object_storage(self) -> DistributedSettings:
        """Require shared object storage when Spark is explicitly selected.

        Auto mode retains the zero-configuration local path when no shared store
        is configured. Kubernetes charts configure both together for the scalable
        path, avoiding an arbitrary document-count switch.
        """

        if self.finalization_engine == "spark" and not self.artifact_uri:
            raise ValueError("spark finalization requires distributed.artifact_uri")
        return self


class FileSettings(_SettingsModel):
    """Describes where source documents are discovered before OCR begins."""

    source: Literal["local", "manifest", "stage", "snowflake_stage", "azure_blob", "s3"] = "local"
    # Keep zero-configuration discovery independent from bundled benchmark
    # corpora. User-facing profiles and the app normally replace this root.
    input_path: Path = Path("data")
    manifest_path: Path | None = None
    stage_prefix: str | None = None
    include_globs: list[str] = Field(default_factory=lambda: ["**/*"])
    # Snowflake reports its own MD5 for staged files, which cannot be compared
    # with the content hashes every other source records. Reading the bytes once
    # keeps document identity consistent across runtimes; disable it only where
    # cross-runtime comparison and gold evaluation are not needed.
    stage_content_hash: bool = True

    @field_validator("source", mode="before")
    @classmethod
    def source_must_be_registered(cls, value: object) -> object:
        """Validate file-source names against the shared provider catalog."""

        return _validate_provider_name(value, "file_source")


class AzureBlobSettings(_SettingsModel):
    """Configures optional Azure Blob input discovery and local download staging."""

    account_url: str | None = None
    connection_string: str | None = None
    container: str | None = None
    prefix: str | None = None
    sas_token: str | None = None
    download_path: Path = Path("out/azure-blob")


class S3Settings(_SettingsModel):
    """Configures S3-compatible input discovery and local download staging."""

    bucket: str | None = None
    prefix: str | None = None
    endpoint_url: str | None = None
    region: str | None = None
    download_path: Path = Path("out/s3")


class OcrSettings(_SettingsModel):
    """Configures OCR provider selection plus provider-specific execution knobs."""

    provider: str = "mineru_internal"
    fallback_primary_provider: str = "builtin_text"
    fallback_secondary_provider: str = "mineru_internal"
    fallback_min_characters_per_page: int = 80
    fallback_max_sparse_page_ratio: float = 0.2
    fallback_max_unbroken_text_ratio: float = 0.5
    fallback_max_fragmented_text_ratio: float = 0.15
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

    def consumption_model(self) -> str:
        """Return the name document parsing should be billed under.

        Most OCR providers charge one way, so the provider name identifies the
        rate. Snowflake does not: AI_PARSE_DOCUMENT bills Layout at more than
        five times the OCR rate, and a card that cannot tell them apart has to
        pick one and be wrong about the other. The mode therefore travels with
        the model name into the consumption record.
        """

        if self.provider == "snowflake_cortex":
            return f"{self.provider}-{self.snowflake_parse_mode.lower()}"
        return self.provider

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate OCR provider names before factory construction."""

        return _validate_provider_name(value, "ocr")

    @field_validator("fallback_primary_provider", "fallback_secondary_provider", mode="before")
    @classmethod
    def fallback_providers_must_be_registered(cls, value: object) -> object:
        """Validate composed OCR provider names through the shared catalog."""

        return _validate_provider_name(value, "ocr")

    @field_validator("fallback_min_characters_per_page")
    @classmethod
    def fallback_density_must_be_positive(cls, value: int) -> int:
        """Require a meaningful positive text-density threshold for routing."""

        if value < 1:
            raise ValueError("fallback_min_characters_per_page must be positive")
        return value

    @field_validator(
        "fallback_max_sparse_page_ratio",
        "fallback_max_unbroken_text_ratio",
        "fallback_max_fragmented_text_ratio",
    )
    @classmethod
    def fallback_sparse_ratio_must_be_bounded(cls, value: float) -> float:
        """Keep fallback routing ratios within an interpretable range."""

        if not 0.0 <= value < 1.0:
            raise ValueError("fallback OCR routing ratios must be in [0, 1)")
        return value

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

    @field_validator("timeout_seconds")
    @classmethod
    def timeout_seconds_must_be_positive(cls, value: int) -> int:
        """Reject a timeout that would cancel every OCR subprocess or HTTP request.

        The value is forwarded to subprocess and HTTP transports where zero and
        negative deadlines expire immediately or are read as "no deadline".
        """

        if value <= 0:
            raise ValueError("ocr timeout_seconds must be positive")
        return value

    @field_validator("snowflake_parse_mode", mode="before")
    @classmethod
    def snowflake_parse_mode_uppercase(cls, value: object) -> object:
        """Normalize Snowflake Cortex parse modes to the literals accepted by SQL."""

        return value.upper() if isinstance(value, str) else value


class GenericHttpOcrSettings(_SettingsModel):
    """Maps an arbitrary HTTP OCR response into the normalized document schema."""

    max_response_bytes: int = 25 * 1024 * 1024
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

    @field_validator("max_response_bytes")
    @classmethod
    def max_response_bytes_must_be_positive(cls, value: int) -> int:
        """Accept zero as an explicit opt-out from the response-size bound."""

        if value < 0:
            raise ValueError("generic_http_ocr.max_response_bytes must be non-negative")
        return value


class LlmSettings(_SettingsModel):
    """Select the graph-extraction LLM transport, model, credentials, and timeout.

    Provider-specific construction remains in factories; this model only validates
    the portable configuration shared by local, hosted, and Snowflake backends.
    """

    provider: str = "openai_compatible"
    endpoint: str | None = None
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    api_version: str = "2025-01-01-preview"
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS
    context_window_tokens: int = 32_768

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate LLM provider names before adapter construction."""

        return _validate_provider_name(value, "llm")

    @field_validator("timeout_seconds", "context_window_tokens")
    @classmethod
    def timeout_seconds_must_be_positive(cls, value: int, info: Any) -> int:
        """Reject a timeout that would fail or cancel every provider request.

        A single validated value is propagated to extraction and enrichment calls
        so providers do not retain inconsistent hidden transport defaults.
        """

        if value <= 0:
            raise ValueError(f"llm {info.field_name} must be positive")
        return value


class EmbeddingSettings(_SettingsModel):
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


class ConsumptionSettings(_SettingsModel):
    """Prices used to turn recorded usage into cost.

    Rates are configuration rather than code: providers change prices, and a run
    already recorded its usage, so correcting the table reprices history without
    re-running anything. Keys are ``provider:model``, with ``provider:*`` as a
    deliberate fallback so a new model is priced approximately rather than
    silently counted as free.
    """

    rates: dict[str, dict[str, float]] = Field(default_factory=dict)
    # The hosted model local work is priced against. Without one, a saving would
    # have no stated alternative and could not be checked.
    local_reference: str | None = None
    # What one Snowflake AI credit costs this account. Deliberately zero by
    # default: the figure comes from the customer's contract, and a guessed one
    # would report a confident price that is quietly wrong. Until it is set, the
    # Cortex rates below leave their calls counted as unpriced, which the
    # application says out loud rather than showing as free.
    usd_per_credit: float = 0.0

    def rate_card(self) -> RateCard:
        """Build the domain rate card this configuration describes."""

        return rate_card_from_mapping(
            {
                "rates": self.rates,
                "local_reference": self.local_reference,
                "usd_per_credit": self.usd_per_credit,
            }
        )


class OntologySettings(_SettingsModel):
    """Select the optional reviewable ontology profile used across graph stages.

    When omitted, the loader derives a compatible fallback from graph entity and
    relation settings so generic configurations remain operational.

    ``profile`` carries the resolved profile itself. Runtimes that cannot see the
    author's filesystem — a Snowpark container reading a staged specification —
    receive the ontology by value, because a path alone would dangle there.
    """

    profile_path: Path | None = None
    profile: dict[str, Any] | None = None


class ExtractorSettings(_SettingsModel):
    """Select independently replaceable algorithms for two-pass extraction stages.

    Entity recognition may use an LLM or local GLiNER while relation extraction and
    verification retain their own interfaces and provider choices.
    """

    entity_provider: Literal["llm", "gliner"] = "llm"
    relation_provider: Literal["llm"] = "llm"
    verifier_provider: Literal["llm"] = "llm"
    gliner_model: str = "urchade/gliner_multi-v2.1"
    gliner_threshold: float = 0.50

    @field_validator("gliner_threshold")
    @classmethod
    def gliner_threshold_must_be_probability(cls, value: float) -> float:
        """Keep the local NER confidence threshold on a normalized probability scale.

        Validation occurs before the optional model is loaded, providing an immediate
        configuration error instead of a late inference failure.
        """

        if not 0.0 <= value <= 1.0:
            raise ValueError("gliner_threshold must be between 0 and 1")
        return value


class GraphSettings(_SettingsModel):
    """Define provider-neutral graph quality, throughput, and safety policy.

    Centralizing these defaults keeps OCR and LLM provider profiles behaviorally
    consistent. Explicit overrides remain available for measured corpus or backend
    constraints without duplicating policy across YAML files.
    """

    # These defaults are the provider-neutral quality profile. Provider YAML
    # files should normally select transports and credentials only; keeping the
    # extraction behavior here prevents provider profiles from changing graph
    # semantics merely because a transport configuration was copied.
    chunk_token_size: int = 500
    chunk_token_overlap: int = 60
    # A compact window keeps the structured extraction task exhaustive. Larger
    # section-sized requests reduce provider calls, but measured scientific-paper
    # runs omit substantial entities and relations even when the response remains
    # below its record limit. Fleet execution supplies throughput by leasing these
    # independent windows dynamically rather than weakening per-window recall.
    extraction_window_tokens: int = 700
    max_chunks_per_llm_call: int = 2
    document_context_tokens: int = 1200
    document_context_max_chunks: int = 3
    max_document_context_entities: int = 3
    extraction_parallelism: int = 2
    # Each window receives a completeness/gleaning pass, so two bounded responses
    # can recover more records without risking one oversized, truncated JSON body.
    max_entities_per_batch: int = 40
    max_relations_per_batch: int = 40
    gleaning_max_passes: int = 1
    gleaning_saturation_threshold: int = 10
    gleaning_min_uncovered_tokens: int = 40
    verify_relations: bool = True
    verification_min_confidence: float = 0.75
    entity_resolution_enabled: bool = True
    resolution_lexical_auto_merge: float = 0.96
    resolution_embedding_auto_merge: float = 0.94
    resolution_candidate_threshold: float = 0.80
    resolution_embedding_lexical_floor: float = 0.45
    resolution_max_candidates_per_mention: int = 3
    # A decision is a compact boolean/confidence/reason tuple. Forty bounded
    # candidates fit comfortably in one strict response and halve provider round
    # trips without changing candidate generation or merge thresholds.
    resolution_adjudication_batch_size: int = 40
    resolution_parallelism: int = 2
    resolution_llm_merge_min_confidence: float = 0.90
    deterministic_seed: int = 17
    min_entity_confidence: float = 0.0
    min_relation_confidence: float = 0.0
    # A grounded entity remains useful even when no relation survives extraction
    # or verification. Consumers can derive a connected/core view later; deleting
    # isolates here irreversibly couples entity recall to relation recall.
    drop_isolated_entities: bool = False
    min_entity_name_length: int = 2
    require_relation_endpoint_grounding: bool = True
    relation_weight_max: float = 10.0
    min_community_size: int = 2
    max_community_size: int = 50
    community_resolution: float = 1.0
    community_co_mention_weight: float = 0.05
    community_report_parallelism: int = 2
    description_merge_parallelism: int = 2
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
        "document_context_tokens",
        "document_context_max_chunks",
        "max_document_context_entities",
        "extraction_window_tokens",
        "extraction_parallelism",
        "max_entities_per_batch",
        "max_relations_per_batch",
        "gleaning_saturation_threshold",
        "min_entity_name_length",
        "min_community_size",
        "max_community_size",
        "community_report_parallelism",
        "description_merge_parallelism",
        "gleaning_min_uncovered_tokens",
        "resolution_max_candidates_per_mention",
        "resolution_adjudication_batch_size",
        "resolution_parallelism",
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

    @field_validator("community_resolution", "community_co_mention_weight")
    @classmethod
    def community_weights_must_be_positive(cls, value: float) -> float:
        """Require positive Leiden resolution and co-mention projection parameters.

        Non-positive values would either invalidate community optimization or erase
        the weak projection used to stabilize sparse graph regions.
        """

        if value <= 0:
            raise ValueError("community weights must be positive")
        return value

    @field_validator(
        "min_entity_confidence",
        "min_relation_confidence",
        "verification_min_confidence",
        "resolution_lexical_auto_merge",
        "resolution_embedding_auto_merge",
        "resolution_candidate_threshold",
        "resolution_embedding_lexical_floor",
        "resolution_llm_merge_min_confidence",
    )
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

    @model_validator(mode="after")
    def community_size_bounds_must_overlap(self) -> GraphSettings:
        """Reject community bounds that make every detected group ineligible."""

        if self.min_community_size > self.max_community_size:
            raise ValueError("min_community_size must not exceed max_community_size")
        return self


class WriterSettings(_SettingsModel):
    """Selects where the assembled graph artifacts are persisted."""

    provider: str = "local_artifacts"
    output_path: Path = Path("out/kg")

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate writer provider names before adapter construction."""

        return _validate_provider_name(value, "writer")


class CacheSettings(_SettingsModel):
    """Controls optional OCR and extraction result caching."""

    provider: Literal["none", "local", "snowflake"] = "none"
    path: Path = Path("out/cache")

    @field_validator("provider", mode="before")
    @classmethod
    def provider_must_be_registered(cls, value: object) -> object:
        """Validate cache provider names before adapter construction."""

        return _validate_provider_name(value, "cache")


class SnowflakeSettings(_SettingsModel):
    """Holds Snowflake connection, stage, image, and SPCS deployment settings.

    ``password_environment_variable`` and ``oauth_token_environment_variable``
    name where a secret lives instead of carrying it. Loading resolves the named
    variable into the matching credential when one is not configured directly,
    which lets a distributed plan travel to a finalizer worker as a reference
    rather than as a value.
    """

    account: str | None = None
    host: str | None = None
    user: str | None = None
    password: str | None = None
    password_environment_variable: str | None = None
    authenticator: str | None = None
    private_key_path: Path | None = None
    oauth_token: str | None = None
    oauth_token_environment_variable: str | None = None
    oauth_token_path: Path | None = Path("/snowflake/session/token")
    store_temporary_credential: bool = False
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
    service_cpu_request: str = "500m"
    service_cpu_limit: str = "1"
    service_memory_request: str = "3Gi"
    service_memory_limit: str = "5Gi"
    service_gpu_count: int = 0
    bulk_target_file_size_mb: int = 128

    @field_validator("authenticator", mode="before")
    @classmethod
    def normalize_authenticator_alias(cls, value: object) -> object:
        """Apply the same friendly aliases to file and environment configuration."""

        return _snowflake_auth_alias(value) if isinstance(value, str) else value

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


class Settings(_SettingsModel):
    """Provide one validated configuration consumed by CLI, factories, and workers.

    The object is the single source of effective runtime behavior after defaults,
    YAML, environment aliases, and explicit command overrides have been merged.
    """

    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    job: JobSettings = Field(default_factory=JobSettings)
    distributed: DistributedSettings = Field(default_factory=DistributedSettings)
    files: FileSettings = Field(default_factory=FileSettings)
    azure_blob: AzureBlobSettings = Field(default_factory=AzureBlobSettings)
    s3: S3Settings = Field(default_factory=S3Settings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    generic_http_ocr: GenericHttpOcrSettings = Field(default_factory=GenericHttpOcrSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    ontology: OntologySettings = Field(default_factory=OntologySettings)
    consumption: ConsumptionSettings = Field(default_factory=ConsumptionSettings)
    extractors: ExtractorSettings = Field(default_factory=ExtractorSettings)
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
        data: dict[str, Any] = _ambient_snowflake_from_env(env)
        if selected_file:
            data = _deep_update(data, _read_yaml(selected_file, env))
        inline_config = env.get("KG_CONFIG_JSON")
        if inline_config:
            data = _deep_update(data, _read_inline_json(inline_config))
        data = _deep_update(data, _from_env(env))
        if overrides:
            data = _deep_update(data, overrides)
        _resolve_snowflake_credential_references(data, env)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(_safe_validation_error(exc)) from None


def _resolve_snowflake_credential_references(data: dict[str, Any], env: dict[str, str]) -> None:
    """Read Snowflake secrets from the environment variables the config names.

    ``password_environment_variable`` and ``oauth_token_environment_variable``
    let a configuration record where a secret lives without embedding it, which
    is what a distributed plan propagates to finalizer workers. Resolving them
    here means every connection path receives the same credential the plan
    describes. A directly configured secret wins, and a named variable that is
    absent leaves the credential unset for preflight to report.
    """

    snowflake = data.get("snowflake")
    if not isinstance(snowflake, dict):
        return
    for reference_key, secret_key in (
        ("password_environment_variable", "password"),
        ("oauth_token_environment_variable", "oauth_token"),
    ):
        variable = snowflake.get(reference_key)
        if not isinstance(variable, str) or not variable or snowflake.get(secret_key):
            continue
        secret = env.get(variable)
        if secret:
            snowflake[secret_key] = secret


def _read_inline_json(value: str) -> dict[str, Any]:
    """Parse an orchestrator-provided configuration without exposing its value.

    Container platforms can inject the complete validated runtime contract through
    ``KG_CONFIG_JSON`` when the image's baked YAML is only a portable baseline.
    Parse failures identify the variable without echoing operator-specific content.
    """

    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"KG_CONFIG_JSON must contain valid JSON: {exc.msg}") from None
    if not isinstance(loaded, dict):
        raise ValueError("KG_CONFIG_JSON must contain a JSON object")
    return loaded


def _safe_validation_error(exc: ValidationError) -> str:
    """Render configuration failures without echoing rejected input values.

    Pydantic's default string includes ``input_value`` which can contain complete
    configuration sections and plaintext credentials. Locations and validator
    messages provide the useful diagnosis without reproducing those inputs.
    """

    details = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error.get("loc", ())) or "configuration"
        details.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "Invalid FlakeGraph configuration: " + "; ".join(details)


def _read_yaml(path: Path, env: dict[str, str]) -> dict[str, Any]:
    """Parse a configuration file into a mapping, reporting parse failures uniformly.

    Callers distinguish configuration problems from defects by catching
    ``ValueError``, so a malformed document is reported the same way as a
    well-formed document of the wrong shape.
    """

    with path.open("r", encoding="utf-8") as handle:
        try:
            loaded = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Configuration file is not valid YAML: {path}: {exc}") from None
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
        return _ENV_PLACEHOLDER_RE.sub(lambda match: _env_placeholder_value(match, env), value)
    return value


def _env_placeholder_value(match: re.Match[str], env: dict[str, str]) -> str:
    env_name = match.group(1)
    value = env.get(env_name)
    if value:
        return value
    raise ValueError(f"Environment variable {env_name} is required for configuration interpolation")


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _from_env(env: dict[str, str]) -> dict[str, Any]:
    """Translate supported environment variables into the nested settings structure.

    Keeping conversion functions in a declarative mapping makes precedence,
    validation, redaction, documentation, and tests share one environment surface.
    """

    # Keep env parsing as data rather than scattered conditionals. This makes
    # redaction, config printing, preflight, and tests reason over the same
    # supported environment-variable surface.
    data = _ai_backend_profile_from_env(env)
    alias_mapping: dict[str, tuple[str, str, Any]] = {
        # FlakeGraph-owned KG_* variables override YAML so container and
        # orchestrator injection follows one predictable precedence rule.
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
        "KG_BATCH_FILES": ("job", "file_batch_size", int),
        "KG_FILE_BATCH_SIZE": ("job", "file_batch_size", int),
        "KG_DISTRIBUTED_DATABASE_URL": ("distributed", "database_url", str),
        "KG_DISTRIBUTED_WORKER_ID": ("distributed", "worker_id", str),
        "KG_DISTRIBUTED_WORKER_STAGES": ("distributed", "worker_stages", _str_to_list),
        "KG_DISTRIBUTED_LEASE_SECONDS": ("distributed", "lease_seconds", int),
        "KG_DISTRIBUTED_POLL_INTERVAL_SECONDS": (
            "distributed",
            "poll_interval_seconds",
            float,
        ),
        "KG_DISTRIBUTED_RETRY_DELAY_SECONDS": (
            "distributed",
            "retry_delay_seconds",
            float,
        ),
        "KG_DISTRIBUTED_MAX_ATTEMPTS": ("distributed", "max_attempts", int),
        "KG_DISTRIBUTED_ARTIFACT_COMPRESSION_LEVEL": (
            "distributed",
            "artifact_compression_level",
            int,
        ),
        "KG_DISTRIBUTED_MAX_ARTIFACT_BYTES": (
            "distributed",
            "max_artifact_bytes",
            int,
        ),
        "KG_DISTRIBUTED_ARTIFACT_URI": ("distributed", "artifact_uri", str),
        "KG_DISTRIBUTED_ARTIFACT_ENDPOINT_URL": (
            "distributed",
            "artifact_endpoint_url",
            str,
        ),
        "KG_DISTRIBUTED_ARTIFACT_ACCESS_KEY_ID": (
            "distributed",
            "artifact_access_key_id",
            str,
        ),
        "KG_DISTRIBUTED_ARTIFACT_SECRET_ACCESS_KEY": (
            "distributed",
            "artifact_secret_access_key",
            str,
        ),
        "KG_DISTRIBUTED_ARTIFACT_REGION": ("distributed", "artifact_region", str),
        "KG_DISTRIBUTED_FINALIZATION_ENGINE": (
            "distributed",
            "finalization_engine",
            str,
        ),
        "KG_DISTRIBUTED_SPARK_MASTER": ("distributed", "spark_master", str),
        "KG_DISTRIBUTED_SPARK_IMAGE": ("distributed", "spark_image", str),
        "KG_DISTRIBUTED_SPARK_NAMESPACE": (
            "distributed",
            "spark_namespace",
            str,
        ),
        "KG_DISTRIBUTED_SPARK_SERVICE_ACCOUNT": (
            "distributed",
            "spark_service_account",
            str,
        ),
        "KG_DISTRIBUTED_SPARK_EXECUTOR_POD_TEMPLATE": (
            "distributed",
            "spark_executor_pod_template",
            str,
        ),
        "KG_DISTRIBUTED_SPARK_EXECUTOR_INSTANCES": (
            "distributed",
            "spark_executor_instances",
            int,
        ),
        "KG_DISTRIBUTED_SPARK_EXECUTOR_CORES": (
            "distributed",
            "spark_executor_cores",
            int,
        ),
        "KG_DISTRIBUTED_SPARK_EXECUTOR_MEMORY": (
            "distributed",
            "spark_executor_memory",
            str,
        ),
        "KG_DISTRIBUTED_SPARK_EXECUTOR_MEMORY_OVERHEAD": (
            "distributed",
            "spark_executor_memory_overhead",
            str,
        ),
        "KG_DISTRIBUTED_SPARK_SHUFFLE_PARTITIONS": (
            "distributed",
            "spark_shuffle_partitions",
            int,
        ),
        "KG_FILE_SOURCE": ("files", "source", _file_source_alias),
        "KG_INPUT_PATH": ("files", "input_path", Path),
        "KG_MANIFEST_PATH": ("files", "manifest_path", Path),
        "KG_STAGE_PREFIX": ("files", "stage_prefix", str),
        "KG_STAGE_CONTENT_HASH": ("files", "stage_content_hash", _str_to_bool),
        "KG_AZURE_BLOB_ACCOUNT_URL": ("azure_blob", "account_url", str),
        "KG_AZURE_BLOB_CONNECTION_STRING": ("azure_blob", "connection_string", str),
        "KG_AZURE_BLOB_CONTAINER": ("azure_blob", "container", str),
        "KG_AZURE_BLOB_PREFIX": ("azure_blob", "prefix", str),
        "KG_AZURE_BLOB_SAS_TOKEN": ("azure_blob", "sas_token", str),
        "KG_AZURE_BLOB_DOWNLOAD_PATH": ("azure_blob", "download_path", Path),
        "KG_S3_BUCKET": ("s3", "bucket", str),
        "KG_S3_PREFIX": ("s3", "prefix", str),
        "KG_S3_ENDPOINT_URL": ("s3", "endpoint_url", str),
        "KG_S3_REGION": ("s3", "region", str),
        "KG_S3_DOWNLOAD_PATH": ("s3", "download_path", Path),
        # Both spellings select the same field, so both accept the same friendly
        # engine aliases rather than making the accepted vocabulary depend on
        # which variable name an operator reached for.
        "KG_OCR_PROVIDER": ("ocr", "provider", _ocr_provider_alias),
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
        "KG_GENERIC_HTTP_OCR_MAX_RESPONSE_BYTES": (
            "generic_http_ocr",
            "max_response_bytes",
            int,
        ),
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
        "KG_LLM_TIMEOUT_SECONDS": ("llm", "timeout_seconds", int),
        "KG_EMBED_PROVIDER": ("embedding", "provider", str),
        "KG_EMBED_ENDPOINT": ("embedding", "endpoint", str),
        "KG_EMBED_MODEL": ("embedding", "model", str),
        "KG_EMBED_API_KEY": ("embedding", "api_key", str),
        "KG_EMBED_API_VERSION": ("embedding", "api_version", str),
        "KG_EMBED_DIM": ("embedding", "dimension", int),
        "KG_EMBED_BATCH_SIZE": ("embedding", "batch_size", int),
        "KG_EMBED_DEVICE": ("embedding", "device", str),
        "KG_ONTOLOGY_PROFILE": ("ontology", "profile_path", Path),
        "KG_ENTITY_EXTRACTOR": ("extractors", "entity_provider", str),
        "KG_GLINER_MODEL": ("extractors", "gliner_model", str),
        "KG_GLINER_THRESHOLD": ("extractors", "gliner_threshold", float),
        "KG_GRAPH_EXTRACTION_WINDOW_TOKENS": (
            "graph",
            "extraction_window_tokens",
            int,
        ),
        "KG_GRAPH_MAX_CHUNKS_PER_LLM_CALL": ("graph", "max_chunks_per_llm_call", int),
        "KG_GRAPH_EXTRACTION_PARALLELISM": ("graph", "extraction_parallelism", int),
        "KG_GRAPH_DESCRIPTION_MERGE_PARALLELISM": (
            "graph",
            "description_merge_parallelism",
            int,
        ),
        "KG_GRAPH_MAX_ENTITIES_PER_BATCH": ("graph", "max_entities_per_batch", int),
        "KG_GRAPH_MAX_RELATIONS_PER_BATCH": ("graph", "max_relations_per_batch", int),
        "KG_GRAPH_GLEANING_MAX_PASSES": ("graph", "gleaning_max_passes", int),
        "KG_GRAPH_GLEANING_MIN_UNCOVERED_TOKENS": (
            "graph",
            "gleaning_min_uncovered_tokens",
            int,
        ),
        "KG_GRAPH_GLEANING_SATURATION_THRESHOLD": (
            "graph",
            "gleaning_saturation_threshold",
            int,
        ),
        "KG_GRAPH_MIN_ENTITY_CONFIDENCE": ("graph", "min_entity_confidence", float),
        "KG_GRAPH_MIN_RELATION_CONFIDENCE": ("graph", "min_relation_confidence", float),
        "KG_GRAPH_VERIFY_RELATIONS": ("graph", "verify_relations", _str_to_bool),
        "KG_GRAPH_VERIFICATION_MIN_CONFIDENCE": (
            "graph",
            "verification_min_confidence",
            float,
        ),
        "KG_GRAPH_ENTITY_RESOLUTION_ENABLED": (
            "graph",
            "entity_resolution_enabled",
            _str_to_bool,
        ),
        "KG_GRAPH_RESOLUTION_LEXICAL_AUTO_MERGE": (
            "graph",
            "resolution_lexical_auto_merge",
            float,
        ),
        "KG_GRAPH_RESOLUTION_EMBEDDING_AUTO_MERGE": (
            "graph",
            "resolution_embedding_auto_merge",
            float,
        ),
        "KG_GRAPH_RESOLUTION_CANDIDATE_THRESHOLD": (
            "graph",
            "resolution_candidate_threshold",
            float,
        ),
        "KG_GRAPH_RESOLUTION_EMBEDDING_LEXICAL_FLOOR": (
            "graph",
            "resolution_embedding_lexical_floor",
            float,
        ),
        "KG_GRAPH_RESOLUTION_MAX_CANDIDATES_PER_MENTION": (
            "graph",
            "resolution_max_candidates_per_mention",
            int,
        ),
        "KG_GRAPH_RESOLUTION_ADJUDICATION_BATCH_SIZE": (
            "graph",
            "resolution_adjudication_batch_size",
            int,
        ),
        "KG_GRAPH_RESOLUTION_PARALLELISM": (
            "graph",
            "resolution_parallelism",
            int,
        ),
        "KG_GRAPH_RESOLUTION_LLM_MERGE_MIN_CONFIDENCE": (
            "graph",
            "resolution_llm_merge_min_confidence",
            float,
        ),
        "KG_GRAPH_DETERMINISTIC_SEED": ("graph", "deterministic_seed", int),
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
        "KG_GRAPH_COMMUNITY_REPORT_PARALLELISM": (
            "graph",
            "community_report_parallelism",
            int,
        ),
        # Both community bounds are env-settable because a cross-field validator
        # rejects a maximum below the minimum: exposing only one makes some
        # otherwise valid maxima unreachable without a configuration file.
        "KG_GRAPH_MIN_COMMUNITY_SIZE": ("graph", "min_community_size", int),
        "KG_GRAPH_MAX_COMMUNITY_SIZE": ("graph", "max_community_size", int),
        "KG_GRAPH_COMMUNITY_RESOLUTION": ("graph", "community_resolution", float),
        "KG_GRAPH_COMMUNITY_CO_MENTION_WEIGHT": (
            "graph",
            "community_co_mention_weight",
            float,
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
        "KG_SNOWFLAKE_STORE_TEMPORARY_CREDENTIAL": (
            "snowflake",
            "store_temporary_credential",
            _str_to_bool,
        ),
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


def _ambient_snowflake_from_env(env: dict[str, str]) -> dict[str, Any]:
    """Load Snowpark-provided Snowflake env vars as low-priority defaults."""

    data: dict[str, Any] = {}
    _apply_env_mapping(
        data,
        {
            "SNOWFLAKE_ACCOUNT": ("snowflake", "account", str),
            "SNOWFLAKE_HOST": ("snowflake", "host", str),
            "SNOWFLAKE_USER": ("snowflake", "user", str),
            "SNOWFLAKE_PASSWORD": ("snowflake", "password", str),
            "SNOWFLAKE_AUTH": ("snowflake", "authenticator", _snowflake_auth_alias),
            "SNOWFLAKE_AUTHENTICATOR": (
                "snowflake",
                "authenticator",
                _snowflake_auth_alias,
            ),
            "SNOWFLAKE_PRIVATE_KEY_PATH": ("snowflake", "private_key_path", Path),
            "SNOWFLAKE_OAUTH_TOKEN": ("snowflake", "oauth_token", str),
            "SNOWFLAKE_OAUTH_TOKEN_PATH": ("snowflake", "oauth_token_path", Path),
            "SNOWFLAKE_DATABASE": ("snowflake", "database", str),
            "SNOWFLAKE_SCHEMA": ("snowflake", "schema", str),
            "SNOWFLAKE_ROLE": ("snowflake", "role", str),
            "SNOWFLAKE_WAREHOUSE": ("snowflake", "warehouse", str),
        },
        env,
    )
    return data


def _apply_env_mapping(
    data: dict[str, Any],
    mapping: dict[str, tuple[str, str, Any]],
    env: dict[str, str],
) -> None:
    for env_name, (group, key, caster) in mapping.items():
        if env_name not in env or env[env_name] == "":
            continue
        try:
            data.setdefault(group, {})[key] = caster(env[env_name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {env_name}: {exc}") from exc


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
