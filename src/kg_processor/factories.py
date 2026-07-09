"""Factory wiring from typed settings to concrete provider adapters.

This module is the only place that maps provider names to implementations. That
centralization makes unsupported-provider errors deterministic and keeps
application modules dependent only on ports.
"""

from __future__ import annotations

from kg_processor.adapters.cache.local_json import LocalJsonCache
from kg_processor.adapters.cache.snowflake import SnowflakeCache
from kg_processor.adapters.embeddings.azure_openai import AzureOpenAIEmbeddingProvider
from kg_processor.adapters.embeddings.hash import HashEmbeddingProvider
from kg_processor.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from kg_processor.adapters.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)
from kg_processor.adapters.embeddings.snowflake_cortex import SnowflakeCortexEmbeddingProvider
from kg_processor.adapters.files.azure_blob import AzureBlobFileSource, AzureBlobFileSourceConfig
from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.files.manifest import ManifestFileSource
from kg_processor.adapters.files.snowflake_stage import SnowflakeStageFileSource
from kg_processor.adapters.jobs.snowflake import SnowflakeJobManager
from kg_processor.adapters.llm.azure_openai import AzureOpenAILlmProvider
from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.adapters.llm.openai_compatible import OpenAICompatibleLlmProvider
from kg_processor.adapters.llm.snowflake_cortex import SnowflakeCortexLlmProvider
from kg_processor.adapters.llm.vllm_local import VllmLocalLlmProvider
from kg_processor.adapters.ocr.builtin_text import BuiltinTextOcrProvider
from kg_processor.adapters.ocr.generic_http import GenericHttpOcrProvider
from kg_processor.adapters.ocr.mineru_api import MineruApiOcrProvider
from kg_processor.adapters.ocr.mineru_internal import MineruInternalOcrProvider
from kg_processor.adapters.ocr.snowflake_cortex import SnowflakeCortexOcrProvider
from kg_processor.adapters.ocr.tesseract_internal import TesseractInternalOcrProvider
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
from kg_processor.adapters.writers.local_artifacts import LocalArtifactsWriter
from kg_processor.adapters.writers.snowflake_bulk import SnowflakeBulkWriter
from kg_processor.adapters.writers.snowflake_direct import SnowflakeDirectWriter
from kg_processor.config.settings import Settings
from kg_processor.ports.cache import PipelineCache
from kg_processor.ports.embeddings import EmbeddingProvider
from kg_processor.ports.file_source import FileSource
from kg_processor.ports.graph_writer import GraphWriter
from kg_processor.ports.llm import LlmProvider
from kg_processor.ports.ocr import OcrProvider


def build_file_source(settings: Settings) -> FileSource:
    """Build the configured file-source adapter."""

    # Provider construction is intentionally explicit. Missing credentials or
    # unknown provider names fail here instead of silently swapping to a test or
    # local implementation.
    if settings.files.source == "local":
        return LocalFileSource(settings.files.input_path, settings.files.include_globs)
    if settings.files.source == "manifest":
        if not settings.files.manifest_path:
            raise ValueError("manifest file source requires files.manifest_path")
        return ManifestFileSource(settings.files.manifest_path, settings.files.include_globs)
    if settings.files.source == "azure_blob":
        if not settings.azure_blob.container:
            raise ValueError("azure_blob file source requires azure_blob.container")
        return AzureBlobFileSource(
            AzureBlobFileSourceConfig(
                account_url=settings.azure_blob.account_url,
                connection_string=settings.azure_blob.connection_string,
                container=settings.azure_blob.container,
                prefix=settings.azure_blob.prefix,
                sas_token=settings.azure_blob.sas_token,
                download_path=settings.azure_blob.download_path,
            ),
            settings.files.include_globs,
        )
    if settings.files.source in {"stage", "snowflake_stage"}:
        if not settings.snowflake.stage:
            raise ValueError("snowflake_stage file source requires snowflake.stage")
        return SnowflakeStageFileSource(
            _snowflake_config(settings),
            settings.snowflake.stage,
            settings.files.stage_prefix,
            settings.files.include_globs,
        )
    raise ValueError(f"Unsupported file source: {settings.files.source}")


def build_ocr_provider(settings: Settings) -> OcrProvider:
    """Build the configured OCR provider adapter."""

    # OCR providers all normalize into ParsedDocument, so the pipeline never
    # needs provider-specific branches after this factory boundary.
    if settings.ocr.provider == "mineru_internal":
        return MineruInternalOcrProvider(settings.ocr.mineru_command)
    if settings.ocr.provider == "mineru_api":
        if not settings.ocr.mineru_api_url:
            raise ValueError("mineru_api OCR requires mineru_api_url")
        return MineruApiOcrProvider(settings.ocr.mineru_api_url, settings.ocr.mineru_api_key)
    if settings.ocr.provider == "builtin_text":
        return BuiltinTextOcrProvider()
    if settings.ocr.provider == "generic_http":
        return GenericHttpOcrProvider(settings.generic_http_ocr)
    if settings.ocr.provider == "tesseract_internal":
        return TesseractInternalOcrProvider(
            settings.ocr.tesseract_command,
            settings.ocr.tesseract_pdf_renderer_command,
            settings.ocr.tesseract_dpi,
        )
    if settings.ocr.provider == "snowflake_cortex":
        return SnowflakeCortexOcrProvider(_snowflake_config(settings))
    raise ValueError(f"Unsupported OCR provider: {settings.ocr.provider}")


def build_llm_provider(settings: Settings) -> LlmProvider:
    """Build the configured LLM provider adapter."""

    # Fake/vLLM/OpenAI/Azure/Cortex all implement the same structured LLM port;
    # selection stays in configuration rather than prompt/extraction code.
    if settings.llm.provider == "fake":
        return FakeLlmProvider()
    if settings.llm.provider == "openai_compatible":
        if not settings.llm.endpoint or not settings.llm.api_key:
            raise ValueError("openai_compatible LLM requires endpoint and api_key")
        return OpenAICompatibleLlmProvider(
            settings.llm.endpoint,
            settings.llm.api_key,
            settings.llm.model,
        )
    if settings.llm.provider == "azure_openai":
        if not settings.llm.endpoint or not settings.llm.api_key:
            raise ValueError("azure_openai LLM requires endpoint and api_key")
        return AzureOpenAILlmProvider(
            settings.llm.endpoint,
            settings.llm.api_key,
            settings.llm.api_version,
            settings.llm.model,
        )
    if settings.llm.provider == "vllm_local":
        if not settings.llm.endpoint:
            raise ValueError("vllm_local LLM requires endpoint")
        return VllmLocalLlmProvider(
            settings.llm.endpoint,
            settings.llm.api_key,
            settings.llm.model,
        )
    if settings.llm.provider == "snowflake_cortex":
        return SnowflakeCortexLlmProvider(_snowflake_config(settings), settings.llm.model)
    raise ValueError(f"Unsupported LLM provider: {settings.llm.provider}")


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured embedding provider adapter."""

    # Embedding providers are dimension-checked later by quality/inspect logic;
    # this factory only enforces the credentials needed to call the provider.
    if settings.embedding.provider == "hash":
        return HashEmbeddingProvider()
    if settings.embedding.provider == "sentence_transformers":
        return SentenceTransformersEmbeddingProvider(device=settings.embedding.device)
    if settings.embedding.provider == "openai_compatible":
        if not settings.embedding.endpoint or not settings.embedding.api_key:
            raise ValueError("openai_compatible embeddings require endpoint and api_key")
        return OpenAICompatibleEmbeddingProvider(
            settings.embedding.endpoint,
            settings.embedding.api_key,
        )
    if settings.embedding.provider == "azure_openai":
        if not settings.embedding.endpoint or not settings.embedding.api_key:
            raise ValueError("azure_openai embeddings require endpoint and api_key")
        return AzureOpenAIEmbeddingProvider(
            settings.embedding.endpoint,
            settings.embedding.api_key,
            settings.embedding.api_version,
        )
    if settings.embedding.provider == "snowflake_cortex":
        return SnowflakeCortexEmbeddingProvider(_snowflake_config(settings))
    raise ValueError(f"Unsupported embedding provider: {settings.embedding.provider}")


def build_writer(settings: Settings) -> GraphWriter:
    """Build the configured graph writer adapter."""

    # Writers receive the same GraphWriteBatch shape. Local artifacts and
    # Snowflake tables should therefore stay schema-parity concerns, not
    # separate pipeline implementations.
    if settings.writer.provider == "local_artifacts":
        return LocalArtifactsWriter(settings.writer.output_path)
    if settings.writer.provider == "snowflake_direct":
        return SnowflakeDirectWriter(
            _snowflake_config(settings),
            settings.embedding.dimension,
        )
    if settings.writer.provider == "snowflake_bulk":
        if not settings.snowflake.bulk_stage:
            raise ValueError("snowflake_bulk writer requires snowflake.bulk_stage")
        return SnowflakeBulkWriter(
            _snowflake_config(settings),
            settings.embedding.dimension,
            settings.snowflake.bulk_stage,
            target_file_size_bytes=settings.snowflake.bulk_target_file_size_mb * 1024 * 1024,
        )
    raise ValueError(f"Unsupported graph writer: {settings.writer.provider}")


def build_cache(settings: Settings) -> PipelineCache | None:
    """Build the optional cache adapter, or None when caching is disabled."""

    if settings.cache.provider == "none":
        return None
    if settings.cache.provider == "local":
        return LocalJsonCache(settings.cache.path)
    if settings.cache.provider == "snowflake":
        return SnowflakeCache(_snowflake_config(settings))
    raise ValueError(f"Unsupported cache provider: {settings.cache.provider}")


def build_job_manager(settings: Settings) -> SnowflakeJobManager | None:
    """Build a Snowflake job manager when leasing or file queues are enabled."""

    if not settings.job.use_lease and not settings.job.use_file_queue:
        return None
    return SnowflakeJobManager(_snowflake_config(settings))


def _snowflake_config(settings: Settings) -> SnowflakeConnectionConfig:
    if not settings.snowflake.database or not settings.snowflake.schema_name:
        raise ValueError("Snowflake providers require database and schema")
    return SnowflakeConnectionConfig(
        account=settings.snowflake.account,
        host=settings.snowflake.host,
        user=settings.snowflake.user,
        password=settings.snowflake.password,
        authenticator=settings.snowflake.authenticator,
        private_key_path=settings.snowflake.private_key_path,
        oauth_token=settings.snowflake.oauth_token,
        oauth_token_path=settings.snowflake.oauth_token_path,
        database=settings.snowflake.database,
        schema_name=settings.snowflake.schema_name,
        role=settings.snowflake.role,
        warehouse=settings.snowflake.warehouse,
    )
