from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg_processor.config.settings import Settings


def test_settings_defaults_match_local_open_source_runtime_profile() -> None:
    """Lock zero-configuration runtime behavior to real providers and shared quality defaults.

    New users should receive the tested path automatically.
    """

    settings = Settings.load(env={})

    assert settings.runtime.runtime == "local"
    assert settings.files.source == "local"
    assert settings.ocr.provider == "mineru_internal"
    assert settings.llm.provider == "openai_compatible"
    assert settings.llm.timeout_seconds == 180
    assert settings.embedding.provider == "sentence_transformers"
    assert settings.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding.dimension == 384
    assert settings.embedding.batch_size == 32
    assert settings.graph.chunk_token_size == 500
    assert settings.graph.chunk_token_overlap == 60
    assert settings.graph.extraction_window_tokens == 700
    assert settings.graph.max_chunks_per_llm_call == 2
    assert settings.graph.max_entities_per_batch == 40
    assert settings.graph.max_relations_per_batch == 40
    assert settings.graph.gleaning_max_passes == 1
    assert settings.graph.drop_isolated_entities is False
    assert settings.graph.verify_relations is True
    assert settings.graph.extraction_parallelism == 2
    assert settings.graph.resolution_parallelism == 2
    assert settings.graph.community_report_parallelism == 2
    assert settings.graph.description_merge_parallelism == 2
    assert settings.distributed.lease_seconds == 300
    assert settings.distributed.worker_stages == [
        "prepare_document",
        "extract_document_context",
        "extract_entity_window",
        "compact_entity_inventory",
        "extract_relation_window",
        "compact_document",
        "finalize_graph",
    ]
    assert settings.writer.provider == "local_artifacts"


def test_settings_env_overrides_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
job:
  job_id: yaml-job
files:
  input_path: yaml-data
embedding:
  dimension: 32
""",
        encoding="utf-8",
    )

    settings = Settings.load(
        config,
        env={
            "KG_JOB_ID": "env-job",
            "KG_INPUT_PATH": str(tmp_path / "env-data"),
            "KG_EMBED_DIM": "64",
        },
    )

    assert settings.job.job_id == "env-job"
    assert settings.files.input_path == tmp_path / "env-data"
    assert settings.embedding.dimension == 64


def test_settings_cli_overrides_env(tmp_path: Path) -> None:
    settings = Settings.load(
        env={"KG_JOB_ID": "env-job"},
        overrides={"job": {"job_id": "cli-job"}},
    )

    assert settings.job.job_id == "cli-job"


def test_settings_loads_inline_container_configuration_before_explicit_env() -> None:
    """Apply a complete orchestrator contract while retaining env precedence."""

    inline = json.dumps(
        {
            "graph": {"max_chunks_per_llm_call": 1},
            "llm": {"model": "inline-model"},
        }
    )

    settings = Settings.load(
        env={
            "KG_CONFIG_JSON": inline,
            "KG_LLM_MODEL": "explicit-env-model",
        }
    )

    assert settings.graph.max_chunks_per_llm_call == 1
    assert settings.llm.model == "explicit-env-model"


@pytest.mark.parametrize("value", ["not-json", "[]"])
def test_settings_rejects_invalid_inline_container_configuration(value: str) -> None:
    """Reject malformed configuration without reproducing its complete value."""

    with pytest.raises(ValueError, match="KG_CONFIG_JSON"):
        Settings.load(env={"KG_CONFIG_JSON": value})


def test_settings_validation_error_does_not_echo_secret_input() -> None:
    """Report a precise invalid field without serializing sibling credentials."""

    with pytest.raises(ValueError) as captured:
        Settings.load(
            env={},
            overrides={
                "snowflake": {
                    "password": "do-not-print-this-password",
                    "service_gpu_count": -1,
                }
            },
        )

    message = str(captured.value)
    assert "snowflake.service_gpu_count" in message
    assert "do-not-print-this-password" not in message


def test_settings_allows_explicit_llm_timeout_override() -> None:
    """Keep the shared timeout overrideable for exceptionally slow validated models.

    Environment configuration should require no YAML edits.
    """

    settings = Settings.load(env={"KG_LLM_TIMEOUT_SECONDS": "900"})

    assert settings.llm.timeout_seconds == 900


def test_settings_loads_file_queue_runtime_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_JOB_USE_FILE_QUEUE": "true",
            "KG_WORKER_ID": "worker-1",
            "KG_JOB_LEASE_SECONDS": "420",
            "KG_BATCH_FILES": "250",
        }
    )

    assert settings.job.use_file_queue is True
    assert settings.job.lease_owner == "worker-1"
    assert settings.job.lease_seconds == 420
    assert settings.job.file_batch_size == 250


def test_settings_loads_spcs_and_snowflake_spec_aliases_from_env() -> None:
    settings = Settings.load(
        env={
            "SNOWFLAKE_ACCOUNT": "EXAMPLE-ACCOUNT",
            "SNOWFLAKE_HOST": "example-account.snowflakecomputing.com",
            "SNOWFLAKE_USER": "KG_SERVICE_USER",
            "SNOWFLAKE_AUTH": "oauth_file",
            "SNOWFLAKE_DATABASE": "KG_DB",
            "SNOWFLAKE_SCHEMA": "GRAPH",
            "SNOWFLAKE_ROLE": "KG_WRITER",
            "SNOWFLAKE_WAREHOUSE": "KG_PROCESSOR_WH",
            "SNOWFLAKE_OAUTH_TOKEN_PATH": "/snowflake/session/token",
            "KG_STAGE": "@KG_DB.GRAPH.KG_DOCS",
            "KG_BULK_STAGE": "@KG_DB.GRAPH.KG_BULK",
        }
    )

    assert settings.snowflake.account == "EXAMPLE-ACCOUNT"
    assert settings.snowflake.host == "example-account.snowflakecomputing.com"
    assert settings.snowflake.user == "KG_SERVICE_USER"
    assert settings.snowflake.authenticator == "oauth"
    assert settings.snowflake.database == "KG_DB"
    assert settings.snowflake.schema_name == "GRAPH"
    assert settings.snowflake.role == "KG_WRITER"
    assert settings.snowflake.warehouse == "KG_PROCESSOR_WH"
    assert settings.snowflake.oauth_token_path == Path("/snowflake/session/token")
    assert settings.snowflake.stage == "@KG_DB.GRAPH.KG_DOCS"
    assert settings.snowflake.bulk_stage == "@KG_DB.GRAPH.KG_BULK"


def test_settings_kg_snowflake_env_overrides_ambient_snowflake_aliases() -> None:
    settings = Settings.load(
        env={
            "SNOWFLAKE_ACCOUNT": "ambient-account",
            "SNOWFLAKE_AUTH": "oauth_file",
            "KG_SNOWFLAKE_ACCOUNT": "configured-account",
            "KG_SNOWFLAKE_AUTHENTICATOR": "keypair",
        }
    )

    assert settings.snowflake.account == "configured-account"
    assert settings.snowflake.authenticator == "SNOWFLAKE_JWT"


def test_settings_normalizes_snowflake_authenticator_alias_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("snowflake:\n  authenticator: keypair\n", encoding="utf-8")

    assert Settings.load(config, env={}).snowflake.authenticator == "SNOWFLAKE_JWT"


def test_settings_rejects_unknown_nested_keys() -> None:
    with pytest.raises(ValueError, match="graph.min_entity_confidnce: Extra inputs"):
        Settings.load(overrides={"graph": {"min_entity_confidnce": 0.9}})


def test_settings_yaml_overrides_ambient_snowflake_env_but_not_kg_env(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
snowflake:
  account: yaml-account
  database: YAML_DB
  schema: YAML_SCHEMA
""",
        encoding="utf-8",
    )

    settings = Settings.load(
        config,
        env={
            "SNOWFLAKE_ACCOUNT": "ambient-account",
            "SNOWFLAKE_DATABASE": "AMBIENT_DB",
            "KG_SNOWFLAKE_SCHEMA": "KG_SCHEMA",
        },
    )

    assert settings.snowflake.account == "yaml-account"
    assert settings.snowflake.database == "YAML_DB"
    assert settings.snowflake.schema_name == "KG_SCHEMA"


def test_settings_loads_spec_blob_and_file_source_aliases_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_FILE_SOURCE": "blob_sdk",
            "KG_BLOB_ACCOUNT_URL": "https://storage.example",
            "KG_BLOB_CONNECTION_STRING": "UseDevelopmentStorage=true",
            "KG_BLOB_CONTAINER": "documents",
            "KG_BLOB_PREFIX": "incoming",
            "KG_BLOB_SAS_TOKEN": "sas",
            "KG_BLOB_DOWNLOAD_PATH": "out/spec-blob-downloads",
        }
    )

    assert settings.files.source == "azure_blob"
    assert settings.azure_blob.account_url == "https://storage.example"
    assert settings.azure_blob.connection_string == "UseDevelopmentStorage=true"
    assert settings.azure_blob.container == "documents"
    assert settings.azure_blob.prefix == "incoming"
    assert settings.azure_blob.sas_token == "sas"
    assert settings.azure_blob.download_path == Path("out/spec-blob-downloads")


def test_settings_expands_ai_backend_profiles_before_specific_provider_overrides() -> None:
    oss_settings = Settings.load(env={"KG_AI_BACKEND": "oss"})

    assert oss_settings.ocr.provider == "mineru_internal"
    assert oss_settings.llm.provider == "openai_compatible"
    assert oss_settings.embedding.provider == "sentence_transformers"

    cortex_settings = Settings.load(env={"KG_AI_BACKEND": "cortex"})

    assert cortex_settings.ocr.provider == "snowflake_cortex"
    assert cortex_settings.llm.provider == "snowflake_cortex"
    assert cortex_settings.embedding.provider == "snowflake_cortex"

    override_settings = Settings.load(
        env={
            "KG_AI_BACKEND": "cortex",
            "KG_OCR_ENGINE": "tesseract",
            "KG_LLM_PROVIDER": "vllm_local",
            "KG_EMBED_PROVIDER": "sentence_transformers",
        }
    )

    assert override_settings.ocr.provider == "tesseract_internal"
    assert override_settings.llm.provider == "vllm_local"
    assert override_settings.embedding.provider == "sentence_transformers"


def test_local_vllm_profile_defaults_to_qwen36_endpoint() -> None:
    settings = Settings.load(Path("configs/local-vllm-mineru-oss.yaml"), env={})

    assert settings.llm.provider == "vllm_local"
    assert settings.llm.endpoint == "http://localhost:8000/v1"
    assert settings.llm.model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert settings.llm.api_key is None


def test_local_vllm_qwen_profile_defaults_to_real_local_providers() -> None:
    """Verify the quick-start profile selects vLLM and local embeddings."""

    settings = Settings.load(
        Path("data/martial_arts/configs/local-vllm-qwen36.yaml"),
        env={},
    )

    assert settings.ocr.provider == "builtin_text"
    assert settings.files.input_path == Path("data/martial_arts/files")
    assert "*.html" in settings.files.include_globs
    assert all(not pattern.startswith("**/") for pattern in settings.files.include_globs)
    assert "*.txt" in settings.files.include_globs
    assert "**/*.png" not in settings.files.include_globs
    assert settings.llm.provider == "vllm_local"
    assert settings.llm.endpoint == "http://localhost:8000/v1"
    assert settings.llm.model == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert settings.llm.api_key is None
    assert settings.llm.timeout_seconds == 180
    assert settings.embedding.provider == "sentence_transformers"
    assert settings.embedding.endpoint is None
    assert settings.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding.api_key is None
    assert settings.embedding.dimension == 384
    assert settings.ontology.profile_path == Path("data/martial_arts/ontology.yaml")
    assert settings.graph.chunk_token_size == 500
    assert settings.graph.chunk_token_overlap == 60
    assert settings.graph.extraction_window_tokens == 700
    assert settings.graph.max_chunks_per_llm_call == 2
    assert settings.graph.gleaning_max_passes == 1
    assert settings.graph.extraction_parallelism == 2
    assert settings.graph.resolution_embedding_lexical_floor == 0.45
    assert settings.graph.resolution_max_candidates_per_mention == 3
    assert settings.graph.resolution_adjudication_batch_size == 40
    assert settings.graph.resolution_parallelism == 2
    assert settings.graph.community_report_parallelism == 2
    assert settings.writer.output_path == Path("out/local-vllm-qwen36")


def test_local_ollama_profile_is_portable_and_keyless() -> None:
    """Keep the quick-start profile independent from host-specific inference stacks."""

    settings = Settings.load(
        Path("data/martial_arts/configs/local-ollama-qwen36.yaml"),
        env={},
    )

    assert settings.llm.provider == "ollama"
    assert settings.llm.endpoint == "http://localhost:11434"
    assert settings.llm.model == "qwen3.6:35b-a3b-q4_K_M"
    assert settings.llm.api_key is None
    assert settings.embedding.provider == "sentence_transformers"
    assert settings.writer.output_path == Path("out/local-ollama-qwen36")


def test_settings_loads_graph_parallelism_overrides_from_env() -> None:
    """Preserve concurrency overrides for measured multi-replica provider deployments.

    Every independent LLM-heavy stage remains tunable.
    """

    settings = Settings.load(
        Path("data/martial_arts/configs/local-vllm-qwen36.yaml"),
        env={
            "KG_GRAPH_EXTRACTION_PARALLELISM": "3",
            "KG_GRAPH_RESOLUTION_PARALLELISM": "2",
            "KG_GRAPH_RESOLUTION_MAX_CANDIDATES_PER_MENTION": "4",
            "KG_GRAPH_COMMUNITY_REPORT_PARALLELISM": "5",
        },
    )

    assert settings.graph.extraction_parallelism == 3
    assert settings.graph.resolution_parallelism == 2
    assert settings.graph.resolution_max_candidates_per_mention == 4
    assert settings.graph.community_report_parallelism == 5


def test_settings_rejects_unknown_ai_backend_profile() -> None:
    with pytest.raises(ValueError, match="KG_AI_BACKEND must be one of"):
        Settings.load(env={"KG_AI_BACKEND": "surprise"})


def test_settings_rejects_unknown_provider_names_before_preflight(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
files:
  source: file-source-that-does-not-exist
  input_path: data/martial_arts/files
ocr:
  provider: ocr-that-does-not-exist
llm:
  provider: llm-that-does-not-exist
embedding:
  provider: embedding-that-does-not-exist
writer:
  provider: writer-that-does-not-exist
cache:
  provider: cache-that-does-not-exist
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        Settings.load(config, env={})

    message = str(exc_info.value)
    assert "Unsupported file_source provider 'file-source-that-does-not-exist'" in message
    assert "azure_blob" in message
    assert "Unsupported ocr provider 'ocr-that-does-not-exist'" in message
    assert "Supported providers:" in message
    assert "mineru_internal" in message
    assert "Unsupported llm provider 'llm-that-does-not-exist'" in message
    assert "openai_compatible" in message
    assert "Unsupported embedding provider 'embedding-that-does-not-exist'" in message
    assert "sentence_transformers" in message
    assert "Unsupported writer provider 'writer-that-does-not-exist'" in message
    assert "local_artifacts" in message
    assert "Unsupported cache provider 'cache-that-does-not-exist'" in message
    assert "snowflake" in message


def test_settings_interpolates_yaml_environment_placeholders(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  endpoint: ${KG_LLM_ENDPOINT}
  api_key: ${KG_LLM_API_KEY}
""",
        encoding="utf-8",
    )

    settings = Settings.load(
        config,
        env={
            "KG_LLM_ENDPOINT": "https://example.test/v1",
            "KG_LLM_API_KEY": "secret",
        },
    )

    assert settings.llm.endpoint == "https://example.test/v1"
    assert settings.llm.api_key == "secret"


def test_settings_omits_missing_whole_value_yaml_environment_placeholders(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
embedding:
  dimension: ${KG_EMBED_DIM}
snowflake:
  account: ${KG_SNOWFLAKE_ACCOUNT}
  host: ${KG_SNOWFLAKE_HOST}
""",
        encoding="utf-8",
    )

    settings = Settings.load(
        config,
        env={
            "KG_EMBED_DIM": "",
        },
    )

    assert settings.embedding.dimension == 384
    assert settings.snowflake.account is None
    assert settings.snowflake.host is None


def test_settings_rejects_missing_mixed_yaml_environment_placeholder(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
llm:
  endpoint: https://${KG_MISSING_HOST}/v1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="KG_MISSING_HOST"):
        Settings.load(config, env={})


def test_settings_loads_mineru_provider_options_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_MINERU_METHOD": "ocr",
            "KG_MINERU_BACKEND": "pipeline",
            "KG_MINERU_EFFORT": "high",
            "KG_MINERU_API_URL": "http://mineru-api:8000",
            "KG_MINERU_SERVER_URL": "http://mineru-vlm:30000",
            "KG_MINERU_START_PAGE_ID": "2",
            "KG_MINERU_END_PAGE_ID": "4",
            "KG_MINERU_FORMULA": "false",
            "KG_MINERU_TABLE": "true",
            "KG_MINERU_IMAGE_ANALYSIS": "false",
            "KG_MINERU_CLIENT_SIDE_OUTPUT": "true",
        }
    )

    assert settings.ocr.mineru_method == "ocr"
    assert settings.ocr.mineru_backend == "pipeline"
    assert settings.ocr.mineru_effort == "high"
    assert settings.ocr.mineru_api_url == "http://mineru-api:8000"
    assert settings.ocr.mineru_server_url == "http://mineru-vlm:30000"
    assert settings.ocr.mineru_start_page_id == 2
    assert settings.ocr.mineru_end_page_id == 4
    assert settings.ocr.mineru_formula is False
    assert settings.ocr.mineru_table is True
    assert settings.ocr.mineru_image_analysis is False
    assert settings.ocr.mineru_client_side_output_generation is True


def test_settings_loads_tesseract_provider_options_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_TESSERACT_COMMAND": "custom-tesseract",
            "KG_TESSERACT_PDF_RENDERER_COMMAND": "custom-pdftoppm",
            "KG_TESSERACT_DPI": "200",
        }
    )

    assert settings.ocr.tesseract_command == "custom-tesseract"
    assert settings.ocr.tesseract_pdf_renderer_command == "custom-pdftoppm"
    assert settings.ocr.tesseract_dpi == 200


def test_settings_loads_generic_http_ocr_metadata_paths_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_GENERIC_HTTP_OCR_MAX_RESPONSE_BYTES": "1048576",
            "KG_GENERIC_HTTP_OCR_BLOCK_CONFIDENCE_PATH": "layout.score",
            "KG_GENERIC_HTTP_OCR_BLOCK_METADATA_PATH": "layout.attributes",
            "KG_GENERIC_HTTP_OCR_ASSET_CONFIDENCE_PATH": "media.score",
            "KG_GENERIC_HTTP_OCR_ASSET_METADATA_PATH": "media.details",
        }
    )

    assert settings.generic_http_ocr.max_response_bytes == 1048576
    assert settings.generic_http_ocr.block_confidence_path == "layout.score"
    assert settings.generic_http_ocr.block_metadata_path == "layout.attributes"
    assert settings.generic_http_ocr.asset_confidence_path == "media.score"
    assert settings.generic_http_ocr.asset_metadata_path == "media.details"


def test_settings_env_caster_errors_name_offending_variable() -> None:
    with pytest.raises(ValueError, match="KG_TESSERACT_DPI"):
        Settings.load(env={"KG_TESSERACT_DPI": "not-an-int"})


def test_settings_loads_azure_blob_options_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_FILE_SOURCE": "azure_blob",
            "KG_AZURE_BLOB_ACCOUNT_URL": "https://storage.example",
            "KG_AZURE_BLOB_CONTAINER": "documents",
            "KG_AZURE_BLOB_PREFIX": "incoming",
            "KG_AZURE_BLOB_SAS_TOKEN": "sas",
            "KG_AZURE_BLOB_DOWNLOAD_PATH": "out/blob-downloads",
        }
    )

    assert settings.files.source == "azure_blob"
    assert settings.azure_blob.account_url == "https://storage.example"
    assert settings.azure_blob.container == "documents"
    assert settings.azure_blob.prefix == "incoming"
    assert settings.azure_blob.sas_token == "sas"
    assert settings.azure_blob.download_path == Path("out/blob-downloads")


def test_settings_loads_manifest_file_source_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_FILE_SOURCE": "manifest",
            "KG_MANIFEST_PATH": "data/martial_arts/manifest.jsonl",
        }
    )

    assert settings.files.source == "manifest"
    assert settings.files.manifest_path == Path("data/martial_arts/manifest.jsonl")


def test_settings_loads_snowflake_deployment_options_from_env() -> None:
    digest = "sha256:" + "A" * 64
    settings = Settings.load(
        env={
            "KG_SNOWFLAKE_OAUTH_TOKEN_PATH": "/snowflake/session/token",
            "KG_SNOWFLAKE_IMAGE_REPOSITORY": "KG_DB.GRAPH.KG_IMAGES",
            "KG_SNOWFLAKE_IMAGE_NAME": "flakegraph:latest",
            "KG_SNOWFLAKE_IMAGE_DIGEST": digest,
            "KG_SNOWFLAKE_COMPUTE_POOL": "SYSTEM_COMPUTE_POOL_GPU",
            "KG_SNOWFLAKE_COMPUTE_POOL_INSTANCE_FAMILY": "GPU_NV_SM",
            "KG_SNOWFLAKE_COMPUTE_POOL_MIN_NODES": "1",
            "KG_SNOWFLAKE_COMPUTE_POOL_MAX_NODES": "3",
            "KG_SNOWFLAKE_SERVICE_NAME": "KG_PROCESSOR_JOB",
            "KG_SNOWFLAKE_SERVICE_SPEC_STAGE": "@KG_DB.GRAPH.KG_SERVICE_SPECS",
            "KG_SPCS_CPU_REQUEST": "2",
            "KG_SPCS_CPU_LIMIT": "4",
            "KG_SPCS_MEMORY_REQUEST": "16Gi",
            "KG_SPCS_MEMORY_LIMIT": "32Gi",
            "KG_SPCS_GPU_COUNT": "1",
            "KG_SNOWFLAKE_BULK_TARGET_FILE_MB": "192",
        }
    )

    assert settings.snowflake.oauth_token_path == Path("/snowflake/session/token")
    assert settings.snowflake.image_repository == "KG_DB.GRAPH.KG_IMAGES"
    assert settings.snowflake.image_name == "flakegraph:latest"
    assert settings.snowflake.image_digest == digest.lower()
    assert settings.snowflake.compute_pool == "SYSTEM_COMPUTE_POOL_GPU"
    assert settings.snowflake.compute_pool_instance_family == "GPU_NV_SM"
    assert settings.snowflake.compute_pool_min_nodes == 1
    assert settings.snowflake.compute_pool_max_nodes == 3
    assert settings.snowflake.service_name == "KG_PROCESSOR_JOB"
    assert settings.snowflake.service_spec_stage == "@KG_DB.GRAPH.KG_SERVICE_SPECS"
    assert settings.snowflake.service_cpu_request == "2"
    assert settings.snowflake.service_cpu_limit == "4"
    assert settings.snowflake.service_memory_request == "16Gi"
    assert settings.snowflake.service_memory_limit == "32Gi"
    assert settings.snowflake.service_gpu_count == 1
    assert settings.snowflake.bulk_target_file_size_mb == 192


def test_settings_loads_embedding_batch_and_device_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_EMBED_PROVIDER": "sentence_transformers",
            "KG_EMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
            "KG_EMBED_DIM": "384",
            "KG_EMBED_BATCH_SIZE": "16",
            "KG_EMBED_DEVICE": "cpu",
        }
    )

    assert settings.embedding.provider == "sentence_transformers"
    assert settings.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding.dimension == 384
    assert settings.embedding.batch_size == 16
    assert settings.embedding.device == "cpu"


def test_settings_loads_description_merge_options_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_GRAPH_DESCRIPTION_MERGE_MIN_OBSERVATIONS": "3",
            "KG_GRAPH_DESCRIPTION_MERGE_MAX_DESCRIPTIONS": "6",
            "KG_GRAPH_DESCRIPTION_MERGE_MAX_EVIDENCE": "4",
        }
    )

    assert settings.graph.description_merge_min_observations == 3
    assert settings.graph.description_merge_max_descriptions == 6
    assert settings.graph.description_merge_max_evidence == 4


def test_settings_loads_graph_quality_options_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_GRAPH_MAX_CHUNKS_PER_LLM_CALL": "4",
            "KG_GRAPH_MAX_ENTITIES_PER_BATCH": "12",
            "KG_GRAPH_MAX_RELATIONS_PER_BATCH": "18",
            "KG_GRAPH_GLEANING_MAX_PASSES": "2",
            "KG_GRAPH_GLEANING_SATURATION_THRESHOLD": "9",
            "KG_GRAPH_MIN_ENTITY_CONFIDENCE": "0.7",
            "KG_GRAPH_MIN_RELATION_CONFIDENCE": "0.6",
            "KG_GRAPH_MIN_ENTITY_NAME_LENGTH": "3",
            "KG_GRAPH_REQUIRE_RELATION_ENDPOINT_GROUNDING": "false",
            "KG_GRAPH_ENTITY_BLOCKLIST": "chapter,page, appendix ",
            "KG_GRAPH_FAIL_ON_QUALITY_ERROR": "false",
        }
    )

    assert settings.graph.max_chunks_per_llm_call == 4
    assert settings.graph.max_entities_per_batch == 12
    assert settings.graph.max_relations_per_batch == 18
    assert settings.graph.gleaning_max_passes == 2
    assert settings.graph.gleaning_saturation_threshold == 9
    assert settings.graph.min_entity_confidence == 0.7
    assert settings.graph.min_relation_confidence == 0.6
    assert settings.graph.min_entity_name_length == 3
    assert settings.graph.require_relation_endpoint_grounding is False
    assert settings.graph.entity_blocklist == ["chapter", "page", "appendix"]
    assert settings.graph.fail_on_quality_error is False


def test_settings_accepts_zero_gleaning_passes() -> None:
    settings = Settings.load(overrides={"graph": {"gleaning_max_passes": 0}})

    assert settings.graph.gleaning_max_passes == 0


def test_settings_loads_distributed_worker_controls_from_environment() -> None:
    settings = Settings.load(
        env={
            "KG_RUNTIME": "kubernetes",
            "KG_DISTRIBUTED_DATABASE_URL": "postgresql://database/flakegraph",
            "KG_DISTRIBUTED_WORKER_ID": "worker-1",
            "KG_DISTRIBUTED_WORKER_STAGES": ("prepare_document,extract_relation_window"),
            "KG_DISTRIBUTED_LEASE_SECONDS": "1200",
            "KG_DISTRIBUTED_POLL_INTERVAL_SECONDS": "0.5",
            "KG_DISTRIBUTED_RETRY_DELAY_SECONDS": "4",
            "KG_DISTRIBUTED_MAX_ATTEMPTS": "5",
            "KG_DISTRIBUTED_ARTIFACT_COMPRESSION_LEVEL": "7",
            "KG_DISTRIBUTED_MAX_ARTIFACT_BYTES": "2048",
            "KG_DISTRIBUTED_ARTIFACT_URI": "s3://flakegraph/runs",
            "KG_DISTRIBUTED_ARTIFACT_ENDPOINT_URL": "http://objects:8333",
            "KG_DISTRIBUTED_ARTIFACT_ACCESS_KEY_ID": "access",
            "KG_DISTRIBUTED_ARTIFACT_SECRET_ACCESS_KEY": "secret",
            "KG_DISTRIBUTED_FINALIZATION_ENGINE": "spark",
            "KG_DISTRIBUTED_SPARK_MASTER": "k8s://https://kubernetes.default.svc",
            "KG_DISTRIBUTED_SPARK_IMAGE": "registry/flakegraph-spark@sha256:abc",
            "KG_DISTRIBUTED_SPARK_NAMESPACE": "graph-system",
            "KG_DISTRIBUTED_SPARK_SERVICE_ACCOUNT": "graph-spark",
            "KG_DISTRIBUTED_SPARK_EXECUTOR_POD_TEMPLATE": "/spark/executor.yaml",
            "KG_DISTRIBUTED_SPARK_EXECUTOR_INSTANCES": "4",
            "KG_DISTRIBUTED_SPARK_EXECUTOR_MEMORY_OVERHEAD": "3g",
        }
    )

    assert settings.runtime.runtime == "kubernetes"
    assert settings.distributed.database_url == "postgresql://database/flakegraph"
    assert settings.distributed.worker_id == "worker-1"
    assert settings.distributed.worker_stages == [
        "prepare_document",
        "extract_relation_window",
    ]
    assert settings.distributed.lease_seconds == 1200
    assert settings.distributed.poll_interval_seconds == 0.5
    assert settings.distributed.retry_delay_seconds == 4
    assert settings.distributed.max_attempts == 5
    assert settings.distributed.artifact_compression_level == 7
    assert settings.distributed.max_artifact_bytes == 2048
    assert settings.distributed.artifact_uri == "s3://flakegraph/runs"
    assert settings.distributed.artifact_endpoint_url == "http://objects:8333"
    assert settings.distributed.finalization_engine == "spark"
    assert settings.distributed.spark_namespace == "graph-system"
    assert settings.distributed.spark_service_account == "graph-spark"
    assert settings.distributed.spark_executor_pod_template == "/spark/executor.yaml"
    assert settings.distributed.spark_executor_instances == 4
    assert settings.distributed.spark_executor_memory_overhead == "3g"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"embedding": {"batch_size": 0}}, "embedding batch_size must be positive"),
        ({"llm": {"timeout_seconds": 0}}, "llm timeout_seconds must be positive"),
        (
            {"generic_http_ocr": {"max_response_bytes": -1}},
            "generic_http_ocr.max_response_bytes must be non-negative",
        ),
        ({"graph": {"chunk_token_size": 0}}, "chunk_token_size must be positive"),
        ({"graph": {"chunk_token_overlap": -1}}, "chunk_token_overlap must be non-negative"),
        (
            {"graph": {"chunk_token_size": 10, "chunk_token_overlap": 10}},
            "chunk_token_overlap must be smaller than chunk_token_size",
        ),
        ({"graph": {"gleaning_max_passes": -1}}, "gleaning_max_passes must be non-negative"),
        ({"graph": {"relation_weight_max": 0}}, "relation_weight_max must be positive"),
        (
            {"graph": {"resolution_max_candidates_per_mention": 0}},
            "graph integer limits must be positive",
        ),
        (
            {"graph": {"resolution_parallelism": 0}},
            "graph integer limits must be positive",
        ),
        (
            {"graph": {"resolution_embedding_lexical_floor": 1.1}},
            "confidence thresholds must be between 0 and 1",
        ),
        (
            {"snowflake": {"compute_pool_min_nodes": 3, "compute_pool_max_nodes": 2}},
            "compute_pool_max_nodes must be greater than or equal to min nodes",
        ),
        (
            {"snowflake": {"image_digest": "sha256:not-a-real-digest"}},
            "image_digest must be a sha256:<64 hex chars> OCI digest",
        ),
        (
            {
                "distributed": {
                    "worker_stages": [
                        "extract_entity_window",
                        "extract_entity_window",
                    ]
                }
            },
            "distributed.worker_stages must not contain duplicates",
        ),
        (
            {"distributed": {"finalization_engine": "spark"}},
            "spark finalization requires distributed.artifact_uri",
        ),
    ],
)
def test_settings_rejects_invalid_runtime_bounds(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Settings.load(overrides=overrides)


def test_settings_expose_both_community_bounds_to_the_environment() -> None:
    """A coupled pair of bounds must be reachable together without a config file.

    ``min_community_size`` may not exceed ``max_community_size``, so a deployment
    that only injects environment variables needs both to reach a small maximum.
    """

    settings = Settings.load(
        env={
            "KG_GRAPH_MIN_COMMUNITY_SIZE": "1",
            "KG_GRAPH_MAX_COMMUNITY_SIZE": "1",
        }
    )

    assert settings.graph.min_community_size == 1
    assert settings.graph.max_community_size == 1


def test_settings_accept_friendly_ocr_engine_names_under_either_variable() -> None:
    """Two spellings of the same selector must accept the same vocabulary."""

    by_engine = Settings.load(env={"KG_OCR_ENGINE": "cortex"})
    by_provider = Settings.load(env={"KG_OCR_PROVIDER": "cortex"})

    assert by_engine.ocr.provider == "snowflake_cortex"
    assert by_provider.ocr.provider == "snowflake_cortex"


def test_settings_reject_a_non_positive_ocr_timeout() -> None:
    """The value becomes a subprocess and HTTP deadline that must be reachable."""

    with pytest.raises(ValueError, match="ocr timeout_seconds must be positive"):
        Settings.load(overrides={"ocr": {"timeout_seconds": 0}})


def test_settings_resolve_snowflake_secrets_from_the_variables_they_name() -> None:
    """A configuration that records where a secret lives must produce that secret."""

    settings = Settings.load(
        env={"DEPLOY_SNOWFLAKE_PASSWORD": "s3cret", "DEPLOY_SNOWFLAKE_TOKEN": "t0ken"},
        overrides={
            "snowflake": {
                "password_environment_variable": "DEPLOY_SNOWFLAKE_PASSWORD",
                "oauth_token_environment_variable": "DEPLOY_SNOWFLAKE_TOKEN",
            }
        },
    )

    assert settings.snowflake.password == "s3cret"
    assert settings.snowflake.oauth_token == "t0ken"


def test_settings_keep_a_directly_configured_snowflake_secret() -> None:
    """A named variable is a fallback and never replaces a configured value."""

    settings = Settings.load(
        env={"DEPLOY_SNOWFLAKE_PASSWORD": "from-environment"},
        overrides={
            "snowflake": {
                "password": "from-configuration",
                "password_environment_variable": "DEPLOY_SNOWFLAKE_PASSWORD",
            }
        },
    )

    assert settings.snowflake.password == "from-configuration"


def test_settings_report_a_malformed_configuration_file_as_a_value_error(
    tmp_path: Path,
) -> None:
    """Callers distinguish bad configuration from defects by catching ValueError."""

    config_file = tmp_path / "broken.yaml"
    config_file.write_text("llm:\n  provider: [unclosed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid YAML"):
        Settings.load(config_file, env={})
