from __future__ import annotations

from pathlib import Path

import pytest

from kg_processor.config.settings import Settings


def test_settings_defaults_match_local_open_source_runtime_profile() -> None:
    settings = Settings.load(env={})

    assert settings.runtime.runtime == "local"
    assert settings.files.source == "local"
    assert settings.ocr.provider == "mineru_internal"
    assert settings.llm.provider == "openai_compatible"
    assert settings.embedding.provider == "sentence_transformers"
    assert settings.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding.dimension == 384
    assert settings.embedding.batch_size == 32
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


def test_settings_loads_file_queue_runtime_aliases_from_env() -> None:
    settings = Settings.load(
        env={
            "KG_JOB_USE_FILE_QUEUE": "true",
            "KG_WORKER_ID": "worker-1",
            "KG_LEASE_MINUTES": "7",
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


def test_settings_rejects_unknown_ai_backend_profile() -> None:
    with pytest.raises(ValueError, match="KG_AI_BACKEND must be one of"):
        Settings.load(env={"KG_AI_BACKEND": "surprise"})


def test_settings_rejects_unknown_provider_names_before_preflight(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
files:
  source: file-source-that-does-not-exist
  input_path: data/samples
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
            "KG_GENERIC_HTTP_OCR_BLOCK_CONFIDENCE_PATH": "layout.score",
            "KG_GENERIC_HTTP_OCR_BLOCK_METADATA_PATH": "layout.attributes",
            "KG_GENERIC_HTTP_OCR_ASSET_CONFIDENCE_PATH": "media.score",
            "KG_GENERIC_HTTP_OCR_ASSET_METADATA_PATH": "media.details",
        }
    )

    assert settings.generic_http_ocr.block_confidence_path == "layout.score"
    assert settings.generic_http_ocr.block_metadata_path == "layout.attributes"
    assert settings.generic_http_ocr.asset_confidence_path == "media.score"
    assert settings.generic_http_ocr.asset_metadata_path == "media.details"


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
            "KG_MANIFEST_PATH": "data/samples/manifest.jsonl",
        }
    )

    assert settings.files.source == "manifest"
    assert settings.files.manifest_path == Path("data/samples/manifest.jsonl")


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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"embedding": {"batch_size": 0}}, "embedding batch_size must be positive"),
        ({"graph": {"chunk_token_size": 0}}, "chunk_token_size must be positive"),
        ({"graph": {"chunk_token_overlap": -1}}, "chunk_token_overlap must be non-negative"),
        (
            {"graph": {"chunk_token_size": 10, "chunk_token_overlap": 10}},
            "chunk_token_overlap must be smaller than chunk_token_size",
        ),
        ({"graph": {"gleaning_max_passes": -1}}, "gleaning_max_passes must be non-negative"),
        ({"graph": {"relation_weight_max": 0}}, "relation_weight_max must be positive"),
        (
            {"snowflake": {"compute_pool_min_nodes": 3, "compute_pool_max_nodes": 2}},
            "compute_pool_max_nodes must be greater than or equal to min nodes",
        ),
        (
            {"snowflake": {"image_digest": "sha256:not-a-real-digest"}},
            "image_digest must be a sha256:<64 hex chars> OCI digest",
        ),
    ],
)
def test_settings_rejects_invalid_runtime_bounds(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Settings.load(overrides=overrides)
