from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kg_processor.adapters.cache.local_json import LocalJsonCache
from kg_processor.adapters.cache.snowflake import SnowflakeCache
from kg_processor.adapters.embeddings import sentence_transformers as sentence_transformers_module
from kg_processor.adapters.embeddings.azure_openai import AzureOpenAIEmbeddingProvider
from kg_processor.adapters.embeddings.hash import HashEmbeddingProvider
from kg_processor.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from kg_processor.adapters.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)
from kg_processor.adapters.embeddings.snowflake_cortex import SnowflakeCortexEmbeddingProvider
from kg_processor.adapters.files.azure_blob import AzureBlobFileSource
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
from kg_processor.adapters.ocr.snowflake_cortex import SnowflakeCortexOcrProvider
from kg_processor.adapters.ocr.tesseract_internal import TesseractInternalOcrProvider
from kg_processor.adapters.writers.local_artifacts import LocalArtifactsWriter
from kg_processor.adapters.writers.snowflake_bulk import SnowflakeBulkWriter
from kg_processor.config.preflight import run_preflight
from kg_processor.config.settings import (
    CacheSettings,
    EmbeddingSettings,
    FileSettings,
    LlmSettings,
    OcrSettings,
    Settings,
    WriterSettings,
)
from kg_processor.factories import (
    build_cache,
    build_embedding_provider,
    build_file_source,
    build_job_manager,
    build_llm_provider,
    build_ocr_provider,
    build_writer,
)

_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000100ffff030000060005"
    "57bfab3d0000000049454e44ae426082"
)


def test_preflight_accepts_explicit_local_test_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_parent = tmp_path / "out-parent"
    input_dir.mkdir()
    output_parent.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": output_parent / "kg"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert not result.errors


def test_preflight_rejects_unsupported_provider_names(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.model_construct(
        files=FileSettings.model_construct(
            source=cast(Any, "file-source-that-does-not-exist"),
            input_path=input_dir,
            include_globs=["**/*"],
        ),
        ocr=OcrSettings.model_construct(provider="ocr-that-does-not-exist"),
        llm=LlmSettings.model_construct(provider="llm-that-does-not-exist"),
        embedding=EmbeddingSettings.model_construct(
            provider="embedding-that-does-not-exist",
            dimension=384,
            batch_size=32,
        ),
        writer=WriterSettings.model_construct(
            provider="writer-that-does-not-exist",
            output_path=tmp_path / "out",
        ),
        cache=CacheSettings.model_construct(
            provider=cast(Any, "cache-that-does-not-exist"),
            path=tmp_path / "cache",
        ),
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        "Unsupported file_source provider 'file-source-that-does-not-exist'" in error
        for error in result.errors
    )
    assert any(
        "Unsupported ocr provider 'ocr-that-does-not-exist'" in error for error in result.errors
    )
    assert any(
        "Unsupported llm provider 'llm-that-does-not-exist'" in error for error in result.errors
    )
    assert any(
        "Unsupported embedding provider 'embedding-that-does-not-exist'" in error
        for error in result.errors
    )
    assert any(
        "Unsupported writer provider 'writer-that-does-not-exist'" in error
        for error in result.errors
    )
    assert any(
        "Unsupported cache provider 'cache-that-does-not-exist'" in error
        for error in result.errors
    )


def test_preflight_fails_for_missing_mineru(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "mineru_internal", "mineru_command": "definitely-missing-mineru"},
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("MinerU command is not available" in error for error in result.errors)


def test_preflight_rejects_invalid_mineru_internal_remote_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        "kg_processor.config.preflight.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {
                "provider": "mineru_internal",
                "mineru_api_url": "mineru-api:8000",
                "mineru_server_url": "mineru-vlm:30000",
            },
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("mineru_internal api_url must be an http(s) URL" in error for error in result.errors)
    assert any(
        "mineru_internal server_url must be an http(s) URL" in error for error in result.errors
    )


def test_preflight_accepts_valid_mineru_internal_remote_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        "kg_processor.config.preflight.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {
                "provider": "mineru_internal",
                "mineru_api_url": "http://mineru-api:8000",
                "mineru_server_url": "https://mineru-vlm.example",
            },
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert "MinerU internal API endpoint is a valid HTTP URL" in result.checks
    assert "MinerU internal server URL is a valid HTTP URL" in result.checks


def test_preflight_fails_for_missing_mineru_api_url(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "mineru_api"},
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("mineru_api OCR requires mineru_api_url" in error for error in result.errors)


def test_preflight_rejects_invalid_mineru_api_url(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "mineru_api", "mineru_api_url": "mineru.example"},
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("mineru_api OCR endpoint must be an http(s) URL" in error for error in result.errors)


def test_preflight_fails_for_missing_generic_http_ocr_endpoint(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "generic_http"},
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        "generic_http OCR requires generic_http_ocr.endpoint" in error for error in result.errors
    )


def test_preflight_rejects_invalid_generic_http_ocr_endpoint(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "generic_http"},
            "generic_http_ocr": {"endpoint": "ocr.example/parse"},
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        "generic_http OCR endpoint must be an http(s) URL" in error for error in result.errors
    )


def test_preflight_rejects_local_files_not_supported_by_builtin_text(tmp_path: Path) -> None:
    document = tmp_path / "receipt.png"
    document.write_bytes(_TINY_PNG)
    settings = Settings.load(
        overrides={
            "files": {"input_path": document},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        "builtin_text OCR does not support configured file suffixes: .png" in error
        for error in result.errors
    )


def test_preflight_accepts_xlsx_for_builtin_text(tmp_path: Path) -> None:
    document = tmp_path / "spreadsheet.xlsx"
    document.write_bytes(b"preflight only checks local provider suffix compatibility")
    settings = Settings.load(
        overrides={
            "files": {"input_path": document},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert "builtin_text OCR supports all configured local file types" in result.checks


def test_preflight_rejects_zero_based_page_range_for_builtin_text(tmp_path: Path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("Alice works at Acme.", encoding="utf-8")
    settings = Settings.load(
        overrides={
            "files": {"input_path": document},
            "ocr": {"provider": "builtin_text", "page_range": "0"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        "builtin_text page_range uses one-based page numbers" in error
        for error in result.errors
    )


def test_preflight_rejects_manifest_files_not_supported_by_tesseract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("Alice works at Acme.", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"path":"notes.txt"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "kg_processor.config.preflight.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    settings = Settings.load(
        overrides={
            "files": {"source": "manifest", "manifest_path": manifest},
            "ocr": {
                "provider": "tesseract_internal",
                "tesseract_command": "tesseract",
                "tesseract_pdf_renderer_command": "pdftoppm",
            },
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        "tesseract_internal OCR does not support configured file suffixes: .txt" in error
        for error in result.errors
    )


def test_preflight_accepts_tesseract_image_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "receipt.png"
    image.write_bytes(_TINY_PNG)
    monkeypatch.setattr(
        "kg_processor.config.preflight.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": image},
            "ocr": {
                "provider": "tesseract_internal",
                "tesseract_command": "tesseract",
                "tesseract_pdf_renderer_command": "pdftoppm",
            },
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert "tesseract_internal OCR supports all configured local file types" in result.checks


@pytest.mark.parametrize("provider", ["mineru_internal", "mineru_api"])
def test_preflight_rejects_local_files_not_supported_by_mineru(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    (input_dir / "page.html").write_text("<p>Not a MinerU input.</p>", encoding="utf-8")
    (input_dir / "notes.txt").write_text("Plain text is handled by builtin_text.", encoding="utf-8")
    monkeypatch.setattr(
        "kg_processor.config.preflight.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    ocr_settings: dict[str, str] = {"provider": provider}
    if provider == "mineru_api":
        ocr_settings["mineru_api_url"] = "https://mineru.example"
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir, "include_globs": ["**/*"]},
            "ocr": ocr_settings,
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        f"{provider} OCR does not support configured file suffixes: .html, .txt" in error
        for error in result.errors
    )


def test_preflight_rejects_tesseract_image_page_range_that_excludes_page_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "receipt.png"
    image.write_bytes(_TINY_PNG)
    monkeypatch.setattr(
        "kg_processor.config.preflight.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": image},
            "ocr": {
                "provider": "tesseract_internal",
                "page_range": "2",
                "tesseract_command": "tesseract",
                "tesseract_pdf_renderer_command": "pdftoppm",
            },
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        "tesseract_internal image OCR only supports page 1" in error for error in result.errors
    )


def test_preflight_rejects_mineru_page_window_with_end_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        "kg_processor.config.preflight.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {
                "provider": "mineru_internal",
                "mineru_start_page_id": 5,
                "mineru_end_page_id": 3,
            },
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        "mineru_end_page_id is before mineru_start_page_id" in error
        for error in result.errors
    )


def test_preflight_rejects_snowflake_cortex_page_range_with_end_before_start() -> None:
    settings = Settings.load(
        overrides={
            "files": {"source": "snowflake_stage"},
            "ocr": {"provider": "snowflake_cortex", "page_range": "3-1"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"provider": "local_artifacts"},
            "snowflake": {
                "account": "account",
                "database": "DB",
                "schema": "GRAPH",
                "stage": "@DB.GRAPH.DOCS",
            },
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        "snowflake_cortex page_range '3-1': end page is before start page" in error
        for error in result.errors
    )


def test_preflight_rejects_azure_openai_llm_without_model_or_api_version(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {
                "provider": "azure_openai",
                "endpoint": "https://azure.example/openai/deployments/chat",
                "api_key": "secret",
                "model": "",
                "api_version": "",
            },
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert "azure_openai LLM requires llm.model" in result.errors
    assert "azure_openai LLM requires llm.api_version" in result.errors


def test_preflight_rejects_azure_openai_embedding_without_model_or_api_version(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {
                "provider": "azure_openai",
                "endpoint": "https://azure.example/openai/deployments/embed",
                "api_key": "secret",
                "model": "",
                "api_version": "",
            },
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert "azure_openai embedding requires embedding.model" in result.errors
    assert "azure_openai embedding requires embedding.api_version" in result.errors


def test_preflight_rejects_openai_compatible_external_providers_without_credentials(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "openai_compatible", "endpoint": "", "api_key": ""},
            "embedding": {"provider": "openai_compatible", "endpoint": "", "api_key": ""},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert "openai_compatible LLM requires endpoint and API key" in result.errors
    assert "openai_compatible embedding requires endpoint and API key" in result.errors


def test_factories_build_explicit_openai_compatible_external_providers(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {
                "provider": "openai_compatible",
                "endpoint": "https://llm.example/v1",
                "api_key": "secret",
                "model": "gpt-4.1-mini",
            },
            "embedding": {
                "provider": "openai_compatible",
                "endpoint": "https://embed.example/v1",
                "api_key": "secret",
                "model": "text-embedding-3-small",
            },
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert isinstance(build_llm_provider(settings), OpenAICompatibleLlmProvider)
    assert isinstance(build_embedding_provider(settings), OpenAICompatibleEmbeddingProvider)


def test_factories_build_explicit_azure_openai_external_providers(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {
                "provider": "azure_openai",
                "endpoint": "https://azure.example",
                "api_key": "secret",
                "model": "gpt-4.1-mini-2025-04-14",
                "api_version": "2025-01-01-preview",
            },
            "embedding": {
                "provider": "azure_openai",
                "endpoint": "https://azure.example",
                "api_key": "secret",
                "model": "text-embedding-3-small",
                "api_version": "2025-01-01-preview",
            },
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert isinstance(build_llm_provider(settings), AzureOpenAILlmProvider)
    assert isinstance(build_embedding_provider(settings), AzureOpenAIEmbeddingProvider)


def test_preflight_rejects_snowflake_cortex_providers_without_models() -> None:
    settings = Settings.load(
        overrides={
            "files": {"source": "snowflake_stage"},
            "ocr": {"provider": "snowflake_cortex"},
            "llm": {"provider": "snowflake_cortex", "model": ""},
            "embedding": {"provider": "snowflake_cortex", "model": ""},
            "writer": {"provider": "snowflake_bulk"},
            "cache": {"provider": "snowflake"},
            "snowflake": {
                "account": "account",
                "database": "DB",
                "schema": "GRAPH",
                "stage": "@DB.GRAPH.DOCS",
                "bulk_stage": "@DB.GRAPH.LOAD",
            },
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert "snowflake_cortex LLM requires llm.model" in result.errors
    assert "snowflake_cortex embedding requires embedding.model" in result.errors


def test_factories_build_generic_http_ocr_provider(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "generic_http"},
            "generic_http_ocr": {"endpoint": "https://ocr.example/parse"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert any("Generic HTTP OCR endpoint is configured" in check for check in result.checks)
    assert isinstance(build_ocr_provider(settings), GenericHttpOcrProvider)


def test_preflight_fails_for_missing_tesseract_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr("kg_processor.config.preflight.shutil.which", lambda _command: None)
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {
                "provider": "tesseract_internal",
                "tesseract_command": "missing-tesseract",
                "tesseract_pdf_renderer_command": "missing-pdftoppm",
            },
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("Tesseract command is not available" in error for error in result.errors)
    assert any("PDF renderer command is not available" in error for error in result.errors)


def test_factories_build_tesseract_internal_ocr_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        "kg_processor.config.preflight.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {
                "provider": "tesseract_internal",
                "tesseract_command": "tesseract",
                "tesseract_pdf_renderer_command": "pdftoppm",
                "tesseract_dpi": 200,
            },
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert any("Tesseract command is available" in check for check in result.checks)
    assert any("PDF renderer command is available" in check for check in result.checks)
    provider = build_ocr_provider(settings)
    assert isinstance(provider, TesseractInternalOcrProvider)
    assert provider.dpi == 200


def test_factories_build_explicit_local_providers(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )

    assert build_file_source(settings).list_files() == []
    assert isinstance(build_ocr_provider(settings), BuiltinTextOcrProvider)
    assert isinstance(build_llm_provider(settings), FakeLlmProvider)
    assert isinstance(build_embedding_provider(settings), HashEmbeddingProvider)
    assert isinstance(build_writer(settings), LocalArtifactsWriter)
    assert build_cache(settings) is None


def test_factories_build_manifest_file_source(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("Alice", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"path":"sample.txt"}\n', encoding="utf-8")
    settings = Settings.load(
        overrides={
            "files": {"source": "manifest", "manifest_path": manifest},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert any("Manifest file exists" in check for check in result.checks)
    source = build_file_source(settings)
    assert isinstance(source, ManifestFileSource)
    assert source.list_files()[0].path == document.resolve()


def test_preflight_rejects_missing_manifest_file(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "files": {"source": "manifest", "manifest_path": tmp_path / "missing.jsonl"},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("files.manifest_path" in error for error in result.errors)


def test_preflight_accepts_azure_blob_configuration(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "files": {"source": "azure_blob"},
            "azure_blob": {
                "account_url": "https://storage.example",
                "container": "documents",
                "prefix": "incoming",
                "sas_token": "sas",
                "download_path": tmp_path / "downloads",
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert any("Azure Blob container is configured" in check for check in result.checks)
    assert any("Azure Blob credentials are configured" in check for check in result.checks)


def test_preflight_rejects_azure_blob_without_credentials(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "files": {"source": "azure_blob"},
            "azure_blob": {"container": "documents", "download_path": tmp_path / "downloads"},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("account_url+sas_token" in error for error in result.errors)


def test_factories_build_azure_blob_file_source(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "files": {"source": "azure_blob"},
            "azure_blob": {
                "account_url": "https://storage.example",
                "container": "documents",
                "sas_token": "sas",
                "download_path": tmp_path / "downloads",
            },
        }
    )

    assert isinstance(build_file_source(settings), AzureBlobFileSource)


def test_preflight_rejects_missing_sentence_transformers_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        "kg_processor.config.preflight.importlib.util.find_spec",
        lambda name: None if name == "sentence_transformers" else object(),
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "sentence_transformers"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("local-embeddings extra" in error for error in result.errors)


def test_preflight_accepts_available_sentence_transformers_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        "kg_processor.config.preflight.importlib.util.find_spec",
        lambda name: object() if name == "sentence_transformers" else None,
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "sentence_transformers"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert any("sentence_transformers package is installed" in check for check in result.checks)


def test_preflight_rejects_sentence_transformers_cuda_when_torch_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        "kg_processor.config.preflight.importlib.util.find_spec",
        lambda name: object() if name == "sentence_transformers" else None,
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "sentence_transformers", "device": "cuda"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("requires CUDA but CUDA is not available" in error for error in result.errors)


def test_preflight_rejects_sentence_transformers_cuda_when_cuda_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        "kg_processor.config.preflight.importlib.util.find_spec",
        lambda name: object() if name in {"sentence_transformers", "torch"} else None,
    )
    monkeypatch.setattr(
        "kg_processor.config.preflight.importlib.import_module",
        lambda name: SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        )
        if name == "torch"
        else None,
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "sentence_transformers", "device": "cuda:0"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("requires CUDA but CUDA is not available" in error for error in result.errors)


def test_preflight_rejects_sentence_transformers_cuda_index_that_does_not_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        "kg_processor.config.preflight.importlib.util.find_spec",
        lambda name: object() if name in {"sentence_transformers", "torch"} else None,
    )
    monkeypatch.setattr(
        "kg_processor.config.preflight.importlib.import_module",
        lambda name: SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        )
        if name == "torch"
        else None,
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "sentence_transformers", "device": "cuda:2"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("requires CUDA but CUDA is not available" in error for error in result.errors)


def test_preflight_accepts_sentence_transformers_cuda_when_torch_cuda_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    monkeypatch.setattr(
        "kg_processor.config.preflight.importlib.util.find_spec",
        lambda name: object() if name in {"sentence_transformers", "torch"} else None,
    )
    monkeypatch.setattr(
        "kg_processor.config.preflight.importlib.import_module",
        lambda name: SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
        )
        if name == "torch"
        else None,
    )
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "sentence_transformers", "device": "cuda:1"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert any("CUDA embedding device is available: cuda:1" in check for check in result.checks)


def test_factories_build_sentence_transformers_embedding_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sentence_transformers_module,
        "_load_sentence_transformer_factory",
        lambda: lambda *_args, **_kwargs: _FakeSentenceTransformerModel(),
    )
    settings = Settings.load(
        overrides={
            "embedding": {
                "provider": "sentence_transformers",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
            }
        }
    )

    provider = build_embedding_provider(settings)

    assert isinstance(provider, SentenceTransformersEmbeddingProvider)


def test_preflight_accepts_local_cache_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    cache_parent = tmp_path / "cache-parent"
    input_dir.mkdir()
    cache_parent.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
            "cache": {"provider": "local", "path": cache_parent / "kg-cache"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert any("Cache path can be created" in check for check in result.checks)


def test_factories_build_local_cache(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "cache": {"provider": "local", "path": tmp_path / "cache"},
        }
    )

    assert isinstance(build_cache(settings), LocalJsonCache)


def test_factories_build_explicit_mineru_api_provider(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {
                "provider": "mineru_api",
                "mineru_api_url": "https://mineru.example",
                "mineru_api_key": "secret",
            },
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
        }
    )

    assert isinstance(build_ocr_provider(settings), MineruApiOcrProvider)


def test_preflight_accepts_vllm_local_without_api_key(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "vllm_local", "endpoint": "http://localhost:8000/v1"},
            "embedding": {"provider": "hash"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert any("vllm_local LLM endpoint is configured" in check for check in result.checks)
    assert isinstance(build_llm_provider(settings), VllmLocalLlmProvider)


def test_preflight_rejects_vllm_local_without_endpoint(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "vllm_local"},
            "writer": {"output_path": tmp_path / "out"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("vllm_local LLM requires endpoint" in error for error in result.errors)


def test_preflight_accepts_explicit_snowflake_cortex_configuration(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "files": {"source": "snowflake_stage", "stage_prefix": "input"},
            "ocr": {"provider": "snowflake_cortex", "snowflake_parse_mode": "layout"},
            "llm": {"provider": "snowflake_cortex", "model": "llama3.3-70b"},
            "embedding": {
                "provider": "snowflake_cortex",
                "model": "snowflake-arctic-embed-m-v1.5",
                "dimension": 768,
            },
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
            "snowflake": {
                "account": "account",
                "database": "DB",
                "schema": "SCHEMA",
                "stage": "@DB.SCHEMA.DOC_STAGE",
                "bulk_stage": "@DB.SCHEMA.KG_LOAD_STAGE",
            },
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert settings.ocr.snowflake_parse_mode == "LAYOUT"


def test_preflight_accepts_spcs_snowflake_native_provider_set() -> None:
    settings = Settings.load(
        overrides={
            "runtime": {"runtime": "spcs"},
            "files": {"source": "snowflake_stage", "stage_prefix": "input"},
            "ocr": {"provider": "snowflake_cortex", "snowflake_parse_mode": "layout"},
            "llm": {"provider": "snowflake_cortex", "model": "llama3.3-70b"},
            "embedding": {
                "provider": "snowflake_cortex",
                "model": "snowflake-arctic-embed-m-v1.5",
                "dimension": 768,
            },
            "writer": {"provider": "snowflake_bulk"},
            "cache": {"provider": "snowflake"},
            "snowflake": {
                "account": "account",
                "database": "DB",
                "schema": "SCHEMA",
                "role": "KG_ROLE",
                "warehouse": "KG_WH",
                "stage": "@DB.SCHEMA.DOC_STAGE",
                "bulk_stage": "@DB.SCHEMA.KG_LOAD_STAGE",
                "compute_pool": "KG_POOL",
                "image_repository": "DB.SCHEMA.KG_REPO",
                "service_spec_stage": "@DB.SCHEMA.KG_SERVICE_SPECS",
            },
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert not result.errors
    assert "SPCS runtime uses staged Snowflake input" in result.checks
    assert "SPCS runtime uses Snowflake cache tables" in result.checks
    assert "SPCS runtime uses a dedicated service-compatible compute pool" in result.checks


def test_preflight_rejects_spcs_system_compute_pool() -> None:
    settings = Settings.load(
        overrides={
            "runtime": {"runtime": "spcs"},
            "files": {"source": "snowflake_stage", "stage_prefix": "input"},
            "ocr": {"provider": "snowflake_cortex"},
            "llm": {"provider": "snowflake_cortex", "model": "llama3.3-70b"},
            "embedding": {
                "provider": "snowflake_cortex",
                "model": "snowflake-arctic-embed-m-v1.5",
                "dimension": 768,
            },
            "writer": {"provider": "snowflake_bulk"},
            "cache": {"provider": "snowflake"},
            "snowflake": {
                "account": "account",
                "database": "DB",
                "schema": "SCHEMA",
                "role": "KG_ROLE",
                "warehouse": "KG_WH",
                "stage": "@DB.SCHEMA.DOC_STAGE",
                "bulk_stage": "@DB.SCHEMA.KG_LOAD_STAGE",
                "compute_pool": "SYSTEM_COMPUTE_POOL_CPU",
                "image_repository": "DB.SCHEMA.KG_REPO",
                "service_spec_stage": "@DB.SCHEMA.KG_SERVICE_SPECS",
            },
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert (
        "SPCS runtime requires a dedicated compute pool; Snowflake SYSTEM_COMPUTE_POOL_* "
        "pools reject FlakeGraph EXECUTE JOB SERVICE workloads"
    ) in result.errors


def test_preflight_rejects_spcs_runtime_with_local_provider_set(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "runtime": {"runtime": "spcs"},
            "files": {"source": "local", "input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash", "dimension": 64},
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
            "cache": {"provider": "none"},
            "snowflake": {
                "account": "account",
                "database": "DB",
                "schema": "SCHEMA",
                "role": "KG_ROLE",
                "warehouse": "KG_WH",
                "stage": "@DB.SCHEMA.DOC_STAGE",
                "bulk_stage": "@DB.SCHEMA.KG_LOAD_STAGE",
                "compute_pool": "KG_POOL",
                "image_repository": "DB.SCHEMA.KG_REPO",
                "service_spec_stage": "@DB.SCHEMA.KG_SERVICE_SPECS",
            },
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert "SPCS runtime requires files.source=snowflake_stage" in result.errors
    assert "SPCS runtime requires ocr.provider=snowflake_cortex" in result.errors
    assert "SPCS runtime requires llm.provider=snowflake_cortex" in result.errors
    assert "SPCS runtime requires embedding.provider=snowflake_cortex" in result.errors
    assert (
        "SPCS runtime requires writer.provider=snowflake_direct or snowflake_bulk"
        in result.errors
    )
    assert "SPCS runtime requires cache.provider=snowflake" in result.errors


def test_preflight_rejects_snowflake_cortex_ocr_without_stage_source(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"source": "local", "input_path": input_dir},
            "ocr": {"provider": "snowflake_cortex"},
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("requires files.source=snowflake_stage" in error for error in result.errors)


def test_factories_build_explicit_snowflake_providers(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "files": {"source": "snowflake_stage", "stage_prefix": "input"},
            "ocr": {"provider": "snowflake_cortex"},
            "llm": {"provider": "snowflake_cortex", "model": "llama3.3-70b"},
            "embedding": {
                "provider": "snowflake_cortex",
                "model": "snowflake-arctic-embed-m-v1.5",
                "dimension": 768,
            },
            "writer": {"provider": "local_artifacts", "output_path": tmp_path / "out"},
            "snowflake": {
                "account": "account",
                "database": "DB",
                "schema": "SCHEMA",
                "stage": "@DB.SCHEMA.DOC_STAGE",
            },
        }
    )

    assert isinstance(build_file_source(settings), SnowflakeStageFileSource)
    assert isinstance(build_ocr_provider(settings), SnowflakeCortexOcrProvider)
    assert isinstance(build_llm_provider(settings), SnowflakeCortexLlmProvider)
    assert isinstance(build_embedding_provider(settings), SnowflakeCortexEmbeddingProvider)


def test_factories_build_snowflake_cache(tmp_path: Path) -> None:
    settings = Settings.load(
        overrides={
            "cache": {"provider": "snowflake"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )

    assert isinstance(build_cache(settings), SnowflakeCache)


def test_preflight_rejects_snowflake_bulk_without_bulk_stage(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "writer": {"provider": "snowflake_bulk"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any(
        "snowflake_bulk writer requires snowflake.bulk_stage" in error for error in result.errors
    )


def test_factories_build_explicit_snowflake_bulk_writer(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash", "dimension": 64},
            "writer": {"provider": "snowflake_bulk"},
            "snowflake": {
                "account": "account",
                "database": "DB",
                "schema": "SCHEMA",
                "bulk_stage": "@DB.SCHEMA.KG_LOAD_STAGE",
                "bulk_target_file_size_mb": 64,
            },
        }
    )

    writer = build_writer(settings)

    assert isinstance(writer, SnowflakeBulkWriter)
    assert writer.target_file_size_bytes == 64 * 1024 * 1024


def test_preflight_rejects_job_lease_without_owner(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "job": {"use_lease": True},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("job.use_lease requires job.lease_owner" in error for error in result.errors)


def test_preflight_rejects_file_queue_without_owner(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    settings = Settings.load(
        overrides={
            "job": {"use_file_queue": True},
            "files": {"input_path": input_dir},
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "writer": {"output_path": tmp_path / "out"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )

    result = run_preflight(settings)

    assert not result.ok
    assert any("job.use_file_queue requires job.lease_owner" in error for error in result.errors)


def test_factories_build_job_manager_when_lease_enabled() -> None:
    settings = Settings.load(
        overrides={
            "job": {"use_lease": True, "lease_owner": "worker_1"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )

    assert isinstance(build_job_manager(settings), SnowflakeJobManager)


def test_factories_build_job_manager_when_file_queue_enabled() -> None:
    settings = Settings.load(
        overrides={
            "job": {
                "use_file_queue": True,
                "lease_owner": "worker_1",
                "file_batch_size": 25,
            },
            "ocr": {"provider": "builtin_text"},
            "llm": {"provider": "fake"},
            "embedding": {"provider": "hash"},
            "snowflake": {"account": "account", "database": "DB", "schema": "SCHEMA"},
        }
    )

    result = run_preflight(settings)

    assert result.ok
    assert isinstance(build_job_manager(settings), SnowflakeJobManager)


class _FakeSentenceTransformerModel:
    def encode(self, *_args: object, **_kwargs: object) -> list[list[float]]:
        return [[0.1, 0.2, 0.3]]
