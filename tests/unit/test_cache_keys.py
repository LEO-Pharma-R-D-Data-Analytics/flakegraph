from __future__ import annotations

from pathlib import Path

import pytest

from kg_processor.application.cache_keys import build_extraction_cache_key, build_ocr_cache_key
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.documents import InputFile
from kg_processor.domain.graph import Chunk
from kg_processor.ports.ocr import OcrOptions


def test_ocr_cache_key_excludes_secret_api_key() -> None:
    file = _input_file()

    first = build_ocr_cache_key(
        file,
        "mineru_internal",
        OcrOptions(method="ocr", api_url="https://mineru.example", api_key="first-secret"),
    )
    second = build_ocr_cache_key(
        file,
        "mineru_internal",
        OcrOptions(method="ocr", api_url="https://mineru.example", api_key="rotated-secret"),
    )

    assert first.options_hash == second.options_hash
    assert first.id == second.id


def test_ocr_cache_key_changes_for_behavioral_options() -> None:
    file = _input_file()

    text_key = build_ocr_cache_key(file, "mineru_internal", OcrOptions(method="txt"))
    ocr_key = build_ocr_cache_key(file, "mineru_internal", OcrOptions(method="ocr"))

    assert text_key.options_hash != ocr_key.options_hash
    assert text_key.id != ocr_key.id


def test_ocr_cache_key_changes_for_page_range() -> None:
    file = _input_file()

    full_document = build_ocr_cache_key(file, "builtin_text", OcrOptions())
    first_page = build_ocr_cache_key(file, "builtin_text", OcrOptions(page_range="1"))

    assert full_document.options_hash != first_page.options_hash
    assert full_document.id != first_page.id


def test_extraction_cache_key_changes_for_prompt_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk()
    settings = GraphSettings()
    first = build_extraction_cache_key(
        "graph",
        [chunk],
        "openai_compatible",
        "model",
        settings,
        120,
    )

    monkeypatch.setattr(
        "kg_processor.application.cache_keys.prompt_fingerprints",
        lambda _names: {
            "graph_extraction": {
                "revision": "graph_extraction@changed",
                "checksum": "0" * 64,
            }
        },
    )
    second = build_extraction_cache_key(
        "graph",
        [chunk],
        "openai_compatible",
        "model",
        settings,
        120,
    )

    assert first.options_hash != second.options_hash
    assert first.id != second.id


def _input_file() -> InputFile:
    return InputFile(
        id="file_1",
        path=Path("data/samples/smoke.txt"),
        source_uri="file:///data/samples/smoke.txt",
        checksum="abc123",
        mime_type="text/plain",
        size_bytes=42,
    )


def _chunk() -> Chunk:
    return Chunk(
        id="chunk_1",
        graph_id="graph",
        file_id="file_1",
        document_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith works at Acme Corp.",
        start_offset=0,
        end_offset=31,
        token_count=6,
        content_hash="hash",
    )
