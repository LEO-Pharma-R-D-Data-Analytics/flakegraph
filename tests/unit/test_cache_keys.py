from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kg_processor.application.cache_keys import build_extraction_cache_key, build_ocr_cache_key
from kg_processor.application.extraction_contracts import (
    EntityCandidateBatch,
    RelationCandidateBatch,
    ResolutionDecisionCandidateBatch,
    VerificationCandidateBatch,
    extraction_contract_fingerprint,
)
from kg_processor.config.settings import ExtractorSettings, GraphSettings
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


def test_fallback_ocr_cache_key_changes_with_routing_policy() -> None:
    """Invalidate normalized text when fallback quality selection changes."""

    file = _input_file()
    strict = build_ocr_cache_key(
        file,
        "fallback",
        OcrOptions(method="ocr"),
        {"max_fragmented_text_ratio": 0.15},
    )
    permissive = build_ocr_cache_key(
        file,
        "fallback",
        OcrOptions(method="ocr"),
        {"max_fragmented_text_ratio": 0.30},
    )

    assert strict.id != permissive.id


def test_ocr_cache_key_ignores_unrelated_provider_options() -> None:
    file = _input_file()

    first = build_ocr_cache_key(file, "mineru_internal", OcrOptions(method="ocr"))
    second = build_ocr_cache_key(
        file,
        "mineru_internal",
        OcrOptions(
            method="ocr",
            model_cache_dir="/tmp/other-cache",
            tesseract_dpi=450,
            snowflake_parse_mode="LAYOUT",
        ),
    )

    assert first.options_hash == second.options_hash
    assert first.id == second.id


def test_ocr_cache_key_keeps_snowflake_cortex_specific_options() -> None:
    file = _input_file()

    ocr = build_ocr_cache_key(file, "snowflake_cortex", OcrOptions(snowflake_parse_mode="OCR"))
    layout = build_ocr_cache_key(
        file,
        "snowflake_cortex",
        OcrOptions(snowflake_parse_mode="LAYOUT"),
    )

    assert ocr.options_hash != layout.options_hash
    assert ocr.id != layout.id


def test_ocr_cache_key_changes_for_page_range() -> None:
    file = _input_file()

    full_document = build_ocr_cache_key(file, "builtin_text", OcrOptions())
    first_page = build_ocr_cache_key(file, "builtin_text", OcrOptions(page_range="1"))

    assert full_document.options_hash != first_page.options_hash
    assert full_document.id != first_page.id


def test_generic_http_ocr_cache_key_includes_endpoint_and_mapping_but_not_secret() -> None:
    """Invalidate generic OCR output when its service contract changes."""

    file = _input_file()
    first = build_ocr_cache_key(
        file,
        "generic_http",
        OcrOptions(),
        {"endpoint": "https://ocr-a.example", "pages_path": "pages", "api_key": "one"},
    )
    rotated = build_ocr_cache_key(
        file,
        "generic_http",
        OcrOptions(),
        {"endpoint": "https://ocr-a.example", "pages_path": "pages", "api_key": "two"},
    )
    changed = build_ocr_cache_key(
        file,
        "generic_http",
        OcrOptions(),
        {"endpoint": "https://ocr-b.example", "pages_path": "result.pages", "api_key": "two"},
    )

    assert first.id == rotated.id
    assert first.id != changed.id


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
        "kg_processor.application.cache_keys.two_pass_prompt_fingerprints",
        lambda: {
            "entity_extraction": {
                "revision": "entity_extraction@changed",
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


def test_extraction_cache_key_follows_the_contract_the_provider_was_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not serve output produced for a schema the model was never shown.

    The request's schema decides what a provider may return, so a change to it
    changes the extraction. Tracked by a label somebody has to remember to bump,
    a contract change silently reuses the previous model's answers.
    """

    chunk = _chunk()
    settings = GraphSettings()
    key = build_extraction_cache_key("graph", [chunk], "openai_compatible", "model", settings, 120)

    monkeypatch.setattr(
        "kg_processor.application.cache_keys.extraction_contract_fingerprint",
        lambda: "0" * 64,
    )
    after = build_extraction_cache_key(
        "graph", [chunk], "openai_compatible", "model", settings, 120
    )

    assert key.options_hash != after.options_hash
    assert key.id != after.id


def test_the_extraction_contract_fingerprint_is_read_from_the_contracts() -> None:
    """Derive the fingerprint from the schemas, so no one has to remember to bump it.

    Recomputed here from the same models the requests are built from: a
    fingerprint that stopped tracking them would keep serving cached extraction
    across a contract change, which is the failure it exists to prevent.
    """

    expected = hashlib.sha256(
        "|".join(
            json.dumps(model.model_json_schema(), sort_keys=True, separators=(",", ":"))
            for model in (
                EntityCandidateBatch,
                RelationCandidateBatch,
                VerificationCandidateBatch,
                ResolutionDecisionCandidateBatch,
            )
        ).encode("utf-8")
    ).hexdigest()

    assert extraction_contract_fingerprint() == expected


def test_extraction_cache_key_includes_extractor_provider_and_model() -> None:
    """Do not serve LLM extraction output after switching to GLiNER semantics."""

    common = ("graph", [_chunk()], "openai_compatible", "model", GraphSettings(), 120)

    llm = build_extraction_cache_key(
        *common,
        extractor_settings=ExtractorSettings(entity_provider="llm"),
    )
    gliner = build_extraction_cache_key(
        *common,
        extractor_settings=ExtractorSettings(
            entity_provider="gliner",
            gliner_model="urchade/gliner_multi-v2.1",
            gliner_threshold=0.65,
        ),
    )

    assert llm.id != gliner.id


def _input_file() -> InputFile:
    return InputFile(
        id="file_1",
        path=Path("data/martial_arts/files/smoke.txt"),
        source_uri="file:///data/martial_arts/files/smoke.txt",
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
