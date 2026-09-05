"""Behavioral tests for provider-neutral OCR fallback routing."""

from __future__ import annotations

from pathlib import Path

import pytest

from kg_processor.adapters.ocr.fallback import FallbackOcrProvider
from kg_processor.domain.documents import InputFile, ParsedDocument, ParsedPage
from kg_processor.ports.ocr import OcrOptions


class _StubOcr:
    """Return a configured document or raise a configured error for routing tests."""

    def __init__(
        self,
        document: ParsedDocument | None = None,
        error: Exception | None = None,
    ) -> None:
        self.document = document
        self.error = error
        self.calls = 0
        self.closed = 0

    def parse(self, _file: InputFile, _options: OcrOptions) -> ParsedDocument:
        """Record invocation before returning or raising the configured result."""

        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.document is not None
        return self.document

    def close(self) -> None:
        """Record the release of this provider's retained transport resources."""

        self.closed += 1


def test_fallback_releases_both_wrapped_providers(tmp_path: Path) -> None:
    """A worker builds one pipeline per batch; unreleased pools accumulate.

    The pipeline closes the OCR provider it holds, which is this composition, so
    the wrapped providers are only reached through it.
    """

    del tmp_path
    primary = _StubOcr(_document("primary", provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    _provider(primary, secondary, minimum=80).close()

    assert (primary.closed, secondary.closed) == (1, 1)


def test_fallback_close_tolerates_providers_without_transport_state(tmp_path: Path) -> None:
    """Not every OCR adapter retains resources, and none is required to."""

    del tmp_path

    class _Closeless:
        def parse(self, _file: InputFile, _options: OcrOptions) -> ParsedDocument:
            raise AssertionError("parse is not exercised by this test")

    provider = FallbackOcrProvider(
        _Closeless(),
        _Closeless(),
        primary_name="builtin_text",
        secondary_name="mineru_internal",
        min_characters_per_page=80,
        max_sparse_page_ratio=0.2,
        max_unbroken_text_ratio=0.5,
        max_fragmented_text_ratio=0.15,
    )

    provider.close()


def test_fallback_close_reports_every_provider_failure(tmp_path: Path) -> None:
    """One provider's failure must not hide the other's, or skip its release."""

    del tmp_path

    class _FailingClose(_StubOcr):
        def close(self) -> None:
            super().close()
            raise RuntimeError("connection pool already shut down")

    primary = _FailingClose(_document("primary", provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    with pytest.raises(ExceptionGroup):
        _provider(primary, secondary, minimum=80).close()

    assert secondary.closed == 1


def test_fallback_keeps_sufficient_primary_text_without_secondary_call(tmp_path: Path) -> None:
    """Avoid expensive OCR when the primary parser returned useful page text."""

    primary = _StubOcr(_document("useful primary text " * 12, provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))
    provider = _provider(primary, secondary, minimum=80)

    result = provider.parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 0
    assert result.provider_metadata["fallback_ocr"]["selected"] == "builtin_text"
    assert result.provider_metadata["fallback_ocr"]["reason"] == "primary_text_sufficient"


def test_fallback_routes_sparse_primary_document_to_secondary(tmp_path: Path) -> None:
    """Run real OCR when average page text falls below the configured threshold."""

    primary = _StubOcr(_document("scan", provider="builtin_text", pages=2))
    secondary = _StubOcr(_document("recognized text", provider="mineru_internal"))
    provider = _provider(primary, secondary, minimum=80)

    result = provider.parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 1
    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "mineru_internal"
    assert routing["reason"] == "primary_text_insufficient"
    assert routing["primary_character_count"] == 8
    assert routing["required_character_count"] == 160


def test_fallback_routes_primary_error_without_leaking_message(tmp_path: Path) -> None:
    """Keep fallback useful on parser errors while excluding sensitive error text."""

    primary = _StubOcr(error=RuntimeError("private /path and endpoint"))
    secondary = _StubOcr(_document("recognized text", provider="mineru_internal"))
    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert routing["reason"] == "primary_error"
    assert routing["primary_error_type"] == "RuntimeError"
    assert "private" not in str(routing)


def test_fallback_routes_mixed_document_with_too_many_sparse_pages(tmp_path: Path) -> None:
    """Prevent one dense native page from hiding a scanned or blank companion page."""

    primary = _StubOcr(_document_with_pages(["a" * 500, ""]))
    secondary = _StubOcr(_document("recognized text", provider="mineru_internal"))
    provider = _provider(primary, secondary, minimum=80)

    result = provider.parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "mineru_internal"
    assert routing["sparse_page_count"] == 1
    assert routing["sparse_page_ratio"] == 0.5


def test_fallback_routes_dense_but_concatenated_latin_text(tmp_path: Path) -> None:
    """Reject malformed PDF text layers whose word boundaries have disappeared."""

    malformed = "A" * 120 + " " + "B" * 120
    primary = _StubOcr(_document(malformed, provider="builtin_text"))
    secondary = _StubOcr(_document("recognized prose", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "mineru_internal"
    assert routing["reason"] == "primary_text_malformed"
    assert routing["unbroken_text_ratio"] == 1.0


def test_fallback_does_not_apply_latin_spacing_heuristic_to_cjk(tmp_path: Path) -> None:
    """Keep naturally unsegmented non-Latin text on the sufficient primary path."""

    primary = _StubOcr(_document("漢字" * 100, provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 0
    assert result.provider_metadata["fallback_ocr"]["selected"] == "builtin_text"


def test_fallback_routes_dense_but_fragmented_prose(tmp_path: Path) -> None:
    """Reject legacy text layers that insert spaces inside most prose words."""

    malformed = "The metho d com bines optimiz ation fea tures. " * 20
    primary = _StubOcr(_document(malformed, provider="builtin_text"))
    secondary = _StubOcr(_document("recognized prose", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    routing = result.provider_metadata["fallback_ocr"]
    assert routing["selected"] == "mineru_internal"
    assert routing["reason"] == "primary_text_malformed"
    assert routing["fragmented_text_ratio"] > 0.15


def test_fallback_ignores_latex_variables_when_measuring_fragmentation(tmp_path: Path) -> None:
    """Keep coherent prose with dense one-letter mathematical variables."""

    text = "Useful primary prose explains the model. " * 8 + "$ x y z = a b c $ " * 30
    primary = _StubOcr(_document(text, provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))

    result = _provider(primary, secondary, minimum=80).parse(_file(tmp_path), OcrOptions())

    assert secondary.calls == 0
    assert result.provider_metadata["fallback_ocr"]["selected"] == "builtin_text"


def test_fallback_keeps_primary_when_secondary_cannot_parse_file_type(tmp_path: Path) -> None:
    """Let mixed corpora retain sparse native text instead of invoking incompatible OCR."""

    primary = _StubOcr(_document("short", provider="builtin_text"))
    secondary = _StubOcr(_document("secondary", provider="mineru_internal"))
    provider = FallbackOcrProvider(
        primary,
        secondary,
        primary_name="builtin_text",
        secondary_name="mineru_internal",
        min_characters_per_page=80,
        max_sparse_page_ratio=0.2,
        max_unbroken_text_ratio=0.5,
        max_fragmented_text_ratio=0.15,
        secondary_supported_suffixes=frozenset({".pdf"}),
    )

    result = provider.parse(_file(tmp_path, suffix=".txt"), OcrOptions())

    assert secondary.calls == 0
    assert result.provider_metadata["fallback_ocr"]["selected"] == "builtin_text"
    assert result.provider_metadata["fallback_ocr"]["reason"] == "secondary_file_type_unsupported"


def _provider(primary: _StubOcr, secondary: _StubOcr, minimum: int) -> FallbackOcrProvider:
    """Construct the adapter under test with stable provider identities."""

    return FallbackOcrProvider(
        primary,
        secondary,
        primary_name="builtin_text",
        secondary_name="mineru_internal",
        min_characters_per_page=minimum,
        max_sparse_page_ratio=0.2,
        max_unbroken_text_ratio=0.5,
        max_fragmented_text_ratio=0.15,
    )


def _document(text: str, *, provider: str, pages: int = 1) -> ParsedDocument:
    """Build a minimal normalized document with repeated deterministic pages."""

    return ParsedDocument(
        file_id="file-1",
        checksum="abc",
        source_uri="file:///fixture.pdf",
        mime_type="application/pdf",
        pages=[
            ParsedPage(page_number=index, markdown=text, raw_text=text, blocks=[])
            for index in range(1, pages + 1)
        ],
        provider_metadata={"provider": provider},
    )


def _document_with_pages(texts: list[str]) -> ParsedDocument:
    """Build a primary document with independently controlled page density."""

    return ParsedDocument(
        file_id="file-1",
        checksum="abc",
        source_uri="file:///fixture.pdf",
        mime_type="application/pdf",
        pages=[
            ParsedPage(page_number=index, markdown=text, raw_text=text, blocks=[])
            for index, text in enumerate(texts, start=1)
        ],
        provider_metadata={"provider": "builtin_text"},
    )


def _file(tmp_path: Path, suffix: str = ".pdf") -> InputFile:
    """Create the input-file value required by the OCR port."""

    path = tmp_path / f"fixture{suffix}"
    path.write_bytes(b"%PDF-fixture")
    return InputFile(
        id="file-1",
        source_uri=path.as_uri(),
        path=path,
        checksum="abc",
        mime_type="application/pdf",
        size_bytes=12,
    )
