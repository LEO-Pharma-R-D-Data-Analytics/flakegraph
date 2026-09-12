from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from kg_processor.adapters.ocr.generic_http import GenericHttpOcrProvider
from kg_processor.config.settings import GenericHttpOcrSettings
from kg_processor.domain.documents import InputFile
from kg_processor.ports.ocr import OcrOptions


class FakeClient:
    requests: list[dict[str, Any]] = []
    response_payload: dict[str, object] = {}
    response_content: bytes | None = None
    response_headers: dict[str, str] = {}

    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @contextmanager
    def stream(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        data: dict[str, str],
        files: dict[str, object],
    ) -> Iterator[httpx.Response]:
        assert method == "POST"
        FakeClient.requests.append(
            {
                "url": url,
                "headers": headers,
                "data": data,
                "files": files,
            }
        )
        request = httpx.Request("POST", url, headers=headers)
        if FakeClient.response_content is not None:
            yield httpx.Response(
                200,
                content=FakeClient.response_content,
                headers=FakeClient.response_headers,
                request=request,
            )
            return
        yield httpx.Response(200, json=FakeClient.response_payload, request=request)


def test_generic_http_ocr_posts_file_and_maps_pages_blocks_and_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_content = None
    FakeClient.response_headers = {}
    FakeClient.response_payload = {
        "payload": {
            "state": "done",
            "pages": [
                {
                    "page": "2",
                    "md": "# Page two",
                    "text": "Page two",
                    "lang": "en",
                    "layout": [
                        {
                            "block_id": "b1",
                            "type": "heading",
                            "content": "Page two",
                            "box": [0, 1, 2, 3],
                            "score": "0.94",
                            "attributes": {"font": "bold"},
                        }
                    ],
                }
            ],
            "media": [
                {
                    "asset_id": "img1",
                    "asset_type": "image",
                    "href": "https://ocr.example/assets/img1.png",
                    "page": "2",
                    "score": 0.81,
                    "details": {"width": 640, "height": 480},
                }
            ],
            "warnings": ["low-confidence"],
        }
    }
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = GenericHttpOcrProvider(
        GenericHttpOcrSettings(
            endpoint="https://ocr.example/parse",
            api_key="secret",
            file_field="document",
            result_path="payload",
            status_path="state",
            page_number_path="page",
            markdown_path="md",
            raw_text_path="text",
            detected_language_path="lang",
            blocks_path="layout",
            block_id_path="block_id",
            block_kind_path="type",
            block_text_path="content",
            block_bbox_path="box",
            block_confidence_path="score",
            block_metadata_path="attributes",
            assets_path="media",
            asset_id_path="asset_id",
            asset_kind_path="asset_type",
            asset_uri_path="href",
            asset_page_number_path="page",
            asset_confidence_path="score",
            asset_metadata_path="details",
        )
    )

    document = provider.parse(
        _input_file(input_path),
        OcrOptions(language="en", page_range="1-2", formula=True, table=False),
    )

    request = FakeClient.requests[0]
    assert request["url"] == "https://ocr.example/parse"
    assert request["headers"] == {"Authorization": "Bearer secret"}
    assert request["data"] == {
        "checksum": "checksum",
        "file_id": "file_1",
        "formula": "true",
        "language": "en",
        "mime_type": "application/pdf",
        "page_range": "1-2",
        "source_uri": str(input_path),
        "table": "false",
    }
    assert "document" in request["files"]
    assert document.pages[0].page_number == 2
    assert document.pages[0].markdown == "# Page two"
    assert document.pages[0].detected_language == "en"
    assert document.pages[0].blocks[0].kind == "heading"
    assert document.pages[0].blocks[0].bbox == (0.0, 1.0, 2.0, 3.0)
    assert document.pages[0].blocks[0].metadata == {"confidence": 0.94, "font": "bold"}
    assert document.assets[0].id == "img1"
    assert document.assets[0].page_number == 2
    assert document.assets[0].metadata == {"confidence": 0.81, "height": 480, "width": 640}
    assert document.provider_metadata["provider"] == "generic_http"
    assert document.provider_metadata["warnings"] == ["low-confidence"]


def test_generic_http_ocr_accepts_result_level_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_content = None
    FakeClient.response_headers = {}
    FakeClient.response_payload = {"result": {"markdown": "Alice works at Acme."}}
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = GenericHttpOcrProvider(GenericHttpOcrSettings(endpoint="https://ocr.example/parse"))

    document = provider.parse(_input_file(input_path), OcrOptions())

    assert document.pages[0].page_number == 1
    assert document.pages[0].markdown == "Alice works at Acme."
    assert document.pages[0].raw_text == "Alice works at Acme."
    assert document.pages[0].blocks[0].text == "Alice works at Acme."


def test_generic_http_ocr_raises_mapped_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_content = None
    FakeClient.response_headers = {}
    FakeClient.response_payload = {"result": {"status": "error", "error": "bad file"}}
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = GenericHttpOcrProvider(GenericHttpOcrSettings(endpoint="https://ocr.example/parse"))

    with pytest.raises(RuntimeError, match="Generic HTTP OCR failed"):
        provider.parse(_input_file(input_path), OcrOptions())


def test_generic_http_ocr_rejects_nonempty_pages_without_mapped_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.response_content = None
    FakeClient.response_headers = {}
    FakeClient.response_payload = {"result": {"pages": [{"unmapped": "text"}]}}
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = GenericHttpOcrProvider(GenericHttpOcrSettings(endpoint="https://ocr.example/parse"))

    with pytest.raises(RuntimeError, match="pages did not include mapped text"):
        provider.parse(_input_file(input_path), OcrOptions())


def test_generic_http_ocr_preserves_zero_based_page_number(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.response_content = None
    FakeClient.response_headers = {}
    FakeClient.response_payload = {"result": {"pages": [{"page_number": 0, "text": "zero"}]}}
    monkeypatch.setattr(httpx, "Client", FakeClient)

    document = GenericHttpOcrProvider(
        GenericHttpOcrSettings(
            endpoint="https://ocr.example/parse",
            page_number_path="page_number",
            raw_text_path="text",
        )
    ).parse(_input_file(input_path), OcrOptions())

    assert document.pages[0].page_number == 0


def test_generic_http_ocr_rejects_declared_oversized_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_payload = {}
    FakeClient.response_content = b'{"result":{"markdown":"ok"}}'
    FakeClient.response_headers = {"content-length": "1000"}
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = GenericHttpOcrProvider(
        GenericHttpOcrSettings(
            endpoint="https://ocr.example/parse",
            max_response_bytes=10,
        )
    )

    with pytest.raises(RuntimeError, match="above the configured 10 byte limit"):
        provider.parse(_input_file(input_path), OcrOptions())


def test_generic_http_ocr_rejects_actual_oversized_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_payload = {}
    FakeClient.response_content = b'{"result":{"markdown":"too large"}}'
    FakeClient.response_headers = {}
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = GenericHttpOcrProvider(
        GenericHttpOcrSettings(
            endpoint="https://ocr.example/parse",
            max_response_bytes=10,
        )
    )

    with pytest.raises(RuntimeError, match="above the configured 10 byte limit"):
        provider.parse(_input_file(input_path), OcrOptions())


def test_generic_http_ocr_rejects_invalid_json_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_payload = {}
    FakeClient.response_content = b"not-json"
    FakeClient.response_headers = {}
    monkeypatch.setattr(httpx, "Client", FakeClient)
    provider = GenericHttpOcrProvider(GenericHttpOcrSettings(endpoint="https://ocr.example/parse"))

    with pytest.raises(ValueError, match="response was not valid JSON"):
        provider.parse(_input_file(input_path), OcrOptions())


def test_generic_http_ocr_reports_missing_configured_result_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_content = None
    FakeClient.response_headers = {}
    FakeClient.response_payload = {"payload": {}}
    monkeypatch.setattr(httpx, "Client", FakeClient)
    provider = GenericHttpOcrProvider(
        GenericHttpOcrSettings(
            endpoint="https://ocr.example/parse",
            result_path="payload.result",
        )
    )

    with pytest.raises(ValueError, match="missing result path: payload.result"):
        provider.parse(_input_file(input_path), OcrOptions())


def test_generic_http_ocr_normalizes_text_pages_and_primitive_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_content = None
    FakeClient.response_headers = {}
    FakeClient.response_payload = {
        "result": {
            "pages": [
                "First page as plain text.",
                {
                    "blocks": [
                        "Second page heading",
                        {
                            "text": "Second page body",
                            "bbox": [0, 1, "invalid", 3],
                            "confidence": "invalid",
                        },
                    ]
                },
            ]
        }
    }
    monkeypatch.setattr(httpx, "Client", FakeClient)
    provider = GenericHttpOcrProvider(GenericHttpOcrSettings(endpoint="https://ocr.example/parse"))

    document = provider.parse(_input_file(input_path), OcrOptions())

    assert document.pages[0].markdown == "First page as plain text."
    assert [block.text for block in document.pages[0].blocks] == ["First page as plain text."]
    assert document.pages[0].blocks[0].kind == "text"
    assert document.pages[1].page_number == 2
    assert document.pages[1].markdown == "Second page heading\n\nSecond page body"
    assert [block.text for block in document.pages[1].blocks] == [
        "Second page heading",
        "Second page body",
    ]
    assert document.pages[1].blocks[1].bbox is None
    assert "confidence" not in document.pages[1].blocks[1].metadata


def _input_file(path: Path) -> InputFile:
    return InputFile(
        id="file_1",
        path=path,
        source_uri=str(path),
        checksum="checksum",
        mime_type="application/pdf",
        size_bytes=path.stat().st_size,
    )
