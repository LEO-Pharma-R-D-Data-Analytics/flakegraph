from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from kg_processor.adapters.ocr.mineru_api import MineruApiOcrProvider
from kg_processor.domain.documents import InputFile
from kg_processor.ports.ocr import OcrOptions


class FakeClient:
    requests: list[dict[str, Any]] = []
    response_payload: dict[str, object] = {}

    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(
        self,
        url: str,
        headers: dict[str, str],
        data: dict[str, str],
        files: dict[str, object],
    ) -> httpx.Response:
        FakeClient.requests.append(
            {
                "url": url,
                "headers": headers,
                "data": data,
                "files": files,
            }
        )
        request = httpx.Request("POST", url, headers=headers)
        return httpx.Response(200, json=FakeClient.response_payload, request=request)


def test_mineru_api_ocr_posts_file_parse_and_normalizes_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_payload = {
        "results": [
            {
                "file_name": "sample.pdf",
                "md_content": "# Title\nAlice works at Acme.",
                "status": "done",
            }
        ]
    }
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example", api_key="secret")

    document = provider.parse(
        _input_file(input_path),
        OcrOptions(
            language="en",
            method="auto",
            backend="pipeline",
            formula=True,
            table=False,
            page_range="1-2",
        ),
    )

    request = FakeClient.requests[0]
    assert request["url"] == "https://mineru.example/file_parse"
    assert request["headers"] == {"Authorization": "Bearer secret"}
    assert request["data"] == {
        "backend": "pipeline",
        "formula_enable": "true",
        "lang_list": "en",
        "page_range": "1-2",
        "parse_method": "auto",
        "return_content_list": "true",
        "return_images": "false",
        "return_md": "true",
        "return_middle_json": "true",
        "table_enable": "false",
    }
    assert document.pages[0].markdown == "# Title\nAlice works at Acme."
    assert document.provider_metadata["provider"] == "mineru_api"


def test_mineru_api_ocr_uses_middle_json_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_payload = {
        "data": {
            "middle_json": {
                "pages": [
                    {"page_number": 3, "text": "Page three", "language": "en"},
                ]
            }
        }
    }
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(_input_file(input_path), OcrOptions())

    assert document.pages[0].page_number == 3
    assert document.pages[0].raw_text == "Page three"
    assert document.pages[0].detected_language == "en"


def test_mineru_api_ocr_normalizes_assets_from_result_and_middle_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_payload = {
        "data": {
            "md_content": "Document with assets",
            "images": [
                {
                    "image_id": "top-image",
                    "type": "image",
                    "page_number": 1,
                    "img_path": "images/top.png",
                    "caption": "Top image",
                }
            ],
            "middle_json": {
                "pages": [
                    {
                        "page_number": 2,
                        "text": "Page two",
                        "figures": [
                            {
                                "type": "figure",
                                "url": "https://mineru.example/figures/2.png",
                                "data": "base64-payload",
                            }
                        ],
                    }
                ]
            },
        }
    }
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(_input_file(input_path), OcrOptions())

    assert [asset.id for asset in document.assets] == [
        "top-image",
        document.assets[1].id,
    ]
    assert document.assets[0].kind == "image"
    assert document.assets[0].page_number == 1
    assert document.assets[0].uri == "images/top.png"
    assert document.assets[0].metadata["caption"] == "Top image"
    assert document.assets[1].kind == "figure"
    assert document.assets[1].page_number == 2
    assert document.assets[1].uri == "https://mineru.example/figures/2.png"
    assert document.assets[1].metadata["data_present"] is True
    assert document.assets[1].metadata["data_length"] == len("base64-payload")
    assert "data" not in document.assets[1].metadata


def test_mineru_api_ocr_deduplicates_assets_from_overlapping_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_payload = {
        "result": {
            "md_content": "Document with duplicated image",
            "images": [
                {"id": "image-1", "kind": "image", "path": "images/1.png"},
            ],
            "pages": [
                {
                    "page_number": 1,
                    "text": "Page one",
                    "images": [
                        {"id": "image-1", "kind": "image", "path": "images/1.png"},
                    ],
                }
            ],
        }
    }
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(_input_file(input_path), OcrOptions())

    assert [asset.id for asset in document.assets] == ["image-1"]


def test_mineru_api_ocr_raises_failed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.requests = []
    FakeClient.response_payload = {"status": "failed", "error": "bad file"}
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example")

    with pytest.raises(RuntimeError, match="MinerU API failed"):
        provider.parse(_input_file(input_path), OcrOptions())


def _input_file(path: Path) -> InputFile:
    return InputFile(
        id="file_1",
        path=path,
        source_uri=str(path),
        checksum="checksum",
        mime_type="application/pdf",
        size_bytes=path.stat().st_size,
    )
