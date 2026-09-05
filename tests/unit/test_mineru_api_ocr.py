from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from kg_processor.adapters.ocr.mineru_api import MineruApiOcrProvider
from kg_processor.domain.documents import InputFile
from kg_processor.ports.ocr import OcrOptions


class FakeClient:
    """Serve MinerU responses through the streaming interface the adapter uses."""

    requests: list[dict[str, Any]] = []
    response_payload: dict[str, object] = {}
    status_code: int = 200
    declared_content_length: int | None = None

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
        """Yield one streamed response and record how it was requested."""

        FakeClient.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "data": data,
                "files": files,
                # The uploaded handle must still be open while the body streams.
                "file_handle_open": not files["files"][1].closed,  # type: ignore[index]
            }
        )
        request = httpx.Request("POST", url, headers=headers)
        body = json.dumps(FakeClient.response_payload).encode("utf-8")
        response_headers = (
            {"content-length": str(FakeClient.declared_content_length)}
            if FakeClient.declared_content_length is not None
            else {}
        )
        response = httpx.Response(
            FakeClient.status_code,
            content=body,
            headers=response_headers,
            request=request,
        )
        try:
            yield response
        finally:
            response.close()


@pytest.fixture(autouse=True)
def _reset_fake_client() -> Iterator[None]:
    """Keep one test's response configuration out of the next one."""

    FakeClient.response_payload = {}
    FakeClient.status_code = 200
    FakeClient.declared_content_length = None
    yield


def test_mineru_api_ocr_posts_file_parse_and_normalizes_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
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


def test_mineru_api_ocr_accepts_results_dict_keyed_by_file_stem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.response_payload = {
        "results": {
            "sample": {
                "md_content": "# Sample\nFilename-keyed response.",
                "status": "done",
            }
        }
    }
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(_input_file(input_path), OcrOptions())

    assert document.pages[0].markdown == "# Sample\nFilename-keyed response."


def test_mineru_api_ocr_ignores_invalid_middle_json_when_markdown_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.response_payload = {
        "data": {
            "md_content": "Fallback markdown",
            "middle_json": "{not-json",
        }
    }
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(_input_file(input_path), OcrOptions())

    assert document.pages[0].markdown == "Fallback markdown"


def test_mineru_api_ocr_preserves_zero_page_and_accepts_page_idx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.response_payload = {
        "data": {
            "pages": [
                {"page_number": 0, "text": "Zero page"},
                {"page": "not-a-number", "page_idx": "1", "text": "Second page"},
            ]
        }
    }
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(_input_file(input_path), OcrOptions())

    assert [page.page_number for page in document.pages] == [0, 2]


def test_mineru_api_ocr_normalizes_assets_from_result_and_middle_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
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
    FakeClient.response_payload = {"status": "failed", "error": "bad file"}
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example")

    with pytest.raises(RuntimeError, match="MinerU API failed"):
        provider.parse(_input_file(input_path), OcrOptions())


def test_mineru_api_ocr_uploads_and_reads_the_body_as_one_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upload handle must stay open for the streamed request and response."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.response_payload = {"data": {"md_content": "Streamed markdown"}}
    monkeypatch.setattr(httpx, "Client", FakeClient)

    document = MineruApiOcrProvider("https://mineru.example").parse(
        _input_file(input_path), OcrOptions()
    )

    request = FakeClient.requests[0]
    assert request["method"] == "POST"
    assert request["file_handle_open"] is True
    assert document.pages[0].markdown == "Streamed markdown"


def test_mineru_api_ocr_raises_for_a_rejected_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected upload must fail rather than be parsed as a document."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.response_payload = {"detail": "unsupported media type"}
    FakeClient.status_code = 415
    monkeypatch.setattr(httpx, "Client", FakeClient)

    with pytest.raises(httpx.HTTPStatusError):
        MineruApiOcrProvider("https://mineru.example").parse(
            _input_file(input_path), OcrOptions()
        )


def test_mineru_api_ocr_rejects_a_declared_response_above_the_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized body is refused before it is read."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.response_payload = {"data": {"md_content": "Small body"}}
    FakeClient.declared_content_length = 10_000
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example", max_response_bytes=64)

    with pytest.raises(RuntimeError, match="declared"):
        provider.parse(_input_file(input_path), OcrOptions())


def test_mineru_api_ocr_bounds_a_body_that_understates_its_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound holds while bytes arrive, not only against the declared length."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.response_payload = {"data": {"md_content": "x" * 500}}
    FakeClient.declared_content_length = 1
    monkeypatch.setattr(httpx, "Client", FakeClient)

    provider = MineruApiOcrProvider("https://mineru.example", max_response_bytes=64)

    with pytest.raises(RuntimeError, match="above the configured"):
        provider.parse(_input_file(input_path), OcrOptions())


def test_mineru_api_ocr_tolerates_non_dict_content_list_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A content list is provider data: a stray scalar must not abort the parse."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    FakeClient.response_payload = {
        "data": {
            "content_list": [
                "loose paragraph",
                {"page_idx": 1, "text": "Second page paragraph"},
            ]
        }
    }
    monkeypatch.setattr(httpx, "Client", FakeClient)

    document = MineruApiOcrProvider("https://mineru.example").parse(
        _input_file(input_path), OcrOptions()
    )

    assert [page.page_number for page in document.pages] == [1, 2]
    assert document.pages[0].raw_text == "loose paragraph"
    assert document.pages[1].raw_text == "Second page paragraph"


def _input_file(path: Path) -> InputFile:
    return InputFile(
        id="file_1",
        path=path,
        source_uri=str(path),
        checksum="checksum",
        mime_type="application/pdf",
        size_bytes=path.stat().st_size,
    )
