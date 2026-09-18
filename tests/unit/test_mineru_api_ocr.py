from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from http_fakes import StreamingClient, input_file

from kg_processor.adapters.ocr import mineru_api
from kg_processor.adapters.ocr.mineru_api import MineruApiOcrProvider
from kg_processor.ports.ocr import OcrOptions


def test_mineru_api_ocr_posts_file_parse_and_normalizes_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "results": [
                {
                    "file_name": "sample.pdf",
                    "md_content": "# Title\nAlice works at Acme.",
                    "status": "done",
                }
            ]
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example", api_key="secret")

    document = provider.parse(
        input_file(input_path),
        OcrOptions(
            language="en",
            method="auto",
            backend="pipeline",
            formula=True,
            table=False,
            page_range="1-2",
        ),
    )

    request = client.requests[0]
    assert request["url"] == "https://mineru.example/file_parse"
    assert request["headers"] == {"Authorization": "Bearer secret"}
    assert request["data"] == {
        "backend": "pipeline",
        "formula_enable": "true",
        "end_page_id": "2",
        "lang_list": "en",
        "parse_method": "auto",
        "return_content_list": "true",
        "return_images": "false",
        "return_md": "true",
        "return_middle_json": "true",
        "start_page_id": "1",
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
    client = StreamingClient(
        {
            "data": {
                "middle_json": {
                    "pages": [
                        {"page_number": 3, "text": "Page three", "language": "en"},
                    ]
                }
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert document.pages[0].page_number == 3
    assert document.pages[0].raw_text == "Page three"
    assert document.pages[0].detected_language == "en"


def test_mineru_api_ocr_accepts_results_dict_keyed_by_file_stem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "results": {
                "sample": {
                    "md_content": "# Sample\nFilename-keyed response.",
                    "status": "done",
                }
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert document.pages[0].markdown == "# Sample\nFilename-keyed response."


def test_mineru_api_ocr_ignores_invalid_middle_json_when_markdown_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "data": {
                "md_content": "Fallback markdown",
                "middle_json": "{not-json",
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert document.pages[0].markdown == "Fallback markdown"


def test_mineru_api_ocr_preserves_zero_page_and_accepts_page_idx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "data": {
                "pages": [
                    {"page_number": 0, "text": "Zero page"},
                    {"page": "not-a-number", "page_idx": "1", "text": "Second page"},
                ]
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert [page.page_number for page in document.pages] == [0, 2]


def test_mineru_api_ocr_normalizes_assets_from_result_and_middle_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
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
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

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
    client = StreamingClient(
        {
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
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    document = provider.parse(input_file(input_path), OcrOptions())

    assert [asset.id for asset in document.assets] == ["image-1"]


def test_mineru_api_ocr_raises_failed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"status": "failed", "error": "bad file"})
    monkeypatch.setattr(httpx, "Client", client.open)

    provider = MineruApiOcrProvider("https://mineru.example")

    with pytest.raises(RuntimeError, match="MinerU API failed"):
        provider.parse(input_file(input_path), OcrOptions())


def test_mineru_api_ocr_uploads_and_reads_the_body_as_one_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upload handle must stay open for the streamed request and response."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"data": {"md_content": "Streamed markdown"}})
    monkeypatch.setattr(httpx, "Client", client.open)

    document = MineruApiOcrProvider("https://mineru.example").parse(
        input_file(input_path), OcrOptions()
    )

    request = client.requests[0]
    assert request["method"] == "POST"
    assert request["file_handle_open"] is True
    assert document.pages[0].markdown == "Streamed markdown"


def test_mineru_api_ocr_raises_for_a_rejected_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected upload fails with the parser's own reason, not a status line."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient({"detail": "unsupported media type"}, status_code=415)
    monkeypatch.setattr(httpx, "Client", client.open)

    with pytest.raises(RuntimeError, match="HTTP 415: .*unsupported media type"):
        MineruApiOcrProvider("https://mineru.example").parse(input_file(input_path), OcrOptions())


def test_mineru_api_ocr_rejects_a_declared_response_above_the_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized body is refused before it is read."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {"data": {"md_content": "Small body"}}, response_headers={"content-length": "10000"}
    )
    monkeypatch.setattr(httpx, "Client", client.open)
    monkeypatch.setattr(mineru_api, "_MAX_RESPONSE_BYTES", 64)

    provider = MineruApiOcrProvider("https://mineru.example")

    with pytest.raises(RuntimeError, match="declared"):
        provider.parse(input_file(input_path), OcrOptions())


def test_mineru_api_ocr_tolerates_non_dict_content_list_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A content list is provider data: a stray scalar must not abort the parse."""

    input_path = tmp_path / "sample.pdf"
    input_path.write_bytes(b"%PDF")
    client = StreamingClient(
        {
            "data": {
                "content_list": [
                    "loose paragraph",
                    {"page_idx": 1, "text": "Second page paragraph"},
                ]
            }
        }
    )
    monkeypatch.setattr(httpx, "Client", client.open)

    document = MineruApiOcrProvider("https://mineru.example").parse(
        input_file(input_path), OcrOptions()
    )

    assert [page.page_number for page in document.pages] == [1, 2]
    assert document.pages[0].raw_text == "loose paragraph"
    assert document.pages[1].raw_text == "Second page paragraph"
