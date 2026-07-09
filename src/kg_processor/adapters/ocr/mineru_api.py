"""External MinerU-compatible OCR service adapter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx

from kg_processor.adapters.ocr.mineru_common import mineru_assets_from_payloads
from kg_processor.domain.documents import (
    InputFile,
    LayoutBlock,
    ParsedAsset,
    ParsedDocument,
    ParsedPage,
)
from kg_processor.domain.ids import stable_id
from kg_processor.ports.ocr import OcrOptions


class MineruApiOcrProvider:
    """Calls an external MinerU-compatible `/file_parse` service."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Upload a document to MinerU and normalize markdown, blocks, and assets."""

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=options.timeout_seconds) as client:
            with file.path.open("rb") as handle:
                response = client.post(
                    f"{self.base_url}/file_parse",
                    headers=headers,
                    data=_form_data(options),
                    files={"files": (file.path.name, handle, file.mime_type)},
                )
            response.raise_for_status()
        payload = response.json()
        result = _select_result(payload)
        _raise_if_failed(result, file.source_uri)
        pages = _pages_from_result(file, result)
        assets = _assets_from_result(file, result)
        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=pages,
            assets=assets,
            provider_metadata={
                "provider": "mineru_api",
                "endpoint": self.base_url,
                "response_keys": sorted(result.keys()),
            },
        )


def _form_data(options: OcrOptions) -> dict[str, str]:
    data = {
        "return_md": "true",
        "return_middle_json": "true",
        "return_content_list": "true",
        "return_images": _bool_text(options.image_analysis is True),
    }
    _set_if_present(data, "lang_list", options.language)
    _set_if_present(data, "backend", options.backend)
    _set_if_present(data, "parse_method", options.method)
    _set_if_present(data, "page_range", options.page_range)
    _set_if_present(data, "formula_enable", _optional_bool_text(options.formula))
    _set_if_present(data, "table_enable", _optional_bool_text(options.table))
    return data


def _select_result(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("MinerU API response must be a JSON object")
    for key in ("result", "data"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    results = payload.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            return first
    return payload


def _raise_if_failed(result: dict[str, Any], source_uri: str) -> None:
    status = str(result.get("status", "")).lower()
    if status in {"failed", "error"}:
        raise RuntimeError(f"MinerU API failed for {source_uri}: {result}")
    if result.get("error"):
        raise RuntimeError(f"MinerU API failed for {source_uri}: {result['error']}")


def _pages_from_result(file: InputFile, result: dict[str, Any]) -> list[ParsedPage]:
    pages = _pages_from_raw_pages(file, result.get("pages"))
    if pages:
        return pages
    middle_json = _maybe_json_object(result.get("middle_json"))
    if middle_json:
        pages = _pages_from_raw_pages(file, middle_json.get("pages"))
        if pages:
            return pages
    markdown = _first_string(result, ["md_content", "markdown", "md", "content", "text"])
    if markdown:
        return [_page(file, 1, markdown, result)]
    content_list = result.get("content_list")
    if isinstance(content_list, list):
        text = "\n\n".join(_content_list_text(item) for item in content_list).strip()
        if text:
            return [_page(file, 1, text, {"content_list_items": len(content_list)})]
    raise RuntimeError("MinerU API response did not include markdown, content_list, or pages")


def _assets_from_result(file: InputFile, result: dict[str, Any]) -> list[ParsedAsset]:
    payloads = [result]
    middle_json = _maybe_json_object(result.get("middle_json"))
    if middle_json:
        payloads.append(middle_json)
    return mineru_assets_from_payloads(
        file,
        payloads,
        id_prefix="mineru_api_asset",
    )


def _pages_from_raw_pages(file: InputFile, raw_pages: object) -> list[ParsedPage]:
    if not isinstance(raw_pages, list):
        return []
    pages: list[ParsedPage] = []
    for index, raw_page in enumerate(raw_pages, start=1):
        if isinstance(raw_page, dict):
            page_number = _first_int(raw_page, ["page_number", "page", "page_id"]) or index
            text = _first_string(raw_page, ["markdown", "md", "content", "text", "raw_text"])
            pages.append(_page(file, page_number, text, raw_page))
        else:
            pages.append(_page(file, index, str(raw_page), {}))
    return pages


def _page(file: InputFile, page_number: int, text: str, metadata: dict[str, Any]) -> ParsedPage:
    block = LayoutBlock(
        id=stable_id("mineru_api_block", file.id, page_number, text[:256]),
        page_number=page_number,
        kind=str(metadata.get("type") or metadata.get("kind") or "text"),
        text=text,
        metadata={
            key: value
            for key, value in metadata.items()
            if key not in {"markdown", "md", "content", "text", "raw_text"}
        },
    )
    return ParsedPage(
        page_number=page_number,
        markdown=text,
        raw_text=text,
        blocks=[block],
        detected_language=_first_string(metadata, ["language", "detected_language"]) or None,
    )


def _content_list_text(item: object) -> str:
    if isinstance(item, dict):
        return _first_string(item, ["text", "content", "markdown", "md"])
    return str(item)


def _maybe_json_object(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    return None


def _first_string(payload: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _first_int(payload: dict[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _set_if_present(data: dict[str, str], key: str, value: str | None) -> None:
    if value is not None:
        data[key] = value


def _optional_bool_text(value: bool | None) -> str | None:
    return _bool_text(value) if value is not None else None


def _bool_text(value: bool) -> str:
    return str(value).lower()
