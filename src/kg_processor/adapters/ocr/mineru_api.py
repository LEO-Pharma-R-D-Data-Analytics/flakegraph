"""External MinerU-compatible OCR service adapter."""

from __future__ import annotations

import json
from typing import Any

from kg_processor.adapters.http import HttpClientPool
from kg_processor.adapters.ocr.generic_http import _json_payload
from kg_processor.adapters.ocr.mineru_common import (
    first_int,
    first_string,
    mineru_assets_from_payloads,
)
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

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        max_response_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        """Normalize service configuration and create a lazy HTTP client pool."""

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        if max_response_bytes < 0:
            raise ValueError("MinerU API response limit must be non-negative")
        self.max_response_bytes = max_response_bytes
        self._http = HttpClientPool()

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Upload a document to MinerU and normalize markdown, blocks, and assets."""

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        client = self._http.client(options.timeout_seconds)
        form_data = _form_data(options)
        # Stream the response so the configured bound is enforced while bytes
        # arrive rather than after httpx has already buffered an oversized body.
        with (
            file.path.open("rb") as handle,
            client.stream(
                "POST",
                f"{self.base_url}/file_parse",
                files={"files": (file.path.name, handle, file.mime_type)},
                headers=headers,
                data=form_data,
            ) as response,
        ):
            response.raise_for_status()
            payload = _json_payload(response, self.max_response_bytes)
        result = _select_result(payload)
        _raise_if_failed(result, file.source_uri)
        pages = _require_text(file, _pages_from_result(file, result))
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

    def close(self) -> None:
        """Release retained keep-alive connections for embedded or test callers."""

        self._http.close()


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
    if isinstance(results, dict):
        for value in results.values():
            if isinstance(value, dict):
                return value
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
    markdown = first_string(result, ["md_content", "markdown", "md", "content", "text"])
    if markdown:
        return [_page(file, 1, markdown, result)]
    content_list = result.get("content_list")
    if isinstance(content_list, list):
        grouped: dict[int, list[str]] = {}
        for item in content_list:
            page_index = (
                first_int(item, ["page_idx", "page_index", "index"])
                if isinstance(item, dict)
                else None
            )
            grouped.setdefault((page_index or 0) + 1, []).append(_content_list_text(item))
        pages = [
            _page(
                file,
                page_number,
                "\n\n".join(part for part in parts if part).strip(),
                {"content_list_items": len(parts)},
            )
            for page_number, parts in sorted(grouped.items())
            if any(part for part in parts)
        ]
        if pages:
            return pages
    raise RuntimeError("MinerU API response did not include markdown, content_list, or pages")


def _require_text(file: InputFile, pages: list[ParsedPage]) -> list[ParsedPage]:
    """Reject a parse that produced no text on any page.

    A document that reaches the graph with no content contributes nothing and is
    indistinguishable from one that was never processed, so an empty result is an
    error rather than an empty success.
    """

    if not any(page.raw_text.strip() or page.markdown.strip() for page in pages):
        raise RuntimeError(f"MinerU API returned no text for {file.path.name}")
    return pages


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
            page_number = first_int(raw_page, ["page_number", "page", "page_id"])
            if page_number is None:
                page_idx = first_int(raw_page, ["page_idx", "page_index", "index"])
                page_number = page_idx + 1 if page_idx is not None else index
            text = first_string(raw_page, ["markdown", "md", "content", "text", "raw_text"])
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
        detected_language=first_string(metadata, ["language", "detected_language"]) or None,
    )


def _content_list_text(item: object) -> str:
    if isinstance(item, dict):
        values: list[str] = []
        primary = first_string(item, ["text", "content", "markdown", "md"])
        if primary:
            values.append(primary)
        for key in (
            "list_items",
            "image_caption",
            "image_footnote",
            "table_caption",
            "table_body",
            "table_footnote",
        ):
            value = item.get(key)
            if isinstance(value, list):
                values.extend(str(part) for part in value if str(part).strip())
            elif value is not None and str(value).strip():
                values.append(str(value))
        return "\n".join(values)
    return str(item)


def _maybe_json_object(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _set_if_present(data: dict[str, str], key: str, value: str | None) -> None:
    if value is not None:
        data[key] = value


def _optional_bool_text(value: bool | None) -> str | None:
    return _bool_text(value) if value is not None else None


def _bool_text(value: bool) -> str:
    return str(value).lower()
