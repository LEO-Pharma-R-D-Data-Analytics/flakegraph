"""Snowflake Cortex OCR adapter using `AI_PARSE_DOCUMENT` on staged files."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    as_json_object,
    connect_snowflake,
    scalar_from_first_row,
    split_stage_uri,
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

_SNOWFLAKE_PARSE_SQL = "SELECT AI_PARSE_DOCUMENT(TO_FILE(?, ?), PARSE_JSON(?)::OBJECT, TRUE)"
_BLOB_METADATA_KEYS = {
    "base64",
    "content",
    "data",
    "image_base64",
    "image_data",
}


class SnowflakeCortexOcrProvider:
    """Runs Snowflake Cortex `AI_PARSE_DOCUMENT` for staged input files."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        self.config = config
        self.connector_factory = connector_factory

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Parse a Snowflake stage file and normalize Cortex OCR output."""

        stage, relative_path = split_stage_uri(file.source_uri)
        parse_options = _build_parse_options(options)
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            cursor.execute(
                _SNOWFLAKE_PARSE_SQL,
                [stage, relative_path, json.dumps(parse_options, sort_keys=True)],
            )
            raw_result = scalar_from_first_row(cursor)
        finally:
            cursor.close()
            connection.close()

        result = as_json_object(raw_result)
        value, metadata = _extract_value_or_raise(result, file.source_uri)
        pages = _pages_from_value(file, value)
        assets = _assets_from_value(file, value)
        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=pages,
            assets=assets,
            provider_metadata={
                "provider": "snowflake_cortex",
                "stage": stage,
                "relative_path": relative_path,
                "parse_options": parse_options,
                "snowflake_metadata": metadata,
            },
        )


def _build_parse_options(options: OcrOptions) -> dict[str, object]:
    parse_options: dict[str, object] = {
        "mode": options.snowflake_parse_mode,
        "page_split": options.snowflake_page_split,
    }
    if options.snowflake_extract_images:
        parse_options["extract_images"] = True
    page_filter = _page_filter_from_range(options.page_range)
    if page_filter:
        parse_options["page_filter"] = page_filter
    return parse_options


def _page_filter_from_range(page_range: str | None) -> list[dict[str, int]] | None:
    if not page_range or not page_range.strip():
        return None
    filters: list[dict[str, int]] = []
    for raw_part in page_range.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" not in part:
            page = _parse_page_index(part)
            filters.append({"start": page, "end": page + 1})
            continue
        raw_start, raw_end = part.split("-", 1)
        start = _parse_page_index(raw_start) if raw_start.strip() else 0
        entry = {"start": start}
        if raw_end.strip():
            end = _parse_page_index(raw_end)
            if end < start:
                raise ValueError(
                    f"Invalid Snowflake page_range '{page_range}': end page is before start page"
                )
            entry["end"] = end + 1
        filters.append(entry)
    return filters or None


def _parse_page_index(value: str) -> int:
    try:
        page = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid Snowflake page id '{value}'. Page ids are zero-based integers."
        ) from exc
    if page < 0:
        raise ValueError(f"Invalid Snowflake page id '{value}': page ids must be non-negative")
    return page


def _extract_value_or_raise(
    result: dict[str, Any],
    source_uri: str,
) -> tuple[object, dict[str, Any]]:
    error = result.get("error")
    if error:
        raise RuntimeError(f"Snowflake AI_PARSE_DOCUMENT failed for {source_uri}: {error}")
    value = result.get("value", result)
    metadata = result.get("metadata", {})
    return value, metadata if isinstance(metadata, dict) else {}


def _pages_from_value(file: InputFile, value: object) -> list[ParsedPage]:
    if isinstance(value, dict):
        raw_pages = value.get("pages")
        if isinstance(raw_pages, list):
            return [
                _page_from_raw(file, raw_page, index) for index, raw_page in enumerate(raw_pages)
            ]
        content = _first_string(value, ["content", "markdown", "text", "raw_text"])
        if content:
            return [_page_from_text(file, 1, content, value)]
    if isinstance(value, str):
        return [_page_from_text(file, 1, value, {})]
    return [_page_from_text(file, 1, json.dumps(value, ensure_ascii=False), {})]


def _page_from_raw(file: InputFile, raw_page: object, index: int) -> ParsedPage:
    if not isinstance(raw_page, dict):
        return _page_from_text(file, index + 1, str(raw_page), {})
    page_index = _first_int(raw_page, ["index", "page_index"])
    page_number = _first_int(raw_page, ["page_number", "page"])
    if page_number is None:
        page_number = page_index + 1 if page_index is not None else index + 1
    content = _first_string(raw_page, ["content", "markdown", "text", "raw_text"])
    return _page_from_text(file, page_number, content, raw_page)


def _page_from_text(
    file: InputFile,
    page_number: int,
    text: str,
    metadata: dict[str, Any],
) -> ParsedPage:
    block = LayoutBlock(
        id=stable_id("snowflake_block", file.id, page_number, text[:256]),
        page_number=page_number,
        kind=str(metadata.get("type") or metadata.get("kind") or "page"),
        text=text,
        metadata={
            key: value
            for key, value in metadata.items()
            if key not in {"content", "markdown", "text", "raw_text"}
        },
    )
    return ParsedPage(
        page_number=page_number,
        markdown=text,
        raw_text=text,
        blocks=[block],
        detected_language=_first_string(metadata, ["language", "detected_language"]) or None,
    )


def _assets_from_value(file: InputFile, value: object) -> list[ParsedAsset]:
    if not isinstance(value, dict):
        return []
    raw_images = value.get("images")
    if not isinstance(raw_images, list):
        return []
    return [
        _asset_from_raw_image(file, raw_image, index)
        for index, raw_image in enumerate(raw_images)
        if isinstance(raw_image, dict)
    ]


def _asset_from_raw_image(file: InputFile, raw_image: dict[str, Any], index: int) -> ParsedAsset:
    raw_id = _first_string(raw_image, ["id", "image_id", "name"])
    page_number = _first_int(raw_image, ["page_number", "page"])
    page_index = _first_int(raw_image, ["page_index", "index"])
    if page_number is None and page_index is not None:
        page_number = page_index + 1
    metadata = _redact_blob_metadata(raw_image)
    return ParsedAsset(
        id=raw_id or stable_id("snowflake_asset", file.id, index, metadata),
        kind=str(raw_image.get("type") or raw_image.get("kind") or "image"),
        page_number=page_number,
        uri=_first_string(raw_image, ["uri", "url", "path", "stage_uri"]) or None,
        metadata=metadata,
    )


def _redact_blob_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in _BLOB_METADATA_KEYS:
            metadata[f"{key}_present"] = bool(value)
            metadata[f"{key}_length"] = len(value) if isinstance(value, str | bytes) else None
        else:
            metadata[key] = value
    return metadata


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
