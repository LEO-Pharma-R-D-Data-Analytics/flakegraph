"""Configurable HTTP OCR adapter for future OCR platforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from kg_processor.config.settings import GenericHttpOcrSettings
from kg_processor.domain.documents import (
    InputFile,
    LayoutBlock,
    ParsedAsset,
    ParsedDocument,
    ParsedPage,
)
from kg_processor.domain.ids import stable_id
from kg_processor.ports.ocr import OcrOptions

BBOX_COORDINATE_COUNT = 4


class GenericHttpOcrProvider:
    """Calls a configurable HTTP OCR service and maps its JSON response."""

    def __init__(self, settings: GenericHttpOcrSettings) -> None:
        if not settings.endpoint:
            raise ValueError("generic_http OCR requires generic_http_ocr.endpoint")
        self.settings = settings
        self.endpoint = settings.endpoint

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Upload a file to the OCR endpoint and normalize the returned payload."""

        headers = _headers(self.settings)
        with httpx.Client(timeout=options.timeout_seconds) as client:
            with file.path.open("rb") as handle:
                response = client.post(
                    self.endpoint,
                    headers=headers,
                    data=_request_data(file, options),
                    files={self.settings.file_field: (file.path.name, handle, file.mime_type)},
                )
            response.raise_for_status()

        payload = response.json()
        result = _select_result(payload, self.settings.result_path)
        _raise_if_failed(result, file.source_uri, self.settings)
        pages = _pages_from_result(file, result, self.settings)
        assets = _assets_from_result(file, result, self.settings)
        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=pages,
            assets=assets,
            provider_metadata={
                "provider": "generic_http",
                "endpoint": self.endpoint,
                "result_path": self.settings.result_path,
                "response_keys": sorted(result.keys()) if isinstance(result, dict) else [],
                "warnings": _as_list(_get_path(result, self.settings.warnings_path)),
            },
        )


def _headers(settings: GenericHttpOcrSettings) -> dict[str, str]:
    if not settings.api_key:
        return {}
    return {settings.api_key_header: f"{settings.api_key_prefix}{settings.api_key}"}


def _request_data(file: InputFile, options: OcrOptions) -> dict[str, str]:
    data: dict[str, str] = {
        "file_id": file.id,
        "source_uri": file.source_uri,
        "checksum": file.checksum,
        "mime_type": file.mime_type,
    }
    _set_if_present(data, "language", options.language)
    _set_if_present(data, "page_range", options.page_range)
    _set_if_present(data, "method", options.method)
    _set_if_present(data, "backend", options.backend)
    _set_if_present(data, "effort", options.effort)
    _set_if_present(data, "formula", _optional_bool_text(options.formula))
    _set_if_present(data, "table", _optional_bool_text(options.table))
    _set_if_present(data, "image_analysis", _optional_bool_text(options.image_analysis))
    return data


def _select_result(payload: object, result_path: str | None) -> object:
    if result_path:
        result = _get_path(payload, result_path)
        if result is None:
            raise ValueError(f"Generic HTTP OCR response is missing result path: {result_path}")
        return result
    if isinstance(payload, dict):
        for key in ("result", "data"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    return payload


def _raise_if_failed(result: object, source_uri: str, settings: GenericHttpOcrSettings) -> None:
    status = _as_text(_get_path(result, settings.status_path)).strip().lower()
    if status in {"failed", "failure", "error", "errored"}:
        raise RuntimeError(f"Generic HTTP OCR failed for {source_uri}: status={status}")
    error = _get_path(result, settings.error_path)
    if error:
        raise RuntimeError(f"Generic HTTP OCR failed for {source_uri}: {error}")


def _pages_from_result(
    file: InputFile,
    result: object,
    settings: GenericHttpOcrSettings,
) -> list[ParsedPage]:
    raw_pages = _get_path(result, settings.pages_path)
    if isinstance(raw_pages, list) and raw_pages:
        return [
            _page_from_payload(file, page, index, settings)
            for index, page in enumerate(raw_pages, start=1)
        ]
    page = _page_from_payload(file, result, 1, settings)
    if page.markdown.strip() or page.raw_text.strip():
        return [page]
    raise RuntimeError("Generic HTTP OCR response did not include mapped pages or text")


def _page_from_payload(
    file: InputFile,
    payload: object,
    index: int,
    settings: GenericHttpOcrSettings,
) -> ParsedPage:
    if not isinstance(payload, dict):
        text = _as_text(payload)
        return _page(file, index, text, text, [], None)

    page_number = _as_int(_get_path(payload, settings.page_number_path)) or index
    markdown = _as_text(_get_path(payload, settings.markdown_path))
    raw_text = _as_text(_get_path(payload, settings.raw_text_path))
    blocks = _blocks_from_page(file, payload, page_number, settings)
    if not markdown and blocks:
        markdown = "\n\n".join(block.text for block in blocks if block.text).strip()
    if not raw_text:
        raw_text = markdown
    if not markdown:
        markdown = raw_text
    if not blocks and (markdown or raw_text):
        blocks = [
            LayoutBlock(
                id=stable_id("generic_http_block", file.id, page_number, markdown[:256]),
                page_number=page_number,
                kind="text",
                text=markdown or raw_text,
            )
        ]
    return _page(
        file,
        page_number,
        markdown,
        raw_text,
        blocks,
        _optional_text(_get_path(payload, settings.detected_language_path)),
    )


def _page(
    _file: InputFile,
    page_number: int,
    markdown: str,
    raw_text: str,
    blocks: list[LayoutBlock],
    detected_language: str | None,
) -> ParsedPage:
    return ParsedPage(
        page_number=page_number,
        markdown=markdown,
        raw_text=raw_text,
        blocks=blocks,
        detected_language=detected_language,
    )


def _blocks_from_page(
    file: InputFile,
    payload: dict[str, Any],
    page_number: int,
    settings: GenericHttpOcrSettings,
) -> list[LayoutBlock]:
    raw_blocks = _get_path(payload, settings.blocks_path)
    if not isinstance(raw_blocks, list):
        return []
    blocks: list[LayoutBlock] = []
    for index, block in enumerate(raw_blocks):
        if not isinstance(block, dict):
            text = _as_text(block)
            blocks.append(
                LayoutBlock(
                    id=stable_id("generic_http_block", file.id, page_number, index, text[:256]),
                    page_number=page_number,
                    kind="text",
                    text=text,
                )
            )
            continue
        text = _as_text(_get_path(block, settings.block_text_path))
        block_id = _optional_text(_get_path(block, settings.block_id_path)) or stable_id(
            "generic_http_block",
            file.id,
            page_number,
            index,
            text[:256],
        )
        blocks.append(
            LayoutBlock(
                id=block_id,
                page_number=page_number,
                kind=_optional_text(_get_path(block, settings.block_kind_path)) or "text",
                text=text,
                bbox=_bbox(_get_path(block, settings.block_bbox_path)),
                metadata=_metadata(
                    block,
                    metadata_path=settings.block_metadata_path,
                    confidence_path=settings.block_confidence_path,
                ),
            )
        )
    return blocks


def _assets_from_result(
    file: InputFile,
    result: object,
    settings: GenericHttpOcrSettings,
) -> list[ParsedAsset]:
    raw_assets = _get_path(result, settings.assets_path)
    if not isinstance(raw_assets, list):
        return []
    assets: list[ParsedAsset] = []
    for index, asset in enumerate(raw_assets):
        if not isinstance(asset, dict):
            continue
        kind = _optional_text(_get_path(asset, settings.asset_kind_path)) or "asset"
        uri = _optional_text(_get_path(asset, settings.asset_uri_path))
        asset_id = _optional_text(_get_path(asset, settings.asset_id_path)) or stable_id(
            "generic_http_asset",
            file.id,
            index,
            kind,
            uri or "",
        )
        assets.append(
            ParsedAsset(
                id=asset_id,
                kind=kind,
                page_number=_as_int(_get_path(asset, settings.asset_page_number_path)),
                uri=uri,
                metadata=_metadata(
                    asset,
                    metadata_path=settings.asset_metadata_path,
                    confidence_path=settings.asset_confidence_path,
                ),
            )
        )
    return assets


def _get_path(payload: object, path: str | None) -> object:
    if path is None or path == "":
        return payload
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if 0 <= index < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _optional_text(value: object) -> str | None:
    text = _as_text(value)
    return text if text else None


def _as_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or len(value) != BBOX_COORDINATE_COUNT
    ):
        return None
    coordinates: list[float] = []
    for item in value:
        if isinstance(item, int | float):
            coordinates.append(float(item))
        else:
            return None
    return (coordinates[0], coordinates[1], coordinates[2], coordinates[3])


def _metadata(
    payload: dict[str, Any],
    *,
    metadata_path: str | None,
    confidence_path: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    mapped_metadata = _get_path(payload, metadata_path)
    if isinstance(mapped_metadata, dict):
        metadata.update(mapped_metadata)
    confidence = _confidence(_get_path(payload, confidence_path))
    if confidence is not None:
        metadata["confidence"] = confidence
    return metadata


def _confidence(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _set_if_present(data: dict[str, str], key: str, value: str | None) -> None:
    if value is not None:
        data[key] = value


def _optional_bool_text(value: bool | None) -> str | None:
    return str(value).lower() if value is not None else None
