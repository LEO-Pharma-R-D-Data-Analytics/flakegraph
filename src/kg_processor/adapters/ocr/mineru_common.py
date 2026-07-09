"""Shared MinerU normalization helpers.

Internal MinerU CLI output and MinerU-compatible HTTP services expose similar
asset shapes but arrive through different transport paths. Keeping asset
normalization here prevents the two adapters from drifting while still letting
each adapter own its page/block response parsing.
"""

from __future__ import annotations

from typing import Any

from kg_processor.domain.documents import InputFile, ParsedAsset
from kg_processor.domain.ids import stable_id

_ASSET_COLLECTION_KEYS = ("images", "assets", "media", "figures")
_BLOB_METADATA_KEYS = {
    "base64",
    "data",
    "image_base64",
    "image_data",
    "bytes",
    "content_bytes",
}


def mineru_assets_from_payloads(
    file: InputFile,
    payloads: list[dict[str, Any]],
    *,
    id_prefix: str,
) -> list[ParsedAsset]:
    """Extract normalized assets from MinerU middle-json/content payloads."""

    assets: list[ParsedAsset] = []
    for payload in payloads:
        _extend_assets(file, assets, payload, id_prefix=id_prefix)
    return _dedupe_assets(assets)


def _extend_assets(
    file: InputFile,
    assets: list[ParsedAsset],
    payload: dict[str, Any],
    *,
    id_prefix: str,
) -> None:
    for key in _ASSET_COLLECTION_KEYS:
        _append_assets_from_collection(
            file,
            assets,
            payload.get(key),
            parent_page_number=None,
            id_prefix=id_prefix,
        )
    raw_pages = payload.get("pages")
    if isinstance(raw_pages, list):
        for page_index, raw_page in enumerate(raw_pages, start=1):
            if not isinstance(raw_page, dict):
                continue
            page_number = _first_int(raw_page, ["page_number", "page", "page_id"]) or page_index
            for key in _ASSET_COLLECTION_KEYS:
                _append_assets_from_collection(
                    file,
                    assets,
                    raw_page.get(key),
                    parent_page_number=page_number,
                    id_prefix=id_prefix,
                )
    content_list = payload.get("content_list")
    if isinstance(content_list, list):
        _append_assets_from_collection(
            file,
            assets,
            content_list,
            parent_page_number=None,
            id_prefix=id_prefix,
        )


def _append_assets_from_collection(
    file: InputFile,
    assets: list[ParsedAsset],
    value: object,
    *,
    parent_page_number: int | None,
    id_prefix: str,
) -> None:
    if not isinstance(value, list):
        return
    for index, item in enumerate(value):
        if isinstance(item, dict) and _looks_like_asset(item):
            assets.append(_asset_from_raw(file, item, index, parent_page_number, id_prefix))


def _looks_like_asset(raw: dict[str, Any]) -> bool:
    kind = _first_string(raw, ["type", "kind", "category"]).lower()
    return bool(
        kind in {"image", "figure", "table", "formula", "asset"}
        or _first_string(raw, ["uri", "url", "path", "img_path", "image_path"])
        or any(key in raw for key in _BLOB_METADATA_KEYS)
    )


def _asset_from_raw(
    file: InputFile,
    raw: dict[str, Any],
    index: int,
    parent_page_number: int | None,
    id_prefix: str,
) -> ParsedAsset:
    raw_id = _first_string(raw, ["id", "asset_id", "image_id", "img_id", "name"])
    page_number = (
        _first_int(raw, ["page_number", "page", "page_id"])
        or _page_from_zero_based_index(raw)
        or parent_page_number
    )
    uri = _first_string(raw, ["uri", "url", "path", "img_path", "image_path"]) or None
    kind = str(raw.get("type") or raw.get("kind") or "image")
    metadata = _redact_blob_metadata(raw)
    return ParsedAsset(
        id=raw_id or stable_id(id_prefix, file.id, page_number, index, kind, uri or ""),
        kind=kind,
        page_number=page_number,
        uri=uri,
        metadata=metadata,
    )


def _page_from_zero_based_index(raw: dict[str, Any]) -> int | None:
    page_index = _first_int(raw, ["page_index", "index"])
    return page_index + 1 if page_index is not None else None


def _redact_blob_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in _BLOB_METADATA_KEYS:
            metadata[f"{key}_present"] = bool(value)
            metadata[f"{key}_length"] = len(value) if isinstance(value, str | bytes) else None
        else:
            metadata[key] = value
    return metadata


def _dedupe_assets(assets: list[ParsedAsset]) -> list[ParsedAsset]:
    deduped: list[ParsedAsset] = []
    seen: set[str] = set()
    for asset in assets:
        if asset.id in seen:
            continue
        seen.add(asset.id)
        deduped.append(asset)
    return deduped


def _first_string(payload: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _first_int(payload: dict[str, Any], keys: list[str]) -> int | None:
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
