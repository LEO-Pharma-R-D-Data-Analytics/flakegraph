"""Convert normalized OCR documents into writer-ready document/page/block rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kg_processor.domain.documents import InputFile, ParsedDocument
from kg_processor.domain.ids import stable_id


@dataclass(frozen=True)
class DocumentArtifacts:
    """Writer-ready document rows and trace data produced from OCR output."""

    document: dict[str, Any]
    pages: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    assets: list[dict[str, Any]]
    trace_event: dict[str, Any]


def build_document_artifacts(
    graph_id: str,
    file: InputFile,
    parsed: ParsedDocument,
    ocr_cache_id: str,
    cache_hit: bool,
) -> DocumentArtifacts:
    """Build every document-level artifact row needed by graph writers."""

    return DocumentArtifacts(
        document=document_row(graph_id, file, parsed),
        pages=page_rows(graph_id, file, parsed),
        blocks=block_rows(graph_id, file, parsed),
        assets=asset_rows(graph_id, file, parsed),
        trace_event=ocr_trace_event(file, parsed, ocr_cache_id, cache_hit),
    )


def ocr_trace_event(
    file: InputFile,
    parsed: ParsedDocument,
    ocr_cache_id: str,
    cache_hit: bool,
) -> dict[str, Any]:
    """Build a compact OCR trace event for reports and debugging."""

    return {
        "stage": "ocr",
        "file_id": file.id,
        "cache_id": ocr_cache_id,
        "cache_hit": cache_hit,
        "provider": parsed.provider_metadata.get("provider"),
        "pages": len(parsed.pages),
        "assets": len(parsed.assets),
        "blocks": sum(len(page.blocks) for page in parsed.pages),
    }


def document_row(graph_id: str, file: InputFile, parsed: ParsedDocument) -> dict[str, Any]:
    """Build the Snowflake-shaped document row for one parsed file."""

    return {
        "id": file.id,
        "graph_id": graph_id,
        "file_id": file.id,
        "checksum": file.checksum,
        "source_uri": file.source_uri,
        "mime_type": file.mime_type,
        "size_bytes": file.size_bytes,
        "ocr_provider": parsed.provider_metadata.get("provider"),
    }


def page_rows(graph_id: str, file: InputFile, parsed: ParsedDocument) -> list[dict[str, Any]]:
    """Build Snowflake-shaped page rows for one parsed document."""

    return [
        {
            "id": stable_id("page", graph_id, file.id, page.page_number),
            "graph_id": graph_id,
            "file_id": file.id,
            "page_number": page.page_number,
            "markdown": page.markdown,
            "raw_text": page.raw_text,
            "detected_language": page.detected_language,
        }
        for page in parsed.pages
    ]


def block_rows(graph_id: str, file: InputFile, parsed: ParsedDocument) -> list[dict[str, Any]]:
    """Build Snowflake-shaped OCR block rows for one parsed document."""

    return [
        {
            "id": block.id,
            "graph_id": graph_id,
            "file_id": file.id,
            "page_number": page.page_number,
            "kind": block.kind,
            "text": block.text,
            "bbox": list(block.bbox) if block.bbox is not None else None,
            "metadata": block.metadata,
        }
        for page in parsed.pages
        for block in page.blocks
    ]


def asset_rows(graph_id: str, file: InputFile, parsed: ParsedDocument) -> list[dict[str, Any]]:
    """Build Snowflake-shaped asset rows for one parsed document."""

    return [
        {
            "id": asset.id,
            "graph_id": graph_id,
            "file_id": file.id,
            "kind": asset.kind,
            "page_number": asset.page_number,
            "uri": asset.uri,
            "metadata": asset.metadata,
        }
        for asset in parsed.assets
    ]
