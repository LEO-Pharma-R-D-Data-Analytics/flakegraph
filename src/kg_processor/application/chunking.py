"""Deterministic document chunking.

Chunk ids include graph/file/page/chunk position and a text hash so reruns are
stable while still detecting changed OCR output. The chunker keeps page/block
provenance because every extracted graph observation must cite its source text.
"""

from __future__ import annotations

import re

from kg_processor.domain.documents import LayoutBlock, ParsedDocument
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import normalize_text, sha256_hex, stable_id

_TOKEN_RE = re.compile(r"\S+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def chunk_document(
    document: ParsedDocument,
    graph_id: str,
    chunk_token_size: int,
    chunk_token_overlap: int,
) -> list[Chunk]:
    """Split a parsed document into stable, provenance-rich text chunks."""

    chunks: list[Chunk] = []
    global_index = 0
    ocr_generation_id = _ocr_generation_id(document)
    for page in document.pages:
        text = page.markdown or page.raw_text
        page_asset_ids = [
            asset.id for asset in document.assets if asset.page_number in {None, page.page_number}
        ]
        for part_index, (content, start, end, token_count) in enumerate(
            _split_text(text, chunk_token_size, chunk_token_overlap)
        ):
            normalized = normalize_text(content)
            if not normalized:
                continue
            content_hash = sha256_hex(normalized)
            chunks.append(
                Chunk(
                    id=stable_id(
                        "chunk",
                        graph_id,
                        document.file_id,
                        page.page_number,
                        part_index,
                        content_hash,
                    ),
                    graph_id=graph_id,
                    file_id=document.file_id,
                    document_id=document.file_id,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    content=content,
                    start_offset=start,
                    end_offset=end,
                    token_count=token_count,
                    content_hash=content_hash,
                    section_path=_section_path_for_offset(text, start),
                    block_ids=_block_ids_for_span(page.blocks, content, start, end),
                    asset_ids=page_asset_ids,
                    ocr_generation_id=ocr_generation_id,
                )
            )
            global_index += 1
    return chunks


def compute_ordered_chunk_hash(chunks: list[Chunk]) -> str:
    """Hash chunk order and content for extraction cache invalidation."""

    return sha256_hex("\x1e".join(f"{chunk.chunk_index}\x1f{chunk.content}" for chunk in chunks))


def _split_text(
    text: str,
    chunk_token_size: int,
    chunk_token_overlap: int,
) -> list[tuple[str, int, int, int]]:
    matches = list(_TOKEN_RE.finditer(text))
    if not matches:
        return []
    chunks: list[tuple[str, int, int, int]] = []
    step = max(1, chunk_token_size - chunk_token_overlap)
    for token_start in range(0, len(matches), step):
        token_end = min(len(matches), token_start + chunk_token_size)
        selected = matches[token_start:token_end]
        if not selected:
            continue
        start = selected[0].start()
        end = selected[-1].end()
        chunks.append((text[start:end], start, end, len(selected)))
        if token_end == len(matches):
            break
    return chunks


def _section_path_for_offset(text: str, offset: int) -> list[str]:
    stack: list[str] = []
    for match in _HEADING_RE.finditer(text):
        if match.start() > offset:
            break
        level = len(match.group(1))
        title = " ".join(match.group(2).strip().split())
        if not title:
            continue
        stack = stack[: level - 1]
        stack.append(title)
    return stack


def _block_ids_for_span(
    blocks: list[LayoutBlock],
    chunk_content: str,
    start: int,
    end: int,
) -> list[str]:
    if not blocks:
        return []
    offset_matches = [block.id for block in blocks if _block_overlaps_span(block, start, end)]
    if offset_matches:
        return offset_matches
    normalized_chunk = normalize_text(chunk_content)
    text_matches = [
        block.id
        for block in blocks
        if block.text and normalize_text(block.text) in normalized_chunk
    ]
    return text_matches or [block.id for block in blocks]


def _block_overlaps_span(block: LayoutBlock, start: int, end: int) -> bool:
    block_start = _metadata_int(block.metadata, "start_offset")
    block_end = _metadata_int(block.metadata, "end_offset")
    if block_start is None or block_end is None:
        return False
    return block_start < end and start < block_end


def _metadata_int(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _ocr_generation_id(document: ParsedDocument) -> str | None:
    for key in ("ocr_generation_id", "generation_id", "provider_generation_id"):
        value = document.provider_metadata.get(key)
        if value:
            return str(value)
    return None
