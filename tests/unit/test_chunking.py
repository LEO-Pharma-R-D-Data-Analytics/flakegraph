from __future__ import annotations

import pytest

from kg_processor.application import chunking
from kg_processor.application.chunking import chunk_document, compute_ordered_chunk_hash
from kg_processor.domain.documents import LayoutBlock, ParsedAsset, ParsedDocument, ParsedPage
from kg_processor.domain.graph import Chunk


def test_chunk_document_is_stable_and_token_bounded() -> None:
    text = " ".join(f"Token{i}" for i in range(25))
    parsed = ParsedDocument(
        file_id="file_1",
        checksum="abc",
        source_uri="file:///sample.txt",
        mime_type="text/plain",
        pages=[ParsedPage(page_number=1, markdown=text, raw_text=text)],
    )

    chunks = chunk_document(parsed, "graph", chunk_token_size=10, chunk_token_overlap=2)
    second = chunk_document(parsed, "graph", chunk_token_size=10, chunk_token_overlap=2)

    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in second]
    assert [chunk.token_count for chunk in chunks] == [10, 10, 9]
    assert compute_ordered_chunk_hash(chunks) == compute_ordered_chunk_hash(second)


def test_chunk_document_preserves_layout_and_ocr_provenance() -> None:
    text = "# Intro\nAlice Smith works at Acme Corp.\n\n## Evidence\nMinerU OCR parses tables."
    block_start = text.index("Alice")
    block_end = text.index("\n\n## Evidence")
    parsed = ParsedDocument(
        file_id="file_1",
        checksum="abc",
        source_uri="file:///sample.md",
        mime_type="text/markdown",
        pages=[
            ParsedPage(
                page_number=1,
                markdown=text,
                raw_text=text,
                blocks=[
                    LayoutBlock(
                        id="block_intro",
                        page_number=1,
                        kind="paragraph",
                        text="Alice Smith works at Acme Corp.",
                        metadata={"start_offset": block_start, "end_offset": block_end},
                    )
                ],
            )
        ],
        assets=[
            ParsedAsset(id="asset_doc", kind="image"),
            ParsedAsset(id="asset_page", kind="image", page_number=1),
        ],
        provider_metadata={"ocr_generation_id": "ocr-run-1"},
    )

    chunks = chunk_document(parsed, "graph", chunk_token_size=8, chunk_token_overlap=0)

    assert chunks[0].graph_id == "graph"
    assert chunks[0].document_id == "file_1"
    assert chunks[0].section_path == ["Intro"]
    assert chunks[0].block_ids == ["block_intro"]
    assert chunks[0].asset_ids == ["asset_doc", "asset_page"]
    assert all("asset_doc" not in chunk.asset_ids for chunk in chunks[1:])
    assert chunks[0].ocr_generation_id == "ocr-run-1"


def test_chunk_document_indexes_each_block_text_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_texts = [f"Block {index} has evidence." for index in range(20)]
    text = "\n".join(block_texts)
    blocks = [
        LayoutBlock(
            id=f"block_{index}",
            page_number=1,
            kind="paragraph",
            text=block_text,
        )
        for index, block_text in enumerate(block_texts)
    ]
    parsed = ParsedDocument(
        file_id="file_1",
        checksum="abc",
        source_uri="file:///sample.txt",
        mime_type="text/plain",
        pages=[ParsedPage(page_number=1, markdown=text, raw_text=text, blocks=blocks)],
    )
    original = chunking._find_block_text_span  # noqa: SLF001
    calls = 0

    def counted_find(page_text: str, block_text: str, start: int) -> tuple[int, int] | None:
        nonlocal calls
        calls += 1
        return original(page_text, block_text, start)

    monkeypatch.setattr(chunking, "_find_block_text_span", counted_find)

    chunks = chunk_document(parsed, "graph", chunk_token_size=4, chunk_token_overlap=1)

    assert len(chunks) > 1
    assert calls == len(blocks)
    assert {block_id for chunk in chunks for block_id in chunk.block_ids} == {
        block.id for block in blocks
    }


def test_ordered_chunk_hash_is_not_ambiguous_when_content_contains_separators() -> None:
    first = [
        _chunk(0, "a"),
        _chunk(1, "b"),
    ]
    second = [
        _chunk(0, "a\x1e1\x1fb"),
    ]

    assert compute_ordered_chunk_hash(first) != compute_ordered_chunk_hash(second)


def _chunk(index: int, content: str) -> Chunk:
    return Chunk(
        id=f"chunk_{index}",
        graph_id="graph",
        file_id="file_1",
        page_number=1,
        chunk_index=index,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=max(1, len(content.split())),
        content_hash=f"hash_{index}",
    )


def test_chunk_hash_separates_identical_text_under_different_identity() -> None:
    """A cached extraction cites chunk ids, so identity must be part of the key.

    Re-ingesting the same bytes through another source changes chunk ids while
    leaving the text alone. Serving the previous result then yields entities
    grounded in ids the run does not have; every one is filtered out and the run
    reports an empty graph instead of a cache fault.
    """

    def _chunk(chunk_id: str) -> Chunk:
        return Chunk(
            id=chunk_id,
            graph_id="g",
            document_id="d",
            file_id="f",
            page_number=1,
            chunk_index=0,
            content="Judo was founded by Jigoro Kano.",
            start_offset=0,
            end_offset=32,
            token_count=8,
            content_hash="abc123",
        )

    assert compute_ordered_chunk_hash([_chunk("chunk_a")]) != compute_ordered_chunk_hash(
        [_chunk("chunk_b")]
    )
