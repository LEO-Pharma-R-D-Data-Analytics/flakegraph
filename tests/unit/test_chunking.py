from __future__ import annotations

from kg_processor.application.chunking import chunk_document, compute_ordered_chunk_hash
from kg_processor.domain.documents import LayoutBlock, ParsedAsset, ParsedDocument, ParsedPage


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
    assert chunks[0].ocr_generation_id == "ocr-run-1"
