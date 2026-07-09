from __future__ import annotations

from kg_processor.adapters.cache.local_json import LocalJsonCache
from kg_processor.domain.documents import LayoutBlock, ParsedDocument, ParsedPage
from kg_processor.domain.graph import ExtractedEntity, ExtractionResult
from kg_processor.ports.cache import ExtractionCacheKey, OcrCacheKey


def test_local_json_cache_roundtrips_ocr_and_extraction(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = LocalJsonCache(tmp_path)
    ocr_key = OcrCacheKey(
        id="ocr_1",
        file_id="file_1",
        checksum="checksum",
        ocr_provider="builtin_text",
        options_hash="options",
    )
    extraction_key = ExtractionCacheKey(
        id="extraction_1",
        graph_id="graph",
        chunk_batch_hash="chunks",
        llm_provider="fake",
        model="fake",
        options_hash="options",
    )
    document = ParsedDocument(
        file_id="file_1",
        checksum="checksum",
        source_uri="file:///sample.txt",
        mime_type="text/plain",
        pages=[
            ParsedPage(
                page_number=1,
                markdown="Alice",
                raw_text="Alice",
                blocks=[LayoutBlock(id="block_1", page_number=1, kind="text", text="Alice")],
            )
        ],
    )
    extraction = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Alice",
                type="PERSON",
                description="Alice is mentioned.",
                source_chunk_id="chunk_1",
            )
        ],
        relations=[],
    )

    assert cache.get_ocr_document(ocr_key) is None
    assert cache.get_extraction_result(extraction_key) is None

    cache.put_ocr_document(ocr_key, document)
    cache.put_extraction_result(extraction_key, extraction)

    assert cache.get_ocr_document(ocr_key) == document
    assert cache.get_extraction_result(extraction_key) == extraction
