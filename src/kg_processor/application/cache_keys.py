"""Behavioral cache keys for OCR and graph extraction.

Secrets are excluded by construction. Prompt/schema/provider options are included
because changing any of them can change the normalized graph, even when the input
file bytes are identical.
"""

from __future__ import annotations

import json
from typing import Any

from kg_processor.application.chunking import compute_ordered_chunk_hash
from kg_processor.application.extraction_schema import EXTRACTION_SCHEMA_REVISION
from kg_processor.application.prompt_registry import (
    EXTRACTION_PROMPT_NAMES,
    prompt_fingerprints,
)
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.documents import InputFile
from kg_processor.domain.graph import Chunk
from kg_processor.domain.ids import sha256_hex, stable_id
from kg_processor.ports.cache import ExtractionCacheKey, OcrCacheKey
from kg_processor.ports.ocr import OcrOptions


def build_ocr_cache_key(
    file: InputFile,
    provider: str,
    options: OcrOptions,
) -> OcrCacheKey:
    """Build a deterministic OCR cache key that excludes secrets."""

    options_hash = _hash_ocr_options(options)
    return OcrCacheKey(
        id=stable_id("ocr_cache", file.id, file.checksum, provider, options_hash),
        file_id=file.id,
        checksum=file.checksum,
        ocr_provider=provider,
        options_hash=options_hash,
    )


def build_extraction_cache_key(
    graph_id: str,
    chunks: list[Chunk],
    provider: str,
    model: str,
    graph_settings: GraphSettings,
    timeout_seconds: int,
) -> ExtractionCacheKey:
    """Build a deterministic extraction cache key from chunks and prompt contract."""

    chunk_batch_hash = compute_ordered_chunk_hash(chunks)
    # Extraction cache invalidation follows the whole prompt/schema contract,
    # not just the model name. This prevents stale graph output after prompt
    # tuning or schema changes while keeping credential rotation cache-neutral.
    options_hash = _hash_payload(
        {
            "graph_settings": graph_settings.model_dump(mode="json"),
            "schema_revision": EXTRACTION_SCHEMA_REVISION,
            "prompts": prompt_fingerprints(EXTRACTION_PROMPT_NAMES),
            "timeout_seconds": timeout_seconds,
        }
    )
    return ExtractionCacheKey(
        id=stable_id("extraction_cache", graph_id, chunk_batch_hash, provider, model, options_hash),
        graph_id=graph_id,
        chunk_batch_hash=chunk_batch_hash,
        llm_provider=provider,
        model=model,
        options_hash=options_hash,
    )


def _hash_ocr_options(options: OcrOptions) -> str:
    payload = options.model_dump(mode="json")
    payload.pop("api_key", None)
    return _hash_payload(payload)


def _hash_payload(value: dict[str, Any]) -> str:
    return sha256_hex(json.dumps(value, sort_keys=True, separators=(",", ":")))
