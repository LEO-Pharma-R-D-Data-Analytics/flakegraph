"""Filesystem JSON cache for local parity with Snowflake cache tables."""

from __future__ import annotations

import json
from pathlib import Path

from kg_processor.domain.documents import ParsedDocument
from kg_processor.domain.graph import ExtractionResult
from kg_processor.ports.cache import ExtractionCacheKey, OcrCacheKey


class LocalJsonCache:
    """Stores cache entries as reviewable JSON files under a local directory."""

    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path

    def get_ocr_document(self, key: OcrCacheKey) -> ParsedDocument | None:
        """Read a cached parsed document for the supplied OCR key."""

        path = self._path("ocr", key.id)
        if not path.exists():
            return None
        return ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))

    def put_ocr_document(self, key: OcrCacheKey, document: ParsedDocument) -> None:
        """Write a parsed document under the supplied OCR key."""

        self._write("ocr", key.id, document.model_dump(mode="json"))

    def get_extraction_result(self, key: ExtractionCacheKey) -> ExtractionResult | None:
        """Read a cached extraction result for the supplied LLM key."""

        path = self._path("extraction", key.id)
        if not path.exists():
            return None
        return ExtractionResult.model_validate_json(path.read_text(encoding="utf-8"))

    def put_extraction_result(self, key: ExtractionCacheKey, result: ExtractionResult) -> None:
        """Write an extraction result under the supplied LLM key."""

        self._write("extraction", key.id, result.model_dump(mode="json"))

    def _write(self, namespace: str, cache_id: str, payload: dict[str, object]) -> None:
        path = self._path(namespace, cache_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _path(self, namespace: str, cache_id: str) -> Path:
        return self.root_path / namespace / f"{cache_id}.json"
