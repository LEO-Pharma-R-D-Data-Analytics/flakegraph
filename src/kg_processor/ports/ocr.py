"""OCR provider port and options.

The options object is broad on purpose: all OCR providers receive one typed
configuration surface, but graph code only consumes the normalized
`ParsedDocument` result and never branches on provider-specific knobs.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from kg_processor.domain.documents import InputFile, ParsedDocument


class OcrOptions(BaseModel):
    """Provider-neutral OCR options; adapters consume the fields they support."""

    language: str | None = None
    page_range: str | None = None
    timeout_seconds: int = 900
    model_cache_dir: str | None = None
    method: str | None = None
    backend: str | None = None
    effort: str | None = None
    api_url: str | None = None
    api_key: str | None = None
    server_url: str | None = None
    start_page_id: int | None = None
    end_page_id: int | None = None
    formula: bool | None = None
    table: bool | None = None
    image_analysis: bool | None = None
    client_side_output_generation: bool = False
    tesseract_command: str = "tesseract"
    tesseract_pdf_renderer_command: str = "pdftoppm"
    tesseract_dpi: int = 300
    snowflake_parse_mode: str = "OCR"
    snowflake_extract_images: bool = False
    snowflake_page_split: bool = True


class OcrProvider(Protocol):
    """Converts an input file into the canonical parsed-document model."""

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Parse the source file into pages, blocks, assets, and provider metadata."""
        ...
