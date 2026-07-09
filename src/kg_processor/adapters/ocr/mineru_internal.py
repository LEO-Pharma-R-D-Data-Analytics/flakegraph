"""Internal MinerU OCR adapter for the default local production profile."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from kg_processor.adapters.ocr.mineru_common import mineru_assets_from_payloads
from kg_processor.domain.documents import (
    InputFile,
    LayoutBlock,
    ParsedAsset,
    ParsedDocument,
    ParsedPage,
)
from kg_processor.domain.ids import stable_id
from kg_processor.ports.ocr import OcrOptions

_BBOX_COORDS = 4


class MineruInternalOcrProvider:
    """Runs the local MinerU CLI and normalizes its generated artifacts."""

    def __init__(self, command: str = "mineru") -> None:
        self.command = command

    def parse(self, file: InputFile, options: OcrOptions) -> ParsedDocument:
        """Execute MinerU locally and read the generated markdown/json outputs."""

        executable = shutil.which(self.command)
        if executable is None:
            raise RuntimeError(
                f"MinerU command '{self.command}' is not installed or not on PATH. "
                "Install MinerU in the image or choose another OCR provider explicitly."
            )

        with tempfile.TemporaryDirectory(prefix="kg-mineru-") as tmp:
            output_dir = Path(tmp)
            command = self.build_command(executable, file, output_dir, options)
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=options.timeout_seconds,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "MinerU failed with exit code "
                    f"{completed.returncode}: {completed.stderr.strip()}"
                )
            return self._load_output(file, output_dir, completed.stdout, command)

    def build_command(
        self,
        executable: str,
        file: InputFile,
        output_dir: Path,
        options: OcrOptions,
    ) -> list[str]:
        """Build the MinerU CLI command from provider-neutral OCR options."""

        command = [
            executable,
            "--path",
            str(file.path),
            "--output",
            str(output_dir),
        ]
        _append_value(command, "--method", options.method)
        _append_value(command, "--backend", options.backend)
        _append_value(command, "--effort", options.effort)
        _append_value(command, "--lang", options.language)
        _append_value(command, "--api-url", options.api_url)
        _append_value(command, "--url", options.server_url)

        start_page_id, end_page_id = _resolve_page_window(options)
        _append_value(command, "--start", start_page_id)
        _append_value(command, "--end", end_page_id)
        _append_bool(command, "--formula", options.formula)
        _append_bool(command, "--table", options.table)
        _append_bool(command, "--image-analysis", options.image_analysis)
        if options.client_side_output_generation:
            command.extend(["--client-side-output-generation", "true"])
        return command

    def _load_output(
        self,
        file: InputFile,
        output_dir: Path,
        stdout: str,
        command: list[str] | None = None,
    ) -> ParsedDocument:
        json_files = sorted(output_dir.rglob("*.json"))
        markdown_files = sorted(output_dir.rglob("*.md"))

        page_json = _first_page_json(json_files)
        if page_json:
            pages, assets = _document_parts_from_mineru_json(file, page_json)
        elif markdown_files:
            text = markdown_files[0].read_text(encoding="utf-8")
            pages = [_page(file, 1, text)]
            assets = []
        elif json_files:
            pages, assets = _document_parts_from_mineru_json(file, json_files[0])
        else:
            raise RuntimeError(f"MinerU did not produce JSON or Markdown output in {output_dir}")

        return ParsedDocument(
            file_id=file.id,
            checksum=file.checksum,
            source_uri=file.source_uri,
            mime_type=file.mime_type,
            pages=pages,
            assets=assets,
            provider_metadata={
                "provider": "mineru_internal",
                "command": self.command,
                "argv": command or [],
                "stdout_tail": stdout[-2000:],
            },
        )


def _pages_from_mineru_json(file: InputFile, path: Path) -> list[ParsedPage]:
    pages, _assets = _document_parts_from_mineru_json(file, path)
    return pages


def _document_parts_from_mineru_json(
    file: InputFile,
    path: Path,
) -> tuple[list[ParsedPage], list[ParsedAsset]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _pages_from_mineru_payload(file, payload), _assets_from_mineru_payload(file, payload)


def _pages_from_mineru_payload(file: InputFile, payload: object) -> list[ParsedPage]:
    if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
        raw_pages = payload["pages"]
    elif isinstance(payload, list):
        raw_pages = payload
    else:
        raw_pages = [{"page_number": 1, "text": json.dumps(payload, ensure_ascii=False)}]

    pages: list[ParsedPage] = []
    for index, raw in enumerate(raw_pages, start=1):
        if not isinstance(raw, dict):
            pages.append(_page(file, index, str(raw)))
            continue
        page_number = int(raw.get("page_number") or raw.get("page") or index)
        text = _first_string(raw, ["markdown", "md", "text", "raw_text", "content"])
        blocks = _blocks_from_raw(file, page_number, raw.get("blocks"))
        pages.append(
            ParsedPage(
                page_number=page_number,
                markdown=text,
                raw_text=text,
                blocks=blocks or [_block(file, page_number, text)],
                detected_language=_first_string(raw, ["language", "detected_language"]) or None,
            )
        )
    return pages


def _assets_from_mineru_payload(file: InputFile, payload: object) -> list[ParsedAsset]:
    if isinstance(payload, dict):
        return mineru_assets_from_payloads(
            file,
            [payload],
            id_prefix="mineru_internal_asset",
        )
    return []


def _first_page_json(json_files: list[Path]) -> Path | None:
    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
            return path
    return None


def _blocks_from_raw(file: InputFile, page_number: int, raw_blocks: Any) -> list[LayoutBlock]:
    if not isinstance(raw_blocks, list):
        return []
    blocks: list[LayoutBlock] = []
    for idx, raw in enumerate(raw_blocks):
        if not isinstance(raw, dict):
            continue
        text = _first_string(raw, ["text", "content", "markdown"]) or ""
        bbox = raw.get("bbox")
        parsed_bbox = (
            tuple(float(v) for v in bbox)
            if isinstance(bbox, list) and len(bbox) == _BBOX_COORDS
            else None
        )
        blocks.append(
            LayoutBlock(
                id=stable_id("block", file.id, page_number, idx, text[:128]),
                page_number=page_number,
                kind=str(raw.get("type") or raw.get("kind") or "text"),
                text=text,
                bbox=parsed_bbox,
                metadata={k: v for k, v in raw.items() if k not in {"text", "content", "markdown"}},
            )
        )
    return blocks


def _first_string(payload: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _append_value(command: list[str], flag: str, value: str | int | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _append_bool(command: list[str], flag: str, value: bool | None) -> None:
    if value is not None:
        command.extend([flag, str(value).lower()])


def _resolve_page_window(options: OcrOptions) -> tuple[int | None, int | None]:
    if options.start_page_id is not None or options.end_page_id is not None:
        return options.start_page_id, options.end_page_id
    if not options.page_range:
        return None, None
    raw = options.page_range.strip()
    if not raw:
        return None, None
    if "-" not in raw:
        page_id = _parse_page_id(raw)
        return page_id, page_id
    start_raw, end_raw = raw.split("-", 1)
    start = _parse_page_id(start_raw) if start_raw.strip() else None
    end = _parse_page_id(end_raw) if end_raw.strip() else None
    if start is not None and end is not None and end < start:
        raise ValueError(f"Invalid MinerU page_range '{raw}': end page is before start page")
    return start, end


def _parse_page_id(value: str) -> int:
    try:
        page_id = int(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid MinerU page id '{value}'. MinerU page ids are zero-based integers."
        ) from exc
    if page_id < 0:
        raise ValueError(f"Invalid MinerU page id '{value}': page ids must be non-negative")
    return page_id


def _page(file: InputFile, page_number: int, text: str) -> ParsedPage:
    return ParsedPage(
        page_number=page_number,
        markdown=text,
        raw_text=text,
        blocks=[_block(file, page_number, text)],
    )


def _block(file: InputFile, page_number: int, text: str) -> LayoutBlock:
    return LayoutBlock(
        id=stable_id("block", file.id, page_number, text[:128]),
        page_number=page_number,
        kind="text",
        text=text,
    )
