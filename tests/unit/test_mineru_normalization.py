from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.ocr.mineru_internal import MineruInternalOcrProvider
from kg_processor.ports.ocr import OcrOptions


def test_mineru_json_output_normalizes_pages_blocks_and_assets(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice Smith", encoding="utf-8")
    file = LocalFileSource(sample).list_files()[0]
    output = tmp_path / "mineru-out"
    output.mkdir()
    (output / "result.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "image_id": "doc-image",
                        "kind": "image",
                        "page_number": 1,
                        "path": "images/doc.png",
                    }
                ],
                "pages": [
                    {
                        "page_number": 3,
                        "markdown": "Alice Smith works at Acme Corp.",
                        "language": "en",
                        "blocks": [
                            {
                                "type": "paragraph",
                                "text": "Alice Smith works at Acme Corp.",
                                "bbox": [1, 2, 3, 4],
                            }
                        ],
                        "figures": [
                            {
                                "type": "figure",
                                "url": "images/page-3.png",
                                "image_base64": "encoded-image",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    parsed = MineruInternalOcrProvider()._load_output(file, output, stdout="ok")

    assert parsed.pages[0].page_number == 3
    assert parsed.pages[0].blocks[0].kind == "paragraph"
    assert parsed.pages[0].blocks[0].bbox == (1, 2, 3, 4)
    assert [asset.id for asset in parsed.assets] == ["doc-image", parsed.assets[1].id]
    assert parsed.assets[0].uri == "images/doc.png"
    assert parsed.assets[1].kind == "figure"
    assert parsed.assets[1].page_number == 3
    assert parsed.assets[1].metadata["image_base64_present"] is True
    assert parsed.assets[1].metadata["image_base64_length"] == len("encoded-image")
    assert "image_base64" not in parsed.assets[1].metadata
    assert parsed.provider_metadata["provider"] == "mineru_internal"


def test_mineru_command_includes_configured_cli_options(tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]

    command = MineruInternalOcrProvider().build_command(
        "/usr/local/bin/mineru",
        file,
        tmp_path / "out",
        OcrOptions(
            language="en",
            page_range="1-3",
            method="ocr",
            backend="pipeline",
            effort="high",
            api_url="http://mineru-api:8000",
            server_url="http://mineru-vlm:30000",
            formula=False,
            table=True,
            image_analysis=False,
            client_side_output_generation=True,
        ),
    )

    assert command == [
        "/usr/local/bin/mineru",
        "--path",
        str(file.path),
        "--output",
        str(tmp_path / "out"),
        "--method",
        "ocr",
        "--backend",
        "pipeline",
        "--effort",
        "high",
        "--lang",
        "en",
        "--api-url",
        "http://mineru-api:8000",
        "--url",
        "http://mineru-vlm:30000",
        "--start",
        "1",
        "--end",
        "3",
        "--formula",
        "false",
        "--table",
        "true",
        "--image-analysis",
        "false",
        "--client-side-output-generation",
        "true",
    ]


def test_mineru_command_explicit_page_ids_override_page_range(tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]

    command = MineruInternalOcrProvider().build_command(
        "mineru",
        file,
        tmp_path / "out",
        OcrOptions(page_range="1-3", start_page_id=4, end_page_id=5),
    )

    assert command[-4:] == ["--start", "4", "--end", "5"]


def test_mineru_output_prefers_markdown_when_json_is_not_page_shaped(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice Smith", encoding="utf-8")
    file = LocalFileSource(sample).list_files()[0]
    output = tmp_path / "mineru-out"
    output.mkdir()
    (output / "metadata.json").write_text('{"status": "ok"}', encoding="utf-8")
    (output / "result.md").write_text("# Alice Smith\n\nWorks at Acme.", encoding="utf-8")

    parsed = MineruInternalOcrProvider()._load_output(file, output, stdout="ok")

    assert parsed.pages[0].markdown == "# Alice Smith\n\nWorks at Acme."


def test_mineru_internal_parse_raises_when_command_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]
    monkeypatch.setattr("kg_processor.adapters.ocr.mineru_internal.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="MinerU command 'mineru' is not installed"):
        MineruInternalOcrProvider().parse(file, OcrOptions())


def test_mineru_internal_parse_raises_with_stderr_when_command_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]
    monkeypatch.setattr(
        "kg_processor.adapters.ocr.mineru_internal.shutil.which",
        lambda command: f"/usr/bin/{command}",
    )

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="bad input")

    monkeypatch.setattr("kg_processor.adapters.ocr.mineru_internal.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="MinerU failed with exit code 2: bad input"):
        MineruInternalOcrProvider().parse(file, OcrOptions())


def test_mineru_internal_rejects_invalid_page_range(tmp_path: Path) -> None:
    sample = tmp_path / "sample.pdf"
    sample.write_bytes(b"%PDF-1.4\n")
    file = LocalFileSource(sample).list_files()[0]

    with pytest.raises(ValueError, match="end page is before start page"):
        MineruInternalOcrProvider().build_command(
            "mineru",
            file,
            tmp_path / "out",
            OcrOptions(page_range="3-1"),
        )
