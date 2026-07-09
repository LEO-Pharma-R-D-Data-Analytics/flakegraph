from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from kg_processor.adapters.files.local import LocalFileSource
from kg_processor.adapters.ocr.builtin_text import BuiltinTextOcrProvider
from kg_processor.ports.ocr import OcrOptions


def test_builtin_text_ocr_parses_text_fixture() -> None:
    source = LocalFileSource(Path("tests/fixtures/simple.txt"))
    file = source.list_files()[0]

    parsed = BuiltinTextOcrProvider().parse(file, OcrOptions())

    assert parsed.file_id == file.id
    assert parsed.pages[0].page_number == 1
    assert "Alice Smith" in parsed.pages[0].raw_text
    assert parsed.provider_metadata["provider"] == "builtin_text"


def test_builtin_text_ocr_parses_html(tmp_path: Path) -> None:
    html = tmp_path / "sample.html"
    html.write_text(
        """
<html>
  <head><style>.hidden { color: red }</style><script>ignored()</script></head>
  <body><h1>Alice Smith</h1><p>Alice works at Acme Corp.</p></body>
</html>
""",
        encoding="utf-8",
    )
    file = LocalFileSource(html).list_files()[0]

    parsed = BuiltinTextOcrProvider().parse(file, OcrOptions())

    assert "Alice Smith" in parsed.pages[0].raw_text
    assert "Alice works at Acme Corp." in parsed.pages[0].raw_text
    assert "ignored" not in parsed.pages[0].raw_text


def test_builtin_text_ocr_parses_pptx_slides(tmp_path: Path) -> None:
    pptx = tmp_path / "sample.pptx"
    _write_minimal_pptx(pptx)
    file = LocalFileSource(pptx).list_files()[0]

    parsed = BuiltinTextOcrProvider().parse(file, OcrOptions())

    assert [page.page_number for page in parsed.pages] == [1, 2]
    assert "Alice Smith" in parsed.pages[0].raw_text
    assert "Acme Corp" in parsed.pages[1].raw_text


def test_builtin_text_ocr_filters_pages_with_one_based_page_range(tmp_path: Path) -> None:
    pptx = tmp_path / "sample.pptx"
    _write_minimal_pptx(pptx)
    file = LocalFileSource(pptx).list_files()[0]

    parsed = BuiltinTextOcrProvider().parse(file, OcrOptions(page_range="2"))

    assert [page.page_number for page in parsed.pages] == [2]
    assert "Acme Corp" in parsed.pages[0].raw_text
    assert parsed.provider_metadata["page_range"] == "2"


def test_builtin_text_ocr_rejects_invalid_page_range(tmp_path: Path) -> None:
    pptx = tmp_path / "sample.pptx"
    _write_minimal_pptx(pptx)
    file = LocalFileSource(pptx).list_files()[0]

    with pytest.raises(ValueError, match="one-based"):
        BuiltinTextOcrProvider().parse(file, OcrOptions(page_range="0"))


def test_builtin_text_ocr_parses_xlsx_worksheets(tmp_path: Path) -> None:
    workbook = tmp_path / "sample.xlsx"
    _write_minimal_xlsx(workbook)
    file = LocalFileSource(workbook).list_files()[0]

    parsed = BuiltinTextOcrProvider().parse(file, OcrOptions())

    assert [page.page_number for page in parsed.pages] == [1, 2]
    assert "Alice Smith\tAcme Corp" in parsed.pages[0].raw_text
    assert "Copenhagen\t42" in parsed.pages[1].raw_text


def _write_minimal_pptx(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "ppt/slides/slide2.xml",
            """
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:sp>
        <p:txBody><a:p><a:r><a:t>Acme Corp</a:t></a:r></a:p></p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>
""",
        )
        archive.writestr(
            "ppt/slides/slide1.xml",
            """
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:sp>
        <p:txBody><a:p><a:r><a:t>Alice Smith</a:t></a:r></a:p></p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>
""",
        )


def _write_minimal_xlsx(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            """
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <si><t>Alice Smith</t></si>
  <si><t>Acme Corp</t></si>
  <si><t>Copenhagen</t></si>
</sst>
""",
        )
        archive.writestr(
            "xl/worksheets/sheet2.xml",
            """
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>2</v></c><c r="B1"><v>42</v></c></row>
  </sheetData>
</worksheet>
""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            """
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
  </sheetData>
</worksheet>
""",
        )
