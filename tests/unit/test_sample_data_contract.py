from __future__ import annotations

import json
from pathlib import Path

from kg_processor.adapters.files.manifest import ManifestFileSource
from kg_processor.domain.ids import sha256_hex

_SAMPLE_DIR = Path("data/samples")
_MANIFEST = _SAMPLE_DIR / "manifest.jsonl"
_EXPECTED_FILES = {
    "martial-arts-schools.pptx",
    "martial-arts-timeline.html",
    "martial-arts-interview.docx",
    "martial-arts-overview.pdf",
    "martial-arts-lineages.pdf",
    "smoke.txt",
}


def test_sample_manifest_pins_every_reusable_original_fixture() -> None:
    rows = [
        json.loads(line)
        for line in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert {row["path"] for row in rows} == _EXPECTED_FILES
    for row in rows:
        path = _SAMPLE_DIR / row["path"]
        assert path.is_file()
        assert row["source_uri"] == f"file://data/samples/{path.name}"
        assert row["checksum"] == sha256_hex(path.read_bytes())
        assert row["size_bytes"] == path.stat().st_size
        assert isinstance(row["mime_type"], str)
        assert row["mime_type"]


def test_sample_manifest_file_source_resolves_full_corpus() -> None:
    files = ManifestFileSource(_MANIFEST).list_files()

    assert {file.path.name for file in files} == _EXPECTED_FILES
    assert all(file.checksum for file in files)
    assert all(file.mime_type for file in files)
    assert all(file.size_bytes > 0 for file in files)


def test_sample_readme_documents_every_reusable_fixture() -> None:
    readme = Path("data/README.md").read_text(encoding="utf-8")

    for filename in _EXPECTED_FILES:
        assert f"`samples/{filename}`" in readme
    assert "`samples/manifest.jsonl`" in readme
    assert "Fixture Matrix" in readme
    assert "Manifest Contract" in readme
