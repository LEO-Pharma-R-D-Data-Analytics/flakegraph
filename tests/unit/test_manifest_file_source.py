from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from kg_processor.adapters.files.manifest import ManifestFileSource
from kg_processor.domain.ids import sha256_hex


def test_manifest_file_source_reads_jsonl_and_verifies_checksums(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("Alice in Copenhagen", encoding="utf-8")
    checksum = sha256_hex(document.read_bytes())
    size_bytes = document.stat().st_size
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": "sample.txt",
                "source_uri": "azblob://documents/sample.txt",
                "file_id": "explicit-file-id",
                "checksum": checksum,
                "mime_type": "text/custom",
                "size_bytes": size_bytes,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    files = ManifestFileSource(manifest).list_files()

    assert len(files) == 1
    assert files[0].id == "explicit-file-id"
    assert files[0].path == document.resolve()
    assert files[0].source_uri == "azblob://documents/sample.txt"
    assert files[0].checksum == checksum
    assert files[0].mime_type == "text/custom"
    assert files[0].size_bytes == size_bytes


def test_manifest_file_source_reads_csv_and_filters_include_globs(tmp_path: Path) -> None:
    keep = tmp_path / "docs" / "keep.pdf"
    drop = tmp_path / "docs" / "drop.txt"
    keep.parent.mkdir()
    keep.write_bytes(b"%PDF sample")
    drop.write_text("drop", encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "source_uri"])
        writer.writeheader()
        writer.writerow({"path": "docs/keep.pdf", "source_uri": "file://keep.pdf"})
        writer.writerow({"path": "docs/drop.txt", "source_uri": "file://drop.txt"})

    files = ManifestFileSource(manifest, include_globs=["**/*.pdf"]).list_files()

    assert [file.path for file in files] == [keep.resolve()]
    assert files[0].mime_type == "application/pdf"


def test_manifest_file_source_rejects_checksum_mismatch(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("Alice", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "sample.txt", "checksum": "wrong"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        ManifestFileSource(manifest).list_files()


def test_manifest_file_source_rejects_size_mismatch(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text("Alice", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"files": [{"path": "sample.txt", "size_bytes": document.stat().st_size + 1}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="size mismatch"):
        ManifestFileSource(manifest).list_files()


def test_manifest_file_source_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Manifest file does not exist"):
        ManifestFileSource(tmp_path / "missing.jsonl").list_files()
