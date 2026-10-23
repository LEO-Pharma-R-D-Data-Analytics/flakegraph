from __future__ import annotations

import hashlib
from pathlib import Path

from kg_processor.adapters.files.local import LocalFileSource


def test_local_file_source_hashes_supported_files(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice in Copenhagen", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"binary")

    files = LocalFileSource(tmp_path).list_files()

    assert len(files) == 1
    assert files[0].path == sample
    assert files[0].checksum == hashlib.sha256(b"Alice in Copenhagen").hexdigest()
    assert files[0].mime_type == "text/plain"


def test_local_file_source_ignores_macos_appledouble_sidecars(tmp_path: Path) -> None:
    """Metadata sidecars must not become duplicate document tasks after transfer."""

    (tmp_path / "history.pdf").write_bytes(b"real-pdf")
    (tmp_path / "._history.pdf").write_bytes(b"apple-double-metadata")

    files = LocalFileSource(tmp_path, ["*.pdf"]).list_files()

    assert [item.path.name for item in files] == ["history.pdf"]


def test_local_file_source_browses_without_hashing(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.md").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    (tmp_path / "scratch.tmp").write_bytes(b"ignored")
    source = LocalFileSource(tmp_path, ["**/*"])

    listing = list(source.browse(limit=10))

    assert [item.name for item in listing] == ["a.txt", "b.md"]
    assert listing[0].uri == (tmp_path / "a.txt").resolve().as_uri()
    assert listing[0].size_bytes == 5
    assert listing[0].modified_at is not None
    assert listing[0].checksum is None
    assert [item.name for item in source.browse(limit=1)] == ["a.txt"]
    # Browsing and listing agree on the corpus, so a count shown before a run
    # is the count the run will ingest.
    assert [file.path.name for file in source.list_files()] == ["a.txt", "b.md"]
