from __future__ import annotations

from pathlib import Path

from kg_processor.adapters.files.local import LocalFileSource


def test_local_file_source_hashes_supported_files(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Alice in Copenhagen", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"binary")

    files = LocalFileSource(tmp_path).list_files()

    assert len(files) == 1
    assert files[0].path == sample
    assert files[0].checksum
    assert files[0].mime_type == "text/plain"
