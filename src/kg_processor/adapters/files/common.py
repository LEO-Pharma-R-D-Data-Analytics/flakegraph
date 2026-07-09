"""Shared file-source helpers for globs, checksums, MIME types, and safe paths."""

from __future__ import annotations

import fnmatch
import mimetypes
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import sha256_hex, stable_id

SUPPORTED_SUFFIXES = {
    ".pdf",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".docx",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".pptx",
    ".xlsx",
}


def build_local_input_file(
    path: Path,
    *,
    source_uri: str | None = None,
    file_id: str | None = None,
    checksum: str | None = None,
    mime_type: str | None = None,
    size_bytes: int | None = None,
) -> InputFile:
    """Validate a local file and build the canonical input-file record."""

    if not path.is_file():
        raise FileNotFoundError(f"Manifest file entry does not exist: {path}")
    actual_size = path.stat().st_size
    if size_bytes is not None and size_bytes != actual_size:
        raise ValueError(
            f"Manifest size mismatch for {path}: expected {size_bytes}, got {actual_size}"
        )
    actual_checksum = sha256_hex(path.read_bytes())
    if checksum and checksum != actual_checksum:
        raise ValueError(
            f"Manifest checksum mismatch for {path}: expected {checksum}, got {actual_checksum}"
        )
    return InputFile(
        id=file_id or stable_id("file", actual_checksum),
        path=path,
        source_uri=source_uri or path.resolve().as_uri(),
        checksum=actual_checksum,
        mime_type=mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        size_bytes=actual_size,
    )


def is_supported_file(path: Path) -> bool:
    """Return whether the file suffix is supported by at least one OCR adapter."""

    return path.suffix.lower() in SUPPORTED_SUFFIXES


def matches_include_globs(label: str, include_globs: Iterable[str]) -> bool:
    """Match a normalized path label against configured include globs."""

    normalized = label.replace("\\", "/").lstrip("./")
    return any(_matches_glob(normalized, pattern) for pattern in include_globs)


def _matches_glob(normalized: str, pattern: str) -> bool:
    posix = PurePosixPath(normalized)
    if pattern in {"*", "**", "**/*"}:
        return True
    if posix.match(pattern) or fnmatch.fnmatch(normalized, pattern):
        return True
    if pattern.startswith("**/"):
        root_pattern = pattern.removeprefix("**/")
        return posix.match(root_pattern) or fnmatch.fnmatch(normalized, root_pattern)
    return False
