# SPDX-License-Identifier: Apache-2.0
"""What a source file is - a document, a dataset, an image - and what that means.

A document is read for its statements. A dataset (a spreadsheet or delimited
table) is profiled once, and its kind decides how much of it is extracted: a
catalogue lists things, measurements record values for things, and anything
else - calculations, templates, schedules - is described rather than
extracted. An image is kept as an asset of the graph even when it has no text
to read.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from kg_processor.config.settings import Settings

ContentKind = Literal["document", "dataset", "image"]
DatasetKind = Literal["catalogue", "measurements", "other"]
DATASET_KINDS: tuple[DatasetKind, ...] = ("catalogue", "measurements", "other")
DATASET_PROFILE_STAGE = "dataset_profile"

_DATASET_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls", ".ods", ".csv", ".tsv"})
_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".heic"}
)
# Which extraction passes each dataset kind runs: (entities, relations).
_DATASET_WINDOWS: dict[str, tuple[bool, bool]] = {
    "catalogue": (True, False),
    "measurements": (True, True),
    "other": (True, False),
}
# A file name token that identifies something - a batch, lot, or order number -
# has a digit and at least this many characters.
_MIN_IDENTIFIER_TOKEN = 4
_IDENTIFIER_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Related files are the same folder's documents that share an identifier with
# the file, or the whole folder when it is this small.
_SMALL_FOLDER = 12
_MAX_RELATED_FILES = 10


# A location outside every known root keeps this many trailing folders.
_FALLBACK_LOCATION_PARTS = 4
LOCATION_PAGE_NUMBER = 0


def relative_location(source_uri: str, roots: Iterable[str]) -> str:
    """A file's path relative to its source root: folders and file name.

    The root is the local input folder, a manifest root, or a bucket or stage
    prefix. A file outside every root keeps its last few folders.
    """

    path = _path(source_uri).replace("\\", "/")
    for root in roots:
        prefix = _path(root).replace("\\", "/").rstrip("/")
        if prefix and path.startswith(prefix + "/"):
            return path[len(prefix) + 1 :]
    return "/".join(PurePosixPath(path).parts[-_FALLBACK_LOCATION_PARTS:])


def source_roots(settings: Settings) -> list[str]:
    """Every root a source file's location may be relative to."""

    files = settings.files
    roots = [str(files.input_path), files.stage_prefix or ""]
    if files.manifest_root is not None:
        roots.append(str(files.manifest_root))
    if files.manifest_path is not None:
        roots.append(str(files.manifest_path.parent))
    # An S3 bucket is its URI's host, an Azure container the first folder
    # of the path; only the prefix after them is a folder of the source.
    if settings.s3.bucket and settings.s3.prefix:
        roots.append(f"s3://{settings.s3.bucket}/{settings.s3.prefix}")
    if settings.azure_blob.container:
        container = settings.azure_blob.container
        prefix = settings.azure_blob.prefix or ""
        roots.append(f"azblob://account/{container}/{prefix}".rstrip("/"))
    roots = [root for root in roots if root]
    # The resolved form too, so a relative input path matches absolute URIs.
    return [*roots, *(str(Path(root).resolve()) for root in roots if "://" not in root)]


def location_text(location: str) -> str:
    """The text of a document's location page."""

    folder, _, name = location.rpartition("/")
    return f"Location: {location}\nFolder: {folder or '.'}\nFile: {name}"


def content_kind(mime_type: str | None, source_uri: str) -> ContentKind:
    """Classify a file by its type: a dataset, an image, or a document."""

    suffix = PurePosixPath(_path(source_uri)).suffix.lower()
    mime = (mime_type or "").lower()
    if suffix in _DATASET_SUFFIXES or mime in {"text/csv", "text/tab-separated-values"}:
        return "dataset"
    if suffix in _IMAGE_SUFFIXES or mime.startswith("image/"):
        return "image"
    return "document"


def dataset_windows(kind: str | None) -> tuple[bool, bool]:
    """Whether a dataset of this kind gets entity and relation extraction.

    An unprofiled file - including a dataset whose profile failed - is
    extracted in full, as any document would be.
    """

    return _DATASET_WINDOWS.get(kind or "", (True, True))


def dataset_profiles(trace: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The dataset profile recorded for each document, by document id."""

    return {
        str(event["document_id"]): event
        for event in trace
        if event.get("stage") == DATASET_PROFILE_STAGE and event.get("document_id")
    }


def window_filter(trace: Iterable[dict[str, Any]]) -> Callable[[str], tuple[bool, bool]]:
    """Map a document id to the extraction passes its profile allows."""

    profiles = dataset_profiles(trace)
    return lambda document_id: dataset_windows(profiles.get(document_id, {}).get("kind"))


def annotate_documents(
    rows: list[dict[str, Any]],
    trace: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill each document row's summary and related files.

    The summary comes from a dataset's profile. Related files are those in the
    same folder that share an identifier token with the file (a batch or order
    number in both names), or every file of a small folder, so an image or a
    data file can be found from the documents beside it.
    """

    profiles = dataset_profiles(trace)
    related = related_file_ids(rows)
    return [
        {
            **row,
            "summary": profiles.get(str(row.get("file_id")), {}).get("summary")
            or row.get("summary"),
            "related_file_ids": related.get(str(row.get("file_id")), []),
        }
        for row in rows
    ]


def related_file_ids(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Relate datasets and images to the documents beside them.

    An image in a subfolder of its own (an experiment's "Pictures") belongs
    with the documents one level up when its own folder holds none.
    """

    by_folder: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_folder[_folder(row)].append(row)
    related: dict[str, list[str]] = {}
    for folder, folder_rows in by_folder.items():
        for row in folder_rows:
            if row.get("kind") not in {"dataset", "image"}:
                continue
            file_id = str(row.get("file_id"))
            others = [other for other in folder_rows if str(other.get("file_id")) != file_id]
            if not any(other.get("kind") == "document" for other in others):
                others += by_folder.get(str(PurePosixPath(folder).parent), [])
            tokens = _identifier_tokens(row)
            sharing = [other for other in others if tokens & _identifier_tokens(other)]
            chosen = sharing or (others if len(others) < _SMALL_FOLDER else [])
            related[file_id] = sorted(str(other.get("file_id")) for other in chosen)[
                :_MAX_RELATED_FILES
            ]
    return related


def _folder(row: dict[str, Any]) -> str:
    folder = row.get("folder")
    if folder is not None:
        return str(folder)
    return str(PurePosixPath(_path(str(row.get("source_uri", "")))).parent)


def _identifier_tokens(row: dict[str, Any]) -> set[str]:
    name = PurePosixPath(_path(str(row.get("source_uri", "")))).stem
    return {
        token.casefold()
        for token in _IDENTIFIER_TOKEN_RE.findall(name)
        if len(token) >= _MIN_IDENTIFIER_TOKEN and any(character.isdigit() for character in token)
    }


def _path(source_uri: str) -> str:
    parsed = urlparse(source_uri)
    return unquote(parsed.path) if parsed.scheme else source_uri
