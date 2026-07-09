"""Snowflake stage file source.

The adapter lists stage contents but still exposes the same `InputFile` contract
as local/manifest sources so the pipeline does not need Snowflake-specific file
logic.
"""

from __future__ import annotations

import fnmatch
import mimetypes
from pathlib import Path
from typing import Any

from kg_processor.adapters.files.common import SUPPORTED_SUFFIXES
from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    connect_snowflake,
    stage_path,
)
from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import stable_id

_LIST_ROW_MIN_COLUMNS = 2
_LIST_ROW_CHECKSUM_COLUMNS = 3


class SnowflakeStageFileSource:
    """Lists supported files from a Snowflake stage."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        stage: str,
        prefix: str | None = None,
        include_globs: list[str] | None = None,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        self.config = config
        self.stage = stage
        self.prefix = prefix
        self.include_globs = include_globs or ["**/*"]
        self.connector_factory = connector_factory

    def list_files(self) -> list[InputFile]:
        """Run LIST against the configured stage and normalize the returned rows."""

        location = stage_path(self.stage, self.prefix)
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            cursor.execute(f"LIST {location}")
            rows = cursor.fetchall()
        finally:
            cursor.close()
            connection.close()

        files = [_input_file_from_list_row(self.stage, row) for row in rows]
        return [
            file
            for file in sorted(files, key=lambda item: item.source_uri)
            if file.path.suffix.lower() in SUPPORTED_SUFFIXES
            and _matches_include_globs(file.path.as_posix(), self.include_globs)
        ]


def _input_file_from_list_row(stage: str, row: object) -> InputFile:
    values = _row_values(row)
    if len(values) < _LIST_ROW_MIN_COLUMNS:
        raise ValueError("Snowflake LIST rows must include at least name and size")
    raw_name = str(values[0])
    size_bytes = int(values[1] or 0)
    checksum = (
        str(values[2] or stable_id("stage_file_checksum", raw_name))
        if len(values) >= _LIST_ROW_CHECKSUM_COLUMNS
        else ""
    )
    relative_path = _relative_path_from_list_name(stage, raw_name)
    source_uri = f"{stage.rstrip('/')}/{relative_path}"
    return InputFile(
        id=stable_id("stage_file", source_uri, checksum),
        path=Path(relative_path),
        source_uri=source_uri,
        checksum=checksum or stable_id("stage_file_checksum", source_uri),
        mime_type=mimetypes.guess_type(relative_path)[0] or "application/octet-stream",
        size_bytes=size_bytes,
    )


def _row_values(row: object) -> list[Any]:
    if isinstance(row, dict):
        return [
            row.get("name") or row.get("NAME"),
            row.get("size") or row.get("SIZE"),
            row.get("md5") or row.get("MD5") or row.get("etag") or row.get("ETAG"),
        ]
    if isinstance(row, tuple | list):
        return list(row)
    raise ValueError(f"Unsupported Snowflake LIST row type: {type(row).__name__}")


def _relative_path_from_list_name(stage: str, raw_name: str) -> str:
    name = raw_name.lstrip("@")
    stage_name = stage.lstrip("@").rstrip("/")
    if name.startswith(stage_name + "/"):
        return name[len(stage_name) + 1 :]
    if "/" in name:
        return name.split("/", 1)[1]
    return name


def _matches_include_globs(relative_path: str, include_globs: list[str]) -> bool:
    return any(
        fnmatch.fnmatch(relative_path, pattern)
        or fnmatch.fnmatch(Path(relative_path).name, pattern)
        for pattern in include_globs
    )
