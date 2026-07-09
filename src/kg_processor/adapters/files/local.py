"""Local filesystem file source for mounted-folder runs and fixtures."""

from __future__ import annotations

from pathlib import Path

from kg_processor.adapters.files.common import build_local_input_file, is_supported_file
from kg_processor.domain.documents import InputFile


class LocalFileSource:
    """Discovers supported files from a local file or directory."""

    def __init__(self, input_path: Path, include_globs: list[str] | None = None) -> None:
        self.input_path = input_path
        self.include_globs = include_globs or ["**/*"]

    def list_files(self) -> list[InputFile]:
        """Return supported local files as canonical input-file records."""

        candidates: dict[Path, None] = {}
        if self.input_path.is_file():
            candidates[self.input_path] = None
        else:
            for pattern in self.include_globs:
                for path in self.input_path.glob(pattern):
                    if path.is_file():
                        candidates[path] = None

        files: list[InputFile] = []
        for path in sorted(candidates):
            if not is_supported_file(path):
                continue
            files.append(build_local_input_file(path))
        return files
