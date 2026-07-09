from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from kg_processor.adapters.files.azure_blob import (
    AzureBlobFileSource,
    AzureBlobFileSourceConfig,
)


def test_azure_blob_file_source_lists_filters_downloads_and_hashes(tmp_path: Path) -> None:
    container = _FakeContainer(
        {
            "incoming/documents/report.pdf": b"%PDF-data",
            "incoming/documents/readme.txt": b"hello",
            "incoming/images/photo.png": b"png",
            "incoming/ignore.tmp": b"ignored",
        }
    )
    source = AzureBlobFileSource(
        _config(tmp_path),
        include_globs=["documents/*"],
        client_factory=lambda _config: _FakeService(container),
    )

    files = source.list_files()

    assert [file.path.name for file in files] == ["readme.txt", "report.pdf"]
    assert files[0].path.read_bytes() == b"hello"
    assert files[0].path == tmp_path / "documents" / "readme.txt"
    assert files[0].source_uri == (
        "https://storage.example/container/incoming/documents/readme.txt"
    )
    assert files[0].mime_type == "text/plain"
    assert files[0].size_bytes == 5
    assert len(files[0].checksum) == 64


def test_azure_blob_double_star_glob_matches_blob_at_prefix_root(tmp_path: Path) -> None:
    container = _FakeContainer(
        {
            "incoming/smoke.txt": b"hello",
            "incoming/nested/report.txt": b"nested",
            "incoming/ignore.tmp": b"ignored",
        }
    )
    source = AzureBlobFileSource(
        _config(tmp_path),
        include_globs=["**/*.txt"],
        client_factory=lambda _config: _FakeService(container),
    )

    files = source.list_files()

    assert [file.path.relative_to(tmp_path).as_posix() for file in files] == [
        "nested/report.txt",
        "smoke.txt",
    ]


def test_azure_blob_file_source_rejects_unsafe_relative_paths(tmp_path: Path) -> None:
    container = _FakeContainer({"incoming/../secret.pdf": b"secret"})
    source = AzureBlobFileSource(
        _config(tmp_path),
        client_factory=lambda _config: _FakeService(container),
    )

    with pytest.raises(ValueError, match="Unsafe Azure Blob relative path"):
        source.list_files()


def test_azure_blob_file_source_uses_connection_string_source_uri(tmp_path: Path) -> None:
    container = _FakeContainer({"file.pdf": b"pdf"})
    source = AzureBlobFileSource(
        AzureBlobFileSourceConfig(
            account_url=None,
            connection_string="UseDevelopmentStorage=true",
            container="container",
            prefix=None,
            sas_token=None,
            download_path=tmp_path,
        ),
        client_factory=lambda _config: _FakeService(container),
    )

    files = source.list_files()

    assert files[0].source_uri == "azblob://container/file.pdf"


def test_azure_blob_file_source_requires_blob_names(tmp_path: Path) -> None:
    class BadContainer(_FakeContainer):
        def list_blobs(self, name_starts_with: str | None = None) -> list[object]:
            return [{}]

    source = AzureBlobFileSource(
        _config(tmp_path),
        client_factory=lambda _config: _FakeService(BadContainer({})),
    )

    with pytest.raises(ValueError, match="non-empty name"):
        source.list_files()


def _config(tmp_path: Path) -> AzureBlobFileSourceConfig:
    return AzureBlobFileSourceConfig(
        account_url="https://storage.example",
        connection_string=None,
        container="container",
        prefix="incoming",
        sas_token="sas",
        download_path=tmp_path,
    )


@dataclass
class _FakeBlob:
    name: str
    content_type: str | None = None


class _FakeDownloader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def chunks(self) -> list[bytes]:
        midpoint = len(self.payload) // 2
        return [self.payload[:midpoint], self.payload[midpoint:]]


class _FakeContainer:
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.blobs = blobs

    def list_blobs(self, name_starts_with: str | None = None) -> list[object]:
        return [
            _FakeBlob(name, "text/plain" if name.endswith(".txt") else None)
            for name in self.blobs
            if name_starts_with is None or name.startswith(name_starts_with)
        ]

    def download_blob(self, blob: str) -> _FakeDownloader:
        return _FakeDownloader(self.blobs[blob])


class _FakeService:
    def __init__(self, container: _FakeContainer) -> None:
        self.container = container

    def get_container_client(self, container: str) -> _FakeContainer:
        assert container == "container"
        return self.container
