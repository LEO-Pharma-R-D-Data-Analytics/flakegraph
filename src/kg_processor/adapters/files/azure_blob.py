"""Azure Blob file source for on-prem/local workers outside Snowflake stages.

Blobs are streamed into a local cache because OCR providers operate on local
paths, while the original Azure URI and checksum remain part of provenance.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib
import mimetypes
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from kg_processor.adapters.files.common import SUPPORTED_SUFFIXES
from kg_processor.domain.documents import InputFile
from kg_processor.domain.ids import stable_id


class AzureBlobDownloader(Protocol):
    """Minimal Azure downloader surface used for streaming blob bytes."""

    def chunks(self) -> Iterable[bytes]:
        """Yield byte chunks from a blob download."""
        ...


class AzureBlobContainerClient(Protocol):
    """Minimal container client surface used by the file-source adapter."""

    def list_blobs(self, name_starts_with: str | None = None) -> Iterable[object]:
        """List blobs under an optional prefix."""
        ...

    def download_blob(self, blob: str) -> AzureBlobDownloader:
        """Open a streaming downloader for one blob name."""
        ...


class AzureBlobServiceClient(Protocol):
    """Minimal service client surface for selecting a blob container."""

    def get_container_client(self, container: str) -> AzureBlobContainerClient:
        """Return the client for the configured container."""
        ...


AzureBlobClientFactory = Callable[["AzureBlobFileSourceConfig"], AzureBlobServiceClient]


@dataclass(frozen=True)
class AzureBlobFileSourceConfig:
    """Connection and download settings for Azure Blob file discovery."""

    account_url: str | None
    connection_string: str | None
    container: str
    prefix: str | None
    sas_token: str | None
    download_path: Path


class AzureBlobFileSource:
    """Downloads supported Azure blobs into a local cache as input files."""

    def __init__(
        self,
        config: AzureBlobFileSourceConfig,
        include_globs: list[str] | None = None,
        client_factory: AzureBlobClientFactory | None = None,
    ) -> None:
        self.config = config
        self.include_globs = include_globs or ["**/*"]
        self.client_factory = client_factory or _load_azure_blob_client

    def list_files(self) -> list[InputFile]:
        """List, filter, download, checksum, and normalize Azure Blob files."""

        client = self.client_factory(self.config)
        container_client = client.get_container_client(self.config.container)
        prefix = _normalized_prefix(self.config.prefix)
        files: list[InputFile] = []
        for blob in container_client.list_blobs(name_starts_with=prefix or None):
            blob_name = _blob_name(blob)
            relative_path = _relative_blob_path(blob_name, prefix)
            if Path(relative_path).suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if not _matches_include_globs(relative_path, self.include_globs):
                continue
            local_path = self.config.download_path / _safe_relative_path(relative_path)
            checksum, size_bytes = _download_blob(container_client, blob_name, local_path)
            files.append(
                InputFile(
                    id=stable_id("azure_blob_file", self.config.container, blob_name, checksum),
                    path=local_path,
                    source_uri=_source_uri(self.config, blob_name),
                    checksum=checksum,
                    mime_type=_content_type(blob, blob_name),
                    size_bytes=size_bytes,
                )
            )
        return sorted(files, key=lambda item: item.source_uri)


def _load_azure_blob_client(config: AzureBlobFileSourceConfig) -> AzureBlobServiceClient:
    try:
        module = importlib.import_module("azure.storage.blob")
    except ImportError as exc:
        raise RuntimeError(
            "azure_blob file source requires azure-storage-blob to be installed"
        ) from exc
    raw_client = getattr(module, "BlobServiceClient", None)
    if raw_client is None:
        raise RuntimeError("azure.storage.blob.BlobServiceClient is not available")
    if config.connection_string:
        return cast(
            AzureBlobServiceClient,
            raw_client.from_connection_string(config.connection_string),
        )
    if not config.account_url or not config.sas_token:
        raise ValueError(
            "azure_blob file source requires connection_string or account_url+sas_token"
        )
    return cast(
        AzureBlobServiceClient,
        raw_client(account_url=config.account_url, credential=config.sas_token),
    )


def _download_blob(
    container_client: AzureBlobContainerClient,
    blob_name: str,
    local_path: Path,
) -> tuple[str, int]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloader = container_client.download_blob(blob_name)
    digest = hashlib.sha256()
    size_bytes = 0
    with local_path.open("wb") as handle:
        for chunk in downloader.chunks():
            digest.update(chunk)
            size_bytes += len(chunk)
            handle.write(chunk)
    return digest.hexdigest(), size_bytes


def _blob_name(blob: object) -> str:
    value = _object_value(blob, "name")
    if not isinstance(value, str) or not value:
        raise ValueError("Azure Blob list items must include a non-empty name")
    return value


def _content_type(blob: object, blob_name: str) -> str:
    direct = _object_value(blob, "content_type")
    if isinstance(direct, str) and direct:
        return direct
    content_settings = _object_value(blob, "content_settings")
    if content_settings is not None:
        value = _object_value(content_settings, "content_type")
        if isinstance(value, str) and value:
            return value
    return mimetypes.guess_type(blob_name)[0] or "application/octet-stream"


def _object_value(value: object, key: str) -> object:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _normalized_prefix(prefix: str | None) -> str:
    return prefix.strip("/") + "/" if prefix and prefix.strip("/") else ""


def _relative_blob_path(blob_name: str, prefix: str) -> str:
    normalized = blob_name.strip("/")
    if prefix and normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return normalized


def _safe_relative_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe Azure Blob relative path: {relative_path}")
    return path


def _source_uri(config: AzureBlobFileSourceConfig, blob_name: str) -> str:
    if config.account_url:
        return f"{config.account_url.rstrip('/')}/{config.container}/{blob_name}"
    return f"azblob://{config.container}/{blob_name}"


def _matches_include_globs(relative_path: str, include_globs: list[str]) -> bool:
    for pattern in include_globs:
        if pattern in {"*", "**/*"}:
            return True
        if fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(
            Path(relative_path).name,
            pattern,
        ):
            return True
        # `Path.glob("**/*.txt")` includes files directly inside the searched
        # root. Azure blobs are matched after prefix stripping, so reproduce
        # that local-file behavior for blobs such as `prefix/smoke.txt`.
        if pattern.startswith("**/") and fnmatch.fnmatch(
            relative_path,
            pattern.removeprefix("**/"),
        ):
            return True
    return False
