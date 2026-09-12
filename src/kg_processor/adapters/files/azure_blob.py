"""Azure Blob file source for on-prem/local workers outside Snowflake stages.

Blobs are streamed into a local cache because OCR providers operate on local
paths, while the original Azure URI and checksum remain part of provenance.
"""

from __future__ import annotations

import hashlib
import importlib
import mimetypes
import re
import uuid
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from kg_processor.adapters.files.common import (
    DOWNLOAD_PARALLELISM,
    cached_input_file,
    claimed_download_root,
    is_supported_file,
    matches_include_globs,
    normalized_prefix,
    object_download_path,
    verify_download_size,
    write_download_metadata,
)
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
        """Return all normalized Azure inputs for compatibility with local callers."""

        return sorted(self.iter_files(), key=lambda item: item.source_uri)

    def iter_files(self) -> Iterator[InputFile]:
        """Stream filtered Azure downloads through a bounded worker pool.

        Azure enumeration remains lazy on the caller thread because cloud SDK
        iterators are not generally safe to consume concurrently. Only independent
        blob downloads enter the bounded pool. ``buffersize`` prevents enumeration
        of a very large container from creating one future per object. Azure's
        listing order and ordered mapping give the distributed planner stable input
        while avoiding a corpus-sized list of rich ``InputFile`` records.
        """

        client = self.client_factory(self.config)
        container_client = client.get_container_client(self.config.container)
        prefix = normalized_prefix(self.config.prefix)
        candidates = self._download_candidates(container_client, prefix)
        download_root = _download_root(self.config)
        with ThreadPoolExecutor(max_workers=DOWNLOAD_PARALLELISM) as executor:
            yield from executor.map(
                lambda candidate: _download_input_file(
                    self.config,
                    container_client,
                    candidate,
                    download_root,
                ),
                candidates,
                buffersize=DOWNLOAD_PARALLELISM,
            )

    def _download_candidates(
        self,
        container_client: AzureBlobContainerClient,
        prefix: str,
    ) -> Iterable[tuple[object, str, str]]:
        """Yield supported blob descriptors without buffering the full listing."""

        for blob in container_client.list_blobs(name_starts_with=prefix or None):
            blob_name = _blob_name(blob)
            relative_path = _relative_blob_path(blob_name, prefix)
            if not is_supported_file(Path(relative_path)):
                continue
            if not matches_include_globs(relative_path, self.include_globs):
                continue
            yield blob, blob_name, relative_path


def _download_input_file(
    config: AzureBlobFileSourceConfig,
    container_client: AzureBlobContainerClient,
    candidate: tuple[object, str, str],
    download_root: Path,
) -> InputFile:
    """Download one independent candidate and build its immutable source record."""

    blob, blob_name, relative_path = candidate
    local_path = object_download_path(
        download_root,
        relative_path,
        stable_id("azure_blob_download_path", config.container, blob_name),
    )
    source_uri = _source_uri(config, blob_name)
    remote_identity = {
        name: str(_object_value(blob, source) or "")
        for name, source in (
            ("etag", "etag"),
            ("size", "size"),
            ("last_modified", "last_modified"),
        )
    }
    cached = cached_input_file(
        local_path,
        remote_identity,
        source_uri,
        stable_id("azure_blob_file", config.container, blob_name),
        _content_type(blob, blob_name),
    )
    if cached is not None:
        return cached
    checksum, size_bytes = _download_blob(container_client, blob_name, local_path)
    verify_download_size(source_uri, remote_identity, size_bytes)
    result = InputFile(
        id=stable_id("azure_blob_file", config.container, blob_name, checksum),
        path=local_path,
        source_uri=source_uri,
        checksum=checksum,
        mime_type=_content_type(blob, blob_name),
        size_bytes=size_bytes,
    )
    write_download_metadata(local_path, remote_identity, result)
    return result


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
    if not config.account_url:
        raise ValueError("azure_blob file source requires connection_string or account_url")
    credential: object = config.sas_token or _default_azure_credential()
    return cast(
        AzureBlobServiceClient,
        raw_client(account_url=config.account_url, credential=credential),
    )


def _default_azure_credential() -> object:
    """Use Azure's standard local, workload, managed-identity credential chain."""

    try:
        identity = importlib.import_module("azure.identity")
    except ImportError as exc:
        raise RuntimeError(
            "Azure identity authentication requires azure-identity to be installed"
        ) from exc
    credential_type = getattr(identity, "DefaultAzureCredential", None)
    if credential_type is None:
        raise RuntimeError("azure.identity.DefaultAzureCredential is not available")
    return credential_type()


def _download_blob(
    container_client: AzureBlobContainerClient,
    blob_name: str,
    local_path: Path,
) -> tuple[str, int]:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloader = container_client.download_blob(blob_name)
    digest = hashlib.sha256()
    size_bytes = 0
    temporary_path = local_path.with_name(f".{local_path.name}.{uuid.uuid4().hex}.part")
    try:
        with temporary_path.open("wb") as handle:
            for chunk in downloader.chunks():
                digest.update(chunk)
                size_bytes += len(chunk)
                handle.write(chunk)
        temporary_path.replace(local_path)
    finally:
        temporary_path.unlink(missing_ok=True)
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


def _relative_blob_path(blob_name: str, prefix: str) -> str:
    normalized = blob_name.strip("/")
    if prefix and normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return normalized


def _download_root(config: AzureBlobFileSourceConfig) -> Path:
    """Claim a run cache for one source and namespace any different source."""

    account = _azure_account_identity(config)
    source_id = stable_id(
        "azure_blob_download",
        account,
        config.container,
        normalized_prefix(config.prefix),
    )
    return claimed_download_root(config.download_path, source_id)


def _source_uri(config: AzureBlobFileSourceConfig, blob_name: str) -> str:
    if config.account_url:
        account_url = _strip_url_query(config.account_url).rstrip("/")
        return f"{account_url}/{config.container}/{blob_name}"
    return f"azblob://{_azure_account_identity(config)}/{config.container}/{blob_name}"


def _azure_account_identity(config: AzureBlobFileSourceConfig) -> str:
    """Return a non-secret account namespace for cache and provenance identity."""

    if config.account_url:
        return _strip_url_query(config.account_url).rstrip("/")
    connection_string = config.connection_string or ""
    match = re.search(r"(?:^|;)AccountName=([^;]+)", connection_string, re.IGNORECASE)
    if match:
        return match.group(1)
    return stable_id("azure_blob_account", connection_string)


def _strip_url_query(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
