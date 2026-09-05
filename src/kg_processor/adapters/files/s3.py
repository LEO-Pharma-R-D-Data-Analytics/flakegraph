"""S3-compatible object storage file source.

The adapter deliberately targets the small boto3 S3 client surface rather than
AWS-specific orchestration. It therefore works with AWS S3, MinIO, and other
S3-compatible stores while preserving the same local-file contract expected by
OCR providers.
"""

from __future__ import annotations

import hashlib
import mimetypes
import uuid
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import boto3

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


class S3Body(Protocol):
    """Streaming response body returned by ``get_object``."""

    def iter_chunks(self, chunk_size: int = ...) -> Iterable[bytes]:
        """Yield bounded byte chunks from one object."""
        ...


class S3Client(Protocol):
    """Minimal boto3-compatible client surface used by this adapter."""

    def get_paginator(self, operation_name: str) -> Any:
        """Return a paginator for ``list_objects_v2``."""
        ...

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]:
        """Open one object for streaming download."""
        ...


@dataclass(frozen=True)
class S3FileSourceConfig:
    """Connection, namespace, and local staging settings for an S3 source."""

    bucket: str
    prefix: str | None
    endpoint_url: str | None
    region: str | None
    download_path: Path


class S3FileSource:
    """Download supported objects from an S3-compatible bucket."""

    def __init__(
        self,
        config: S3FileSourceConfig,
        include_globs: list[str] | None = None,
        client: S3Client | None = None,
    ) -> None:
        """Retain source settings and an optional injected client for tests."""

        self.config = config
        self.include_globs = include_globs or ["**/*"]
        self.client = client

    def list_files(self) -> list[InputFile]:
        """Return all downloaded objects in deterministic source-URI order."""

        return sorted(self.iter_files(), key=lambda item: item.source_uri)

    def iter_files(self) -> Iterator[InputFile]:
        """List lazily and download objects through a bounded worker pool.

        The bounded map prevents a bucket with hundreds of thousands of keys
        from allocating one future per object. Checksums are calculated from the
        downloaded bytes because multipart ETags are not content hashes.
        """

        client = self.client or _build_s3_client(self.config)
        candidates = self._candidates(client)
        download_root = _download_root(self.config)
        with ThreadPoolExecutor(max_workers=DOWNLOAD_PARALLELISM) as executor:
            yield from executor.map(
                lambda candidate: _download_input_file(
                    self.config,
                    client,
                    candidate,
                    download_root,
                ),
                candidates,
                buffersize=DOWNLOAD_PARALLELISM,
            )

    def _candidates(
        self, client: S3Client
    ) -> Iterable[tuple[str, str, str | None, dict[str, str]]]:
        """Yield supported object keys and metadata without buffering the listing."""

        prefix = normalized_prefix(self.config.prefix)
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = str(item.get("Key", ""))
                if not key or key.endswith("/"):
                    continue
                relative_path = _relative_object_path(key, prefix)
                if not is_supported_file(Path(relative_path)):
                    continue
                if not matches_include_globs(relative_path, self.include_globs):
                    continue
                content_type = item.get("ContentType")
                identity = {
                    name: str(item.get(source) or "")
                    for name, source in (
                        ("etag", "ETag"),
                        ("size", "Size"),
                        ("last_modified", "LastModified"),
                    )
                }
                yield key, relative_path, str(content_type) if content_type else None, identity


def _build_s3_client(config: S3FileSourceConfig) -> S3Client:
    """Build a boto3 client using its standard credential provider chain."""

    return cast(
        S3Client,
        boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region,
        ),
    )


def _download_input_file(
    config: S3FileSourceConfig,
    client: S3Client,
    candidate: tuple[str, str, str | None, dict[str, str]],
    download_root: Path,
) -> InputFile:
    """Download one object atomically enough for a single worker invocation."""

    key, relative_path, listed_content_type, remote_identity = candidate
    local_path = object_download_path(
        download_root,
        relative_path,
        stable_id("s3_download_path", config.bucket, key),
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"s3://{config.bucket}/{key}"
    cached = cached_input_file(
        local_path,
        remote_identity,
        source_uri,
        stable_id("s3_file", config.bucket, key),
        listed_content_type or mimetypes.guess_type(key)[0] or "application/octet-stream",
    )
    if cached is not None:
        return cached
    response = client.get_object(Bucket=config.bucket, Key=key)
    body = response.get("Body")
    if body is None or not hasattr(body, "iter_chunks"):
        raise ValueError(f"S3 object response for {key!r} did not include a streaming body")

    checksum, size_bytes = _download_body(body, local_path)
    verify_download_size(source_uri, remote_identity, size_bytes)

    result = InputFile(
        id=stable_id("s3_file", config.bucket, key, checksum),
        path=local_path,
        source_uri=source_uri,
        checksum=checksum,
        mime_type=(
            str(response.get("ContentType") or listed_content_type)
            if response.get("ContentType") or listed_content_type
            else mimetypes.guess_type(key)[0] or "application/octet-stream"
        ),
        size_bytes=size_bytes,
    )
    write_download_metadata(local_path, remote_identity, result)
    return result


def _download_body(body: S3Body, local_path: Path) -> tuple[str, int]:
    """Stream one S3 body through a unique temporary file and atomic replace."""

    digest = hashlib.sha256()
    size_bytes = 0
    temporary_path = local_path.with_name(f".{local_path.name}.{uuid.uuid4().hex}.part")
    try:
        with temporary_path.open("wb") as handle:
            for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                digest.update(chunk)
                size_bytes += len(chunk)
                handle.write(chunk)
        temporary_path.replace(local_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return digest.hexdigest(), size_bytes


def _relative_object_path(key: str, prefix: str) -> str:
    """Return the path below the selected prefix and reject empty results."""

    relative = key[len(prefix) :] if prefix and key.startswith(prefix) else key
    if not relative:
        raise ValueError("S3 object key resolves to an empty relative path")
    return relative


def _download_root(config: S3FileSourceConfig) -> Path:
    """Claim a run cache for one source and namespace any different source."""

    source_id = stable_id(
        "s3_download",
        config.endpoint_url or "aws",
        config.bucket,
        normalized_prefix(config.prefix),
    )
    return claimed_download_root(config.download_path, source_id)
