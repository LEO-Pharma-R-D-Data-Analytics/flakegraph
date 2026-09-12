"""Read-only source browsers used before an ingestion request is submitted."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from flakegraph_app.models import SourceObject

# Keep this cross-runtime list aligned with the processing package without
# importing it: Snowflake runs the app on Python 3.11 while workers use the
# repository's Python 3.14 package. A contract test guards against drift.
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


def list_local_objects(path: Path, limit: int = 1_000) -> Sequence[SourceObject]:
    """List a bounded, deterministic set without materializing the whole tree."""

    if not path.exists():
        raise ValueError(f"Input path does not exist: {path}")
    objects: list[SourceObject] = []
    for item in _local_files(path):
        if item.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        stat = item.stat()
        objects.append(
            SourceObject(
                uri=item.resolve().as_uri(),
                name=item.name,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            )
        )
        if len(objects) >= limit:
            break
    return objects


def _local_files(path: Path) -> Iterator[Path]:
    """Yield files in stable path order while allowing callers to stop early."""

    if path.is_file():
        yield path
        return
    for directory, directory_names, file_names in os.walk(path):
        directory_names.sort()
        for file_name in sorted(file_names):
            candidate = Path(directory) / file_name
            if candidate.is_file():
                yield candidate


def list_s3_objects(source: Mapping[str, str], limit: int = 1_000) -> Sequence[SourceObject]:
    """List S3-compatible object metadata without downloading object bodies."""

    # Cloud SDKs stay optional for Snowflake deployments and are loaded only
    # when the corresponding local source browser is selected.
    import boto3  # noqa: PLC0415

    bucket = source.get("bucket", "").strip()
    if not bucket:
        raise ValueError("S3 bucket is required")
    client = boto3.client(
        "s3",
        endpoint_url=source.get("endpoint_url") or None,
        region_name=source.get("region") or None,
    )
    prefix = source.get("prefix", "").strip("/")
    paginator = client.get_paginator("list_objects_v2")
    objects: list[SourceObject] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key", ""))
            if not key or key.endswith("/") or Path(key).suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            modified = item.get("LastModified")
            objects.append(
                SourceObject(
                    uri=f"s3://{bucket}/{key}",
                    name=key,
                    size_bytes=int(item.get("Size", 0)),
                    modified_at=_isoformat(modified),
                    checksum=str(item.get("ETag", "")).strip('"') or None,
                )
            )
            if len(objects) >= limit:
                return objects
    return objects


def list_azure_objects(
    source: Mapping[str, str],
    limit: int = 1_000,
) -> Sequence[SourceObject]:
    """List Azure Blob metadata using a connection string or default identity."""

    from azure.identity import DefaultAzureCredential  # noqa: PLC0415
    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415

    container = source.get("container", "").strip()
    if not container:
        raise ValueError("Azure Blob container is required")
    connection_string = source.get("connection_string", "").strip()
    account_url = source.get("account_url", "").strip()
    if connection_string:
        service = BlobServiceClient.from_connection_string(connection_string)
    elif account_url:
        service = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
    else:
        raise ValueError("Azure account URL is required when no connection string is supplied")
    prefix = source.get("prefix", "").strip("/")
    objects: list[SourceObject] = []
    for blob in service.get_container_client(container).list_blobs(name_starts_with=prefix or None):
        name = str(blob.name)
        if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        modified = getattr(blob, "last_modified", None)
        objects.append(
            SourceObject(
                uri=f"{account_url.rstrip('/')}/{container}/{name}"
                if account_url
                else f"azblob://{container}/{name}",
                name=name,
                size_bytes=int(getattr(blob, "size", 0) or 0),
                modified_at=_isoformat(modified),
                checksum=None,
            )
        )
        if len(objects) >= limit:
            break
    return objects


def _isoformat(value: Any) -> str | None:
    """Call a provider timestamp's ISO formatter only when it is callable."""

    formatter = getattr(value, "isoformat", None)
    return str(formatter()) if callable(formatter) else None
