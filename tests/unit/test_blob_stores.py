"""Contract tests for local and S3-compatible distributed payload stores."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from kg_processor.adapters.distributed.local_blob import LocalBlobStore
from kg_processor.adapters.distributed.s3_blob import S3BlobStore, S3BlobStoreConfig


def test_local_blob_store_round_trips_and_rejects_path_escape(tmp_path: Path) -> None:
    """Local development storage must preserve bytes within its owned namespace."""

    store = LocalBlobStore((tmp_path / "objects").as_uri())
    store.initialize()

    uri = store.put("run/entities/part.json", b'{"id": 1}', "application/json")
    second_uri = store.put("run/entities/part-2.json", b'{"id": 2}', "application/json")

    assert store.get(uri) == b'{"id": 1}'
    with pytest.raises(ValueError, match="escapes"):
        store.put("../outside", b"bad", "application/octet-stream")
    store.delete_many([uri, second_uri])
    assert not Path(uri.removeprefix("file://")).exists()
    assert not Path(second_uri.removeprefix("file://")).exists()


def test_s3_blob_store_uses_deterministic_path_style_objects(monkeypatch: Any) -> None:
    """S3-compatible endpoints receive exact keys, bytes, and media metadata."""

    client = _MemoryS3Client()
    client_kwargs: dict[str, Any] = {}

    def build_client(*_args: object, **kwargs: Any) -> _MemoryS3Client:
        """Capture transport configuration while returning the in-memory client."""

        client_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(
        "kg_processor.adapters.distributed.s3_blob.boto3.client",
        build_client,
    )
    store = S3BlobStore(
        S3BlobStoreConfig(
            root_uri="s3://flakegraph/prefix",
            endpoint_url="http://object-store:8333",
            access_key_id="access",
            secret_access_key="secret",
        )
    )
    store.initialize()

    uri = store.put("run/chunks/part.parquet", b"parquet", "application/x-parquet")
    second_uri = store.put("run/chunks/part-2.parquet", b"other", "application/x-parquet")

    assert uri == "s3://flakegraph/prefix/run/chunks/part.parquet"
    assert store.get(uri) == b"parquet"
    store.delete_many([uri, second_uri])
    assert client.objects == {}
    assert client.delete_batches == [
        [
            "prefix/run/chunks/part.parquet",
            "prefix/run/chunks/part-2.parquet",
        ]
    ]
    assert client_kwargs["config"].max_pool_connections == 64


def test_s3_blob_store_percent_encodes_uri_reserved_key_characters(monkeypatch: Any) -> None:
    client = _MemoryS3Client()
    monkeypatch.setattr(
        "kg_processor.adapters.distributed.s3_blob.boto3.client",
        lambda *_args, **_kwargs: client,
    )
    store = S3BlobStore(S3BlobStoreConfig(root_uri="s3://flakegraph/prefix"))
    store.initialize()

    uri = store.put("run#1/query?2.json", b"payload", "application/json")

    assert uri == "s3://flakegraph/prefix/run%231/query%3F2.json"
    assert store.get(uri) == b"payload"
    store.delete(uri)
    assert client.objects == {}


class _MemoryS3Body(BytesIO):
    """Provide the streaming-body close/read interface used by the adapter."""


class _MemoryS3Client:
    """Record the small boto3 subset exercised by the storage adapter."""

    def __init__(self) -> None:
        self.buckets: set[str] = set()
        self.objects: dict[tuple[str, str], bytes] = {}
        self.delete_batches: list[list[str]] = []

    def head_bucket(self, *, Bucket: str) -> None:  # noqa: N803 - boto3 API spelling.
        """Return successfully after the test bucket has been initialized."""

        if Bucket not in self.buckets:
            raise ClientError(
                {
                    "Error": {"Code": "NoSuchBucket", "Message": "missing"},
                    "ResponseMetadata": {
                        "HTTPStatusCode": 404,
                        "HTTPHeaders": {},
                        "HostId": "test-host",
                        "RequestId": "test-request",
                        "RetryAttempts": 0,
                    },
                },
                "HeadBucket",
            )

    def create_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        """Create one in-memory bucket."""

        self.buckets.add(Bucket)

    def put_object(  # noqa: N803
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
    ) -> None:
        """Store exact bytes while accepting boto3's content-type argument."""

        assert ContentType
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, _MemoryS3Body]:  # noqa: N803
        """Return a closeable streaming body."""

        return {"Body": _MemoryS3Body(self.objects[(Bucket, Key)])}

    def delete_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        """Delete one object idempotently."""

        self.objects.pop((Bucket, Key), None)

    def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:  # noqa: N803
        """Delete one S3-sized object batch and record its exact keys."""

        keys = [str(item["Key"]) for item in Delete["Objects"]]
        assert Delete["Quiet"] is True
        self.delete_batches.append(keys)
        for key in keys:
            self.objects.pop((Bucket, key), None)
        return {"Errors": []}
