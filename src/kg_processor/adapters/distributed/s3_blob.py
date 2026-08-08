"""S3-compatible immutable payload storage for distributed artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO
from urllib.parse import quote, unquote, urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

_S3_DELETE_BATCH_SIZE = 1_000
_S3_MAX_POOL_CONNECTIONS = 64
_HTTP_CONFLICT_STATUS = 409


@dataclass(frozen=True)
class S3BlobStoreConfig:
    """Describe one S3-compatible artifact namespace and its connection details.

    The root URI supplies the bucket and optional prefix. Endpoint and credentials
    are deployment concerns, allowing the same adapter to target AWS S3, SeaweedFS,
    MinIO-compatible installations, or other SigV4 object stores.
    """

    root_uri: str
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    region: str = "us-east-1"


class S3BlobStore:
    """Persist content-addressed payloads in an S3-compatible object namespace."""

    def __init__(self, config: S3BlobStoreConfig) -> None:
        """Validate the root and construct a path-style-compatible S3 client."""

        parsed = urlparse(config.root_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError("distributed artifact_uri must be an s3:// bucket URI")
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/")
        self.client: S3Client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            region_name=config.region,
            # Source staging and artifact compaction deliberately overlap several
            # independent object operations. Botocore otherwise defaults to ten
            # pooled sockets and serializes part of FlakeGraph's bounded worker
            # concurrency behind the HTTP adapter.
            config=Config(
                max_pool_connections=_S3_MAX_POOL_CONNECTIONS,
                s3={"addressing_style": "path"},
            ),
        )

    def initialize(self) -> None:
        """Create the configured bucket when the endpoint reports it as absent.

        Managed S3 deployments normally provision buckets separately. Local
        S3-compatible fleets benefit from idempotent creation, while authorization
        errors remain visible rather than being mistaken for a missing bucket.
        """

        try:
            self.client.head_bucket(Bucket=self.bucket)
            return
        except ClientError as exc:
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if status not in {400, 404} and error_code not in {
                "404",
                "NoSuchBucket",
                "NotFound",
            }:
                raise
        try:
            self.client.create_bucket(Bucket=self.bucket)
        except ClientError as exc:
            status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if status != _HTTP_CONFLICT_STATUS and error_code not in {
                "BucketAlreadyExists",
                "BucketAlreadyOwnedByYou",
            }:
                raise
            self.client.head_bucket(Bucket=self.bucket)

    def put(self, key: str, payload: bytes, media_type: str) -> str:
        """Write immutable bytes at a deterministic key and return an S3 URI."""

        object_key = self._object_key(key)
        self.client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=payload,
            ContentType=media_type,
        )
        return f"s3://{self.bucket}/{quote(object_key, safe='/')}"

    def get(self, uri: str) -> bytes:
        """Read exact bytes after verifying that the URI belongs to this bucket."""

        bucket, key = self._parse_uri(uri)
        response = self.client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            return bytes(body.read())
        finally:
            body.close()

    def download_to(self, uri: str, destination: BinaryIO) -> None:
        """Stream an owned object through boto3's bounded transfer helper."""

        bucket, key = self._parse_uri(uri)
        self.client.download_fileobj(bucket, key, destination)

    def delete(self, uri: str) -> None:
        """Delete one object; S3 delete semantics make missing objects harmless."""

        bucket, key = self._parse_uri(uri)
        self.client.delete_object(Bucket=bucket, Key=key)

    def delete_many(self, uris: list[str]) -> None:
        """Delete objects through S3's bounded multi-object operation.

        S3 accepts at most 1,000 keys per request. The adapter validates every URI
        before sending anything, then checks per-object errors because a successful
        HTTP response can still report individual deletion failures.
        """

        keys = [self._parse_uri(uri)[1] for uri in uris]
        for offset in range(0, len(keys), _S3_DELETE_BATCH_SIZE):
            batch = keys[offset : offset + _S3_DELETE_BATCH_SIZE]
            response = self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": key} for key in batch], "Quiet": True},
            )
            errors = response.get("Errors", [])
            if errors:
                first_code = str(errors[0].get("Code", "unknown"))
                raise RuntimeError(
                    f"S3 artifact cleanup failed for {len(errors)} object(s); "
                    f"first error code: {first_code}"
                )

    def _object_key(self, key: str) -> str:
        """Join a validated relative artifact key beneath the configured prefix."""

        normalized = key.strip("/")
        if not normalized or normalized.startswith("../") or "/../" in normalized:
            raise ValueError("blob object key must be a non-empty relative path")
        return "/".join(part for part in (self.prefix, normalized) if part)

    def _parse_uri(self, uri: str) -> tuple[str, str]:
        """Return bucket and key while preventing cross-bucket deletion or reads."""

        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValueError("artifact storage URI does not belong to this S3 store")
        key = unquote(parsed.path.lstrip("/"))
        if not key:
            raise ValueError("artifact storage URI is missing an object key")
        return parsed.netloc, key
