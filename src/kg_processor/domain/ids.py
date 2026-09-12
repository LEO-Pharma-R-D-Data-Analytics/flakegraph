"""Stable id helpers used to make reruns and Snowflake merges idempotent."""

from __future__ import annotations

import hashlib
import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Collapse whitespace so ids are stable across OCR formatting differences."""

    return _WHITESPACE_RE.sub(" ", value.strip())


def sha256_hex(value: bytes | str) -> str:
    """Return a hex SHA-256 digest for bytes or UTF-8 text."""

    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def stable_id(prefix: str, *parts: object, length: int = 32) -> str:
    """Build a deterministic prefixed id from ordered semantic parts."""

    joined = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{sha256_hex(joined)[:length]}"
