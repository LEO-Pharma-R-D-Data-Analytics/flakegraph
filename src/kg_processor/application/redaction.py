"""Credential redaction for reviewable JSON artifacts.

The processor intentionally writes rich config snapshots, progress metadata,
and extraction traces so reviewers can understand a run after the fact. This
module keeps those artifacts useful while making sure provider credentials stay
out of logs, local files, and Snowflake VARIANT payloads.
"""

from __future__ import annotations

from typing import Any

_REDACTED = "***"

_SENSITIVE_KEY_EXACT = {
    "access_token",
    "authorization",
    "connection_string",
    "oauth_token",
    "password",
    "private_key",
    "private_key_path",
    "refresh_token",
    "sas_token",
    "token",
}
_SENSITIVE_KEY_SUFFIXES = ("_api_key", "_password", "_secret", "_token")


def redact_sensitive_data(value: Any) -> Any:
    """Return a JSON-like copy with sensitive mapping keys redacted.

    Redaction is intentionally key-based. We do not inspect arbitrary strings
    because document text, quotes, model outputs, and file names can legitimately
    contain words that look secret-ish. Provider adapters should still avoid
    putting credentials into metadata, and this function is the last safety net.
    """

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if is_sensitive_key(str(key)):
                redacted[key] = _REDACTED if item else None
            else:
                redacted[key] = redact_sensitive_data(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    return value


def is_sensitive_key(key: str) -> bool:
    """Return whether a mapping key should be redacted in review artifacts."""

    lowered = key.lower()
    return (
        lowered == "api_key"
        or lowered in _SENSITIVE_KEY_EXACT
        or lowered.endswith(_SENSITIVE_KEY_SUFFIXES)
        or "secret" in lowered
    )
