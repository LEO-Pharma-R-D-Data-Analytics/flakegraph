from __future__ import annotations

from typing import Any

from kg_processor.application.redaction import is_sensitive_key, redact_sensitive_data


def test_redact_sensitive_data_recurses_without_mutating_original_payload() -> None:
    payload: dict[str, Any] = {
        "api_key": "llm-secret",
        "provider": "openai_compatible",
        "nested": [
            {
                "password": "snowflake-secret",
                "connection_string": "AccountKey=secret",
                "api_key_header": "X-API-Key",
                "api_key_prefix": "",
            }
        ],
    }

    redacted = redact_sensitive_data(payload)

    assert redacted == {
        "api_key": "***",
        "provider": "openai_compatible",
        "nested": [
            {
                "password": "***",
                "connection_string": "***",
                "api_key_header": "X-API-Key",
                "api_key_prefix": "",
            }
        ],
    }
    assert payload["api_key"] == "llm-secret"
    assert payload["nested"][0]["password"] == "snowflake-secret"


def test_is_sensitive_key_covers_provider_credential_names() -> None:
    assert is_sensitive_key("api_key")
    assert is_sensitive_key("mineru_api_key")
    assert is_sensitive_key("oauth_token")
    assert is_sensitive_key("authorization")
    assert is_sensitive_key("private_key_path")
    assert is_sensitive_key("client_secret")
    assert not is_sensitive_key("api_key_header")
    assert not is_sensitive_key("api_key_prefix")
