"""Focused regression tests for app isolation and reviewed-request security contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from flakegraph_app.configuration import (
    build_run_config,
    effective_request_fingerprint,
    redacted_config,
)
from flakegraph_app.models import (
    IngestionRequest,
    OutputDestination,
    ProviderSelection,
    RuntimeMode,
    SourceKind,
    StorageKind,
)
from flakegraph_app.providers import EMBEDDING_PROVIDERS, LLM_PROVIDERS, option_by_name
from flakegraph_app.spcs import _without_secrets
from flakegraph_app.ui.ingestion import (
    _approved_preflight,
    _default_run_id,
    _preflight_for_request,
    _record_preflight,
    _synchronize_runtime_provider_state,
)
from flakegraph_app.ui.shared import _synchronize_provider_fields
from streamlit_app import _session_cache_token

_ROOT = Path(__file__).resolve().parents[2]


def test_staged_entry_point_import_does_not_load_optional_runtime_modules() -> None:
    """Keep Snowflake's Python 3.11 entry import independent of worker-only packages.

    The fresh interpreter blocks processing-core and local/fleet dependencies to
    model the declared staged environment. Importing the entry point must not
    attempt to load any runtime backend before host detection and user selection.
    """

    script = """
import importlib.abc
import sys

class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("kg_processor", "psycopg", "boto3", "azure")
        if fullname in blocked or fullname.startswith(tuple(name + "." for name in blocked)):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockOptionalImports())
import streamlit_app

for name in (
    "flakegraph_app.backends.local",
    "flakegraph_app.backends.kubernetes",
    "flakegraph_app.backends.snowflake",
    "flakegraph_app.ui.ingestion",
    "flakegraph_app.ui.run_workspace",
):
    assert name not in sys.modules, name
"""
    environment = {**os.environ, "PYTHONPATH": str(_ROOT / "app")}

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_process_wide_cache_keys_are_namespaced_by_streamlit_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent Streamlit's shared data cache from crossing user-session boundaries."""

    values = iter(("a" * 32, "b" * 32))
    monkeypatch.setattr("streamlit_app.secrets.token_hex", lambda _bytes: next(values))
    first_state: dict[str, Any] = {}
    second_state: dict[str, Any] = {}

    first = _session_cache_token(first_state)
    second = _session_cache_token(second_state)

    assert first == "a" * 32
    assert second == "b" * 32
    assert first != second
    assert _session_cache_token(first_state) == first


def test_provider_switch_clears_stale_fields_and_seeds_new_neutral_defaults() -> None:
    """Ensure a new adapter cannot inherit the previous adapter's connection data."""

    state: dict[str, Any] = {
        "_embedding_fields_provider": "openai_compatible",
        "embedding_model": "old-model",
        "embedding_endpoint": "https://old.example/v1",
        "embedding_credential": "OLD_API_KEY",
        "embedding_dimension": 1536,
    }
    selected = option_by_name(EMBEDDING_PROVIDERS, "snowflake_cortex")

    _synchronize_provider_fields(
        state,
        "embedding",
        selected,
        endpoint_default="",
        model_default="snowflake-arctic-embed-m-v1.5",
    )

    assert state["embedding_model"] == "snowflake-arctic-embed-m-v1.5"
    assert state["embedding_dimension"] == 768
    assert "embedding_endpoint" not in state
    assert "embedding_credential" not in state


def test_runtime_switch_resets_all_provider_widget_ownership() -> None:
    """Keep local endpoints and credential references out of Snowflake requests."""

    state: dict[str, Any] = {
        "_ingestion_provider_runtime": "local",
        "llm_provider": "openai_compatible",
        "llm_endpoint": "https://old.example/v1",
        "llm_credential": "OLD_KEY",
        "_llm_fields_provider": "openai_compatible",
    }
    defaults = {
        "ocr": "snowflake_cortex",
        "llm": "snowflake_cortex",
        "embedding": "snowflake_cortex",
    }

    _synchronize_runtime_provider_state(state, RuntimeMode.SNOWFLAKE, defaults)

    # The stale selection is removed rather than overwritten, so the selector
    # falls back to the new runtime's default instead of carrying the previous
    # runtime's adapter and its transport into the request.
    assert "llm_provider" not in state
    assert "llm_endpoint" not in state
    assert "llm_credential" not in state
    assert "_llm_fields_provider" not in state


def test_effective_config_replaces_stale_provider_fields_but_keeps_neutral_tuning(
    tmp_path: Path,
) -> None:
    """Clear base-profile credentials and transports while preserving safe tuning."""

    profile = tmp_path / "profile.yaml"
    profile.write_text(
        """
llm:
  provider: openai_compatible
  endpoint: https://old.example/v1
  api_key: ${OLD_LLM_KEY}
  timeout_seconds: 45
embedding:
  provider: sentence_transformers
  model: old-model
  endpoint: https://old.example/v1
  api_key: ${OLD_EMBED_KEY}
  dimension: 384
  batch_size: 17
  device: cuda
""".lstrip(),
        encoding="utf-8",
    )
    request = replace(
        _request(tmp_path),
        base_config_path=profile,
        llm=ProviderSelection("snowflake_cortex", model="mistral-large2"),
        embedding=ProviderSelection(
            "snowflake_cortex",
            model="snowflake-arctic-embed-m-v1.5",
            dimension=768,
        ),
    )

    config = build_run_config(request)

    assert config["llm"] == {
        "provider": "snowflake_cortex",
        "model": "mistral-large2",
        "timeout_seconds": 45,
    }
    assert config["embedding"] == {
        "provider": "snowflake_cortex",
        "model": "snowflake-arctic-embed-m-v1.5",
        "dimension": 768,
        "batch_size": 17,
    }


def test_redaction_and_spcs_stripping_cover_connection_urls_and_strings() -> None:
    """Treat DSNs and connection URLs as credentials across every preview boundary."""

    config = {
        "distributed": {
            "database_url": "postgresql://user:password@db/private",
            "connection_string": "AccountName=a;AccountKey=secret",
            "dsn": "user/password@database",
            "oauth_token_path": "/snowflake/session/token",
        }
    }

    preview = redacted_config(config)
    stripped = _without_secrets(config)

    assert preview["distributed"]["database_url"] == "${REDACTED}"
    assert preview["distributed"]["connection_string"] == "${REDACTED}"
    assert preview["distributed"]["dsn"] == "${REDACTED}"
    assert preview["distributed"]["oauth_token_path"] == "/snowflake/session/token"
    assert stripped == {"distributed": {"oauth_token_path": "/snowflake/session/token"}}


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("llm", "api_key", "literal-api-key"),
        ("distributed", "database_url", "postgresql://user:secret@db/name"),
        ("azure_blob", "connection_string", "AccountKey=literal-secret"),
    ],
)
def test_app_rejects_literal_secrets_from_base_profiles(
    tmp_path: Path,
    section: str,
    key: str,
    value: str,
) -> None:
    """Require environment references instead of persisting literal credentials."""

    profile = tmp_path / "profile.yaml"
    profile.write_text(f"{section}:\n  {key}: {value}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Literal secret"):
        build_run_config(replace(_request(tmp_path), base_config_path=profile))


def test_preflight_approval_is_bound_to_the_exact_effective_request(tmp_path: Path) -> None:
    """Invalidate a passing approval after any effective request field changes."""

    request = _request(tmp_path)
    fingerprint = effective_request_fingerprint(request)
    changed = replace(
        request,
        embedding=replace(request.embedding, dimension=1024),
    )
    changed_fingerprint = effective_request_fingerprint(changed)
    state: dict[str, Any] = {}

    _record_preflight(state, fingerprint, {"ok": True, "checks": ["ready"]})

    assert fingerprint != changed_fingerprint
    assert _approved_preflight(state, fingerprint) is not None
    assert _approved_preflight(state, changed_fingerprint) is None

    _record_preflight(state, changed_fingerprint, {"ok": False, "errors": ["not ready"]})
    assert _preflight_for_request(state, changed_fingerprint) is not None
    assert _approved_preflight(state, changed_fingerprint) is None


def test_default_run_ids_combine_operator_timestamp_with_64_bits_of_entropy() -> None:
    """Avoid collisions between concurrent sessions creating runs in one second."""

    now = datetime(2026, 7, 16, 12, 30, 45, tzinfo=UTC)

    first = _default_run_id(now, "0123456789abcdef")
    second = _default_run_id(now, "fedcba9876543210")

    assert first == "graph-20260716-123045-0123456789abcdef"
    assert second == "graph-20260716-123045-fedcba9876543210"
    assert first != second


def test_embedding_model_dimension_is_explicit_in_effective_preflight_config(
    tmp_path: Path,
) -> None:
    """Expose vector shape in reviewed config and reject unknown implicit shapes."""

    explicit = replace(
        _request(tmp_path),
        embedding=ProviderSelection(
            "openai_compatible",
            model="deployment-specific-model",
            endpoint="https://embed.example/v1",
            api_key_environment_variable="EMBED_API_KEY",
            dimension=1024,
        ),
    )
    config = build_run_config(explicit)

    assert config["embedding"]["model"] == "deployment-specific-model"
    assert config["embedding"]["dimension"] == 1024

    implicit = replace(
        explicit,
        embedding=replace(explicit.embedding, dimension=None),
    )
    with pytest.raises(ValueError, match="Embedding dimension is required"):
        build_run_config(implicit)


def test_embedding_provider_catalog_declares_dimensions_for_every_ui_option() -> None:
    """Ensure every embedding selector renders an explicit dimension control."""

    assert all(option.default_dimension for option in EMBEDDING_PROVIDERS)
    assert not any(option.default_dimension for option in LLM_PROVIDERS)


def _request(tmp_path: Path) -> IngestionRequest:
    """Build a complete non-secret local request for app security tests."""

    return IngestionRequest(
        runtime=RuntimeMode.LOCAL,
        job_id="job-1",
        graph_id="graph-1",
        source_kind=SourceKind.LOCAL,
        source={"path": str(tmp_path)},
        ocr=ProviderSelection("fallback"),
        llm=ProviderSelection(
            "openai_compatible",
            model="model",
            endpoint="https://llm.example/v1",
            api_key_environment_variable="LLM_API_KEY",
        ),
        embedding=ProviderSelection(
            "sentence_transformers",
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
        ),
        output=OutputDestination(StorageKind.LOCAL, tmp_path / "out"),
    )
