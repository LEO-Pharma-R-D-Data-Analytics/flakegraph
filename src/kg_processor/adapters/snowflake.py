"""Shared Snowflake connector helpers.

The helpers isolate connection kwargs, stage path normalization, and JSON result
parsing so individual adapters can focus on their table/function contracts.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from kg_processor.adapters.auth.snowflake import SnowflakeAuthConfig, SnowflakeAuthProvider
from kg_processor.ports.auth import AuthProvider

_UNQUOTED_STAGE_RE = re.compile(
    r"^@[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*){0,2}(?:/[A-Za-z0-9_./=$-]*)?$"
)


class SnowflakeCursor(Protocol):
    """Minimal DB-API cursor surface used by Snowflake adapters."""

    def execute(self, sql: str, params: Sequence[object] | None = None) -> object:
        """Execute SQL with qmark parameters and return the connector result."""
        ...

    def fetchone(self) -> Sequence[object] | None:
        """Fetch a single Snowflake row or None."""
        ...

    def fetchall(self) -> list[object]:
        """Fetch all remaining Snowflake rows."""
        ...

    def close(self) -> object:
        """Release cursor resources held by the connector."""
        ...


class SnowflakeConnection(Protocol):
    """Minimal DB-API connection surface used by Snowflake adapters."""

    def cursor(self) -> SnowflakeCursor:
        """Create a cursor for executing Snowflake statements."""
        ...

    def commit(self) -> object:
        """Commit the current Snowflake transaction."""
        ...

    def rollback(self) -> object:
        """Roll back the current Snowflake transaction."""
        ...

    def close(self) -> object:
        """Close the Snowflake connection."""
        ...


ConnectorFactory = Callable[..., SnowflakeConnection]


@dataclass(frozen=True)
class SnowflakeConnectionConfig:
    """Connection and session context needed by Snowflake adapters."""

    account: str | None
    host: str | None
    user: str | None
    password: str | None
    authenticator: str | None
    private_key_path: Path | None
    database: str
    schema_name: str
    role: str | None
    warehouse: str | None
    oauth_token: str | None = None
    oauth_token_path: Path | None = Path("/snowflake/session/token")
    auth_provider: AuthProvider | None = None

    def connect_kwargs(self) -> dict[str, object]:
        """Build connector kwargs from auth plus role/warehouse context."""

        kwargs: dict[str, object] = {
            "database": self.database,
            # Every Snowflake adapter in this package uses DB-API qmark
            # placeholders. The Python connector defaults to pyformat, which
            # tries to apply Python percent-formatting to queries containing
            # `?` and fails before the statement reaches Snowflake.
            "paramstyle": "qmark",
            "schema": self.schema_name,
        }
        kwargs.update(self._auth_provider().connection_kwargs())
        session_context = {
            "role": self.role,
            "warehouse": self.warehouse,
        }
        kwargs.update({key: value for key, value in session_context.items() if value})
        return kwargs

    def _auth_provider(self) -> AuthProvider:
        if self.auth_provider is not None:
            return self.auth_provider
        return SnowflakeAuthProvider(
            SnowflakeAuthConfig(
                account=self.account,
                host=self.host,
                user=self.user,
                password=self.password,
                authenticator=self.authenticator,
                private_key_path=self.private_key_path,
                oauth_token=self.oauth_token,
                oauth_token_path=self.oauth_token_path,
            )
        )


def load_snowflake_connector() -> ConnectorFactory:
    """Import the optional Snowflake connector only when a Snowflake path is used."""

    module = importlib.import_module("snowflake.connector")
    connect = module.connect
    if not callable(connect):
        raise RuntimeError("snowflake.connector.connect is not callable")
    return cast(ConnectorFactory, connect)


def connect_snowflake(
    config: SnowflakeConnectionConfig,
    connector_factory: ConnectorFactory | None = None,
) -> SnowflakeConnection:
    """Open a Snowflake connection using the provided or default connector factory."""

    factory = connector_factory or load_snowflake_connector()
    return factory(**config.connect_kwargs())


def validate_stage_location(value: str) -> str:
    """Validate an unquoted Snowflake stage URI before interpolating it into SQL."""

    if not _UNQUOTED_STAGE_RE.fullmatch(value):
        raise ValueError(
            "Snowflake stage locations must use unquoted identifiers and safe path "
            f"characters, got: {value}"
        )
    return value


def split_stage_uri(source_uri: str) -> tuple[str, str]:
    """Split an @stage/path URI into the stage name and relative path."""

    value = validate_stage_location(source_uri)
    without_at = value[1:]
    if "/" not in without_at:
        raise ValueError(f"Stage file URI must include a relative path: {source_uri}")
    stage_name, relative_path = without_at.split("/", 1)
    if not relative_path:
        raise ValueError(f"Stage file URI must include a relative path: {source_uri}")
    return f"@{stage_name}", relative_path


def stage_path(stage: str, prefix: str | None = None) -> str:
    """Join and validate a Snowflake stage location and optional prefix."""

    base = validate_stage_location(stage.rstrip("/"))
    if not prefix:
        return base
    normalized_prefix = prefix.strip("/")
    return validate_stage_location(f"{base}/{normalized_prefix}") if normalized_prefix else base


def scalar_from_first_row(cursor: SnowflakeCursor) -> object:
    """Return the first column of the first row or raise on an empty result."""

    row = cursor.fetchone()
    if row is None or len(row) == 0:
        raise RuntimeError("Snowflake query returned no rows")
    return row[0]


def as_json_object(value: object) -> dict[str, Any]:
    """Normalize Snowflake VARIANT/OBJECT results into a Python mapping."""

    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Expected Snowflake result to be a JSON object")
