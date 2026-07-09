"""Snowflake authentication provider implementations.

Snowflake connection targets can be the same while credentials differ by
runtime: SPCS receives a short-lived OAuth token file, on-prem jobs may use
key-pair JWT, and local development can use browser/password-style connector
auth. Keeping those branches here makes the rest of the Snowflake adapters care
only about query contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SnowflakeAuthConfig:
    """Credential inputs supported by Snowflake connector authentication modes."""

    account: str | None
    host: str | None
    user: str | None
    password: str | None
    authenticator: str | None
    private_key_path: Path | None
    oauth_token: str | None = None
    oauth_token_path: Path | None = Path("/snowflake/session/token")


@dataclass(frozen=True)
class SnowflakeAuthProvider:
    """Builds Snowflake connector kwargs from local, on-prem, or SPCS credentials."""

    config: SnowflakeAuthConfig

    def connection_kwargs(self) -> dict[str, object]:
        """Return authentication kwargs while omitting unset credential fields."""

        kwargs: dict[str, object] = {}
        optional = {
            "account": self.config.account,
            "host": self.config.host,
            "user": self.config.user,
            "password": self.config.password,
            "authenticator": self.config.authenticator,
        }
        kwargs.update({key: value for key, value in optional.items() if value})
        if self.config.authenticator == "oauth":
            token = self._oauth_token()
            if token:
                kwargs["token"] = token
        if self.config.private_key_path:
            kwargs["private_key_file"] = str(self.config.private_key_path)
        return kwargs

    def _oauth_token(self) -> str | None:
        if self.config.oauth_token:
            return self.config.oauth_token
        if self.config.oauth_token_path and self.config.oauth_token_path.exists():
            # SPCS refreshes this file while the container is running. Reading
            # it at connection time avoids carrying an expired token between
            # queue-draining batches.
            return self.config.oauth_token_path.read_text(encoding="utf-8").strip()
        return None
