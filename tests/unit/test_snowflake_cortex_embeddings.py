from __future__ import annotations

from collections.abc import Sequence

import pytest

from kg_processor.adapters.embeddings.snowflake_cortex import SnowflakeCortexEmbeddingProvider
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
from kg_processor.ports.embeddings import EmbedOptions


class FakeCursor:
    def __init__(self, rows: list[Sequence[object]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, Sequence[object] | None]] = []
        self.index = 0
        self.closed = False

    def execute(self, sql: str, params: Sequence[object] | None = None) -> object:
        self.executed.append((sql, params))
        return None

    def fetchone(self) -> Sequence[object] | None:
        row = self.rows[self.index]
        self.index += 1
        return row

    def fetchall(self) -> list[object]:
        return []

    def close(self) -> object:
        self.closed = True
        return None


class FakeConnection:
    def __init__(self, rows: list[Sequence[object]]) -> None:
        self.cursor_instance = FakeCursor(rows)
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> object:
        return None

    def rollback(self) -> object:
        return None

    def close(self) -> object:
        self.closed = True
        return None


def test_snowflake_cortex_embeddings_call_ai_embed_and_validate_dimension() -> None:
    connection = FakeConnection([([0.1, 0.2, 0.3],), ("[0.4, 0.5, 0.6]",)])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexEmbeddingProvider(_config(), connector_factory=factory)

    vectors = provider.embed(
        ["Alice", "Acme"],
        EmbedOptions(model="snowflake-arctic-embed-m-v1.5", dimension=3),
    )

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert connection.cursor_instance.executed == [
        ("SELECT AI_EMBED('snowflake-arctic-embed-m-v1.5', ?)", ["Alice"]),
        ("SELECT AI_EMBED('snowflake-arctic-embed-m-v1.5', ?)", ["Acme"]),
    ]
    assert connection.cursor_instance.closed
    assert connection.closed


def test_snowflake_cortex_embeddings_reject_dimension_mismatch() -> None:
    connection = FakeConnection([([0.1, 0.2],)])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexEmbeddingProvider(_config(), connector_factory=factory)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        provider.embed(["Alice"], EmbedOptions(model="embed", dimension=3))


def test_snowflake_cortex_embeddings_escape_model_literal() -> None:
    connection = FakeConnection([([0.1],)])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexEmbeddingProvider(_config(), connector_factory=factory)

    provider.embed(["Alice"], EmbedOptions(model="model'quoted", dimension=1))

    assert connection.cursor_instance.executed == [
        ("SELECT AI_EMBED('model''quoted', ?)", ["Alice"])
    ]


def _config() -> SnowflakeConnectionConfig:
    return SnowflakeConnectionConfig(
        account="account",
        host=None,
        user="user",
        password="password",
        authenticator=None,
        private_key_path=None,
        database="DB",
        schema_name="SCHEMA",
        role="ROLE",
        warehouse="WH",
    )
