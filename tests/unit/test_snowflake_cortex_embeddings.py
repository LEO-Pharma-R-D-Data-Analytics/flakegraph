from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from kg_processor.adapters.embeddings.snowflake_cortex import SnowflakeCortexEmbeddingProvider
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
from kg_processor.ports.embeddings import EmbedOptions


class FakeCursor:
    def __init__(self, rows: list[Sequence[object]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, Sequence[object] | None]] = []
        self.timeouts: list[int | None] = []
        self.index = 0
        self.closed = False

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        timeout: int | None = None,
    ) -> object:
        self.executed.append((sql, params))
        self.timeouts.append(timeout)
        return None

    def fetchone(self) -> Sequence[object] | None:
        row = self.rows[self.index]
        self.index += 1
        return row

    def fetchall(self) -> list[object]:
        params = self.executed[-1][1]
        assert params is not None
        batch_size = len(json.loads(str(params[0])))
        rows = self.rows[self.index : self.index + batch_size]
        self.index += batch_size
        return list(rows)

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
    connection = FakeConnection([(0, [0.1, 0.2, 0.3]), (1, "[0.4, 0.5, 0.6]")])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexEmbeddingProvider(_config(), connector_factory=factory)

    vectors = provider.embed(
        ["Alice", "Acme"],
        EmbedOptions(model="snowflake-arctic-embed-m-v1.5", dimension=3),
    )

    assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert connection.cursor_instance.executed == [
        (
            "SELECT source.index, "
            "AI_EMBED('snowflake-arctic-embed-m-v1.5', source.value::STRING) "
            "FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) AS source "
            "ORDER BY source.index",
            ['["Alice", "Acme"]'],
        )
    ]
    assert connection.cursor_instance.closed
    assert not connection.closed
    provider.close()
    assert connection.closed


def test_snowflake_cortex_embeddings_reject_dimension_mismatch() -> None:
    connection = FakeConnection([(0, [0.1, 0.2])])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexEmbeddingProvider(_config(), connector_factory=factory)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        provider.embed(["Alice"], EmbedOptions(model="embed", dimension=3))


def test_snowflake_cortex_embeddings_enforce_configured_batch_size() -> None:
    connection = FakeConnection([(0, [0.1]), (1, [0.2]), (0, [0.3]), (1, [0.4]), (0, [0.5])])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexEmbeddingProvider(_config(), connector_factory=factory)

    vectors = provider.embed(
        ["one", "two", "three", "four", "five"],
        EmbedOptions(model="embed", dimension=1, batch_size=2),
    )

    assert vectors == [[0.1], [0.2], [0.3], [0.4], [0.5]]
    assert [params for _sql, params in connection.cursor_instance.executed] == [
        ['["one", "two"]'],
        ['["three", "four"]'],
        ['["five"]'],
    ]


def test_snowflake_cortex_embeddings_escape_model_literal() -> None:
    connection = FakeConnection([(0, [0.1])])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexEmbeddingProvider(_config(), connector_factory=factory)

    provider.embed(["Alice"], EmbedOptions(model="model'quoted", dimension=1))

    assert connection.cursor_instance.executed == [
        (
            "SELECT source.index, AI_EMBED('model''quoted', source.value::STRING) "
            "FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) AS source "
            "ORDER BY source.index",
            ['["Alice"]'],
        )
    ]


def test_snowflake_cortex_embeddings_bound_every_batch_statement() -> None:
    """A stalled AI_EMBED must release the calling worker rather than hold it."""

    connection = FakeConnection([(0, [0.1]), (1, [0.2]), (0, [0.3])])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexEmbeddingProvider(
        _config(),
        connector_factory=factory,
        timeout_seconds=45,
    )

    provider.embed(
        ["one", "two", "three"],
        EmbedOptions(model="embed", dimension=1, batch_size=2),
    )

    assert connection.cursor_instance.timeouts == [45, 45]


def test_snowflake_cortex_embeddings_reject_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        SnowflakeCortexEmbeddingProvider(_config(), timeout_seconds=0)


def test_snowflake_cortex_embeddings_reject_missing_batch_rows() -> None:
    """A partial set-based response must not silently misalign later texts."""

    connection = FakeConnection([(0, [0.1])])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexEmbeddingProvider(_config(), connector_factory=factory)

    with pytest.raises(ValueError, match="result count mismatch"):
        provider.embed(["Alice", "Acme"], EmbedOptions(model="embed", dimension=1))


def test_snowflake_cortex_embeddings_reject_reordered_batch_rows() -> None:
    """Input order is part of the embedding port contract and must be verified."""

    connection = FakeConnection([(1, [0.1]), (0, [0.2])])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexEmbeddingProvider(_config(), connector_factory=factory)

    with pytest.raises(ValueError, match="unexpected text index"):
        provider.embed(["Alice", "Acme"], EmbedOptions(model="embed", dimension=1))


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
