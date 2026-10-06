from __future__ import annotations

import pytest
from snowflake_fakes import CONFIG, FakeConnection

from kg_processor.adapters.embeddings.snowflake_cortex import SnowflakeCortexEmbeddingProvider
from kg_processor.ports.embeddings import EMBEDDING_TIMEOUT_SECONDS, EmbedOptions


def test_snowflake_cortex_embeddings_call_ai_embed_and_validate_dimension() -> None:
    connection = FakeConnection(result_sets=[[(0, [0.1, 0.2, 0.3]), (1, "[0.4, 0.5, 0.6]")]])
    provider = SnowflakeCortexEmbeddingProvider(CONFIG, connector_factory=lambda **_: connection)

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
    connection = FakeConnection(result_sets=[[(0, [0.1, 0.2])]])
    provider = SnowflakeCortexEmbeddingProvider(CONFIG, connector_factory=lambda **_: connection)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        provider.embed(["Alice"], EmbedOptions(model="embed", dimension=3))


def test_snowflake_cortex_embeddings_enforce_configured_batch_size() -> None:
    connection = FakeConnection(
        result_sets=[[(0, [0.1]), (1, [0.2])], [(0, [0.3]), (1, [0.4])], [(0, [0.5])]]
    )
    provider = SnowflakeCortexEmbeddingProvider(CONFIG, connector_factory=lambda **_: connection)

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
    connection = FakeConnection(result_sets=[[(0, [0.1])]])
    provider = SnowflakeCortexEmbeddingProvider(CONFIG, connector_factory=lambda **_: connection)

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

    connection = FakeConnection(result_sets=[[(0, [0.1]), (1, [0.2])], [(0, [0.3])]])
    provider = SnowflakeCortexEmbeddingProvider(CONFIG, connector_factory=lambda **_: connection)

    provider.embed(
        ["one", "two", "three"],
        EmbedOptions(model="embed", dimension=1, batch_size=2),
    )

    assert connection.cursor_instance.timeouts == [EMBEDDING_TIMEOUT_SECONDS] * 2


def test_snowflake_cortex_embeddings_reject_missing_batch_rows() -> None:
    """A partial set-based response must not silently misalign later texts."""

    connection = FakeConnection(result_sets=[[(0, [0.1])]])
    provider = SnowflakeCortexEmbeddingProvider(CONFIG, connector_factory=lambda **_: connection)

    with pytest.raises(ValueError, match="result count mismatch"):
        provider.embed(["Alice", "Acme"], EmbedOptions(model="embed", dimension=1))


def test_snowflake_cortex_embeddings_reject_reordered_batch_rows() -> None:
    """Input order is part of the embedding port contract and must be verified."""

    connection = FakeConnection(result_sets=[[(1, [0.1]), (0, [0.2])]])
    provider = SnowflakeCortexEmbeddingProvider(CONFIG, connector_factory=lambda **_: connection)

    with pytest.raises(ValueError, match="unexpected text index"):
        provider.embed(["Alice", "Acme"], EmbedOptions(model="embed", dimension=1))
