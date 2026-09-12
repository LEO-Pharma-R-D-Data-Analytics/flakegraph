"""Snowflake Cortex embedding adapter using configured `AI_EMBED` models."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any, cast

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    ReusableSnowflakeConnections,
    SnowflakeConnectionConfig,
    quote_sql_string,
)
from kg_processor.ports.embeddings import EmbedOptions

_INDEXED_VECTOR_COLUMNS = 2
# One batch of short texts through a hosted embedding model. The bound matches
# the HTTP embedding adapters so a stalled provider blocks a chunking worker for
# the same length of time whichever provider is configured.
DEFAULT_EMBEDDING_TIMEOUT_SECONDS = 120


class SnowflakeCortexEmbeddingProvider:
    """Embeds text with Snowflake Cortex `AI_EMBED`."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
        timeout_seconds: int = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    ) -> None:
        """Configure the session and the per-batch statement timeout."""

        if timeout_seconds <= 0:
            raise ValueError("Snowflake Cortex embedding timeout must be positive")
        self.config = config
        self.connector_factory = connector_factory
        self.timeout_seconds = timeout_seconds
        self._connections = ReusableSnowflakeConnections(config, connector_factory)

    def close(self) -> None:
        """Release retained Snowflake sessions for embedded callers."""

        self._connections.close()

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Embed ordered text in bounded, set-based Cortex queries.

        ``FLATTEN`` turns the bound JSON array into rows inside Snowflake, where
        ``AI_EMBED`` can evaluate every text in one statement. This avoids one
        network and authentication round trip per chunk while preserving input
        order through the array index. Batching is enforced here because callers
        may pass a complete corpus and Snowflake limits bound expression size.

        The bound is applied per statement rather than through a session-level
        STATEMENT_TIMEOUT_IN_SECONDS: the session is retained and shared by every
        batch, and the connector cancels the running query when the bound
        elapses, which is what releases the calling worker.
        """

        if not texts:
            return []
        vectors: list[list[float]] = []
        connection = self._connections.get()
        cursor = connection.cursor()
        try:
            for start in range(0, len(texts), options.batch_size):
                batch = texts[start : start + options.batch_size]
                cast(Any, cursor).execute(
                    _embed_batch_sql(options.model),
                    [json.dumps(batch)],
                    timeout=self.timeout_seconds,
                )
                rows = cursor.fetchall()
                if len(rows) != len(batch):
                    raise ValueError(
                        "Snowflake AI_EMBED result count mismatch: "
                        f"expected {len(batch)}, got {len(rows)}"
                    )
                for expected_index, row in enumerate(rows):
                    index, raw_vector = _indexed_vector(row)
                    if index != expected_index:
                        raise ValueError(
                            "Snowflake AI_EMBED returned an unexpected text index: "
                            f"expected {expected_index}, got {index}"
                        )
                    vector = _normalize_vector(raw_vector)
                    if len(vector) != options.dimension:
                        raise ValueError(
                            "Embedding dimension mismatch: "
                            f"expected {options.dimension}, got {len(vector)}"
                        )
                    vectors.append(vector)
        except Exception:
            self._connections.invalidate()
            raise
        finally:
            cursor.close()
        return vectors


def _embed_batch_sql(model: str) -> str:
    """Render a model-safe, order-preserving set-based embedding statement."""

    return (
        "SELECT source.index, "
        f"AI_EMBED({quote_sql_string(model)}, source.value::STRING) "
        "FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) AS source "
        "ORDER BY source.index"
    )


def _indexed_vector(row: object) -> tuple[int, object]:
    """Validate and split one indexed row returned by the batch query."""

    if not isinstance(row, Sequence) or isinstance(row, str | bytes | bytearray):
        raise ValueError("Snowflake AI_EMBED returned a malformed result row")
    if len(row) < _INDEXED_VECTOR_COLUMNS or not isinstance(row[0], int):
        raise ValueError("Snowflake AI_EMBED returned a malformed indexed vector")
    return row[0], row[1]


def _normalize_vector(value: object) -> list[float]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return _normalize_vector(parsed)
    if isinstance(value, Iterable) and not isinstance(value, bytes | bytearray | dict):
        vector: list[float] = []
        for item in value:
            if not isinstance(item, int | float):
                raise ValueError("Snowflake AI_EMBED returned a non-numeric vector value")
            vector.append(float(item))
        return vector
    raise ValueError("Snowflake AI_EMBED returned an unsupported vector value")
