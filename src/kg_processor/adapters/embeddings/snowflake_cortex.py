"""Snowflake Cortex embedding adapter using configured `AI_EMBED` models."""

from __future__ import annotations

import json
from collections.abc import Iterable

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    connect_snowflake,
    scalar_from_first_row,
)
from kg_processor.ports.embeddings import EmbedOptions


class SnowflakeCortexEmbeddingProvider:
    """Embeds text with Snowflake Cortex `AI_EMBED`."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        self.config = config
        self.connector_factory = connector_factory

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Call Cortex once per text and validate vector dimensions."""

        if not texts:
            return []
        vectors: list[list[float]] = []
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            sql = _embed_sql(options.model)
            for text in texts:
                cursor.execute(sql, [text])
                vector = _normalize_vector(scalar_from_first_row(cursor))
                if len(vector) != options.dimension:
                    raise ValueError(
                        "Embedding dimension mismatch: "
                        f"expected {options.dimension}, got {len(vector)}"
                    )
                vectors.append(vector)
        finally:
            cursor.close()
            connection.close()
        return vectors


def _embed_sql(model: str) -> str:
    return f"SELECT AI_EMBED({_sql_string(model)}, ?)"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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
