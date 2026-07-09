"""OpenAI-compatible embedding adapter with strict dimension validation."""

from __future__ import annotations

import httpx

from kg_processor.ports.embeddings import EmbedOptions


class OpenAICompatibleEmbeddingProvider:
    """Embeds text through any OpenAI-compatible `/embeddings` endpoint."""

    def __init__(self, endpoint: str, api_key: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Batch texts through the endpoint and validate returned dimensions."""

        if not texts:
            return []
        vectors: list[list[float]] = []
        with httpx.Client(timeout=120) as client:
            for start in range(0, len(texts), options.batch_size):
                batch = texts[start : start + options.batch_size]
                response = client.post(
                    f"{self.endpoint}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": options.model, "input": batch},
                )
                response.raise_for_status()
                data = response.json()["data"]
                ordered = sorted(data, key=lambda item: item["index"])
                vectors.extend(item["embedding"] for item in ordered)
        for vector in vectors:
            if len(vector) != options.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {options.dimension}, got {len(vector)}"
                )
        return vectors
