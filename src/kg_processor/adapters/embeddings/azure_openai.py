"""Azure OpenAI embedding adapter with deployment-specific endpoints."""

from __future__ import annotations

import httpx

from kg_processor.ports.embeddings import EmbedOptions


class AzureOpenAIEmbeddingProvider:
    """Embeds text through an Azure OpenAI deployment endpoint."""

    def __init__(self, endpoint: str, api_key: str, api_version: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version

    def embed(self, texts: list[str], options: EmbedOptions) -> list[list[float]]:
        """Batch texts through Azure OpenAI and validate returned dimensions."""

        if not texts:
            return []
        vectors: list[list[float]] = []
        with httpx.Client(timeout=120) as client:
            for start in range(0, len(texts), options.batch_size):
                batch = texts[start : start + options.batch_size]
                response = client.post(
                    (
                        f"{self.endpoint}/openai/deployments/{options.model}/embeddings"
                        f"?api-version={self.api_version}"
                    ),
                    headers={"api-key": self.api_key},
                    json={"input": batch},
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
