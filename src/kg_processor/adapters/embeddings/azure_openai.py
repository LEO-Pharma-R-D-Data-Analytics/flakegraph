"""Azure OpenAI embedding adapter with deployment-specific endpoints."""

from __future__ import annotations

import httpx

from kg_processor.adapters.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from kg_processor.ports.embeddings import EmbedOptions


class AzureOpenAIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Embeds text through an Azure OpenAI deployment endpoint."""

    def __init__(self, endpoint: str, api_key: str, api_version: str) -> None:
        super().__init__(endpoint, api_key)
        self.api_version = api_version

    def _post_embeddings(
        self,
        client: httpx.Client,
        batch: list[str],
        options: EmbedOptions,
        *,
        include_dimensions: bool = True,
    ) -> httpx.Response:
        """Submit one batch to an Azure deployment-specific endpoint."""

        payload: dict[str, object] = {"input": batch}
        if include_dimensions:
            payload["dimensions"] = options.dimension
        return client.post(
            (
                f"{self.endpoint}/openai/deployments/{options.model}/embeddings"
                f"?api-version={self.api_version}"
            ),
            headers={"api-key": self.api_key},
            json=payload,
        )

    def _uses_dimensions_parameter(
        self,
        options: EmbedOptions,
        include_dimensions: bool,
    ) -> bool:
        """Azure deployments accept dimensions independently of deployment names."""

        del options
        return include_dimensions
