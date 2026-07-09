"""vLLM adapter implemented through the OpenAI-compatible chat surface."""

from __future__ import annotations

from kg_processor.adapters.llm.openai_compatible import OpenAICompatibleLlmProvider


class VllmLocalLlmProvider(OpenAICompatibleLlmProvider):
    """Specializes the OpenAI-compatible adapter for local vLLM servers."""

    def __init__(self, endpoint: str, api_key: str | None = None, model: str = "") -> None:
        super().__init__(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            provider_name="vllm_local",
        )
