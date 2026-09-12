from __future__ import annotations

import httpx
import pytest

from kg_processor.adapters.embeddings.azure_openai import AzureOpenAIEmbeddingProvider
from kg_processor.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from kg_processor.adapters.llm.azure_openai import AzureOpenAILlmProvider
from kg_processor.adapters.llm.openai_common import ChatCompletion, coerce_string_list
from kg_processor.adapters.llm.openai_compatible import OpenAICompatibleLlmProvider
from kg_processor.adapters.llm.vllm_local import VllmLocalLlmProvider
from kg_processor.ports.embeddings import EmbedOptions
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    DescriptionMergeRequest,
    StructuredCompletionRequest,
)


def test_openai_string_lists_drop_null_and_non_string_values() -> None:
    assert coerce_string_list([None, " Valid question? ", 7, ""]) == ["Valid question?"]


class _MockClient:
    instances = 0

    def __init__(self, timeout: int) -> None:
        self.__class__.instances += 1
        self.timeout = timeout

    def __enter__(self) -> _MockClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, headers: dict[str, str], json: dict[str, object]) -> httpx.Response:
        request = httpx.Request("POST", url, headers=headers)
        if url.endswith("/chat/completions"):
            assert json["max_tokens"] == 8192
            messages = json["messages"]
            assert isinstance(messages, list)
            system_message = messages[0]
            assert isinstance(system_message, dict)
            system_content = str(system_message["content"])
            content: dict[str, object]
            if "merge observed descriptions" in system_content:
                content = {"description": "Alice Smith works at Acme Corp."}
            elif json["model"] == "community":
                content = {
                    "title": "Acme",
                    "summary": "Acme community",
                    "rating": 7,
                    "rating_explanation": "Important because Alice is connected to Acme.",
                    "findings": [{"summary": "Finding", "explanation": "Because"}],
                    "suggested_questions": ["How is Alice connected to Acme?"],
                }
            else:
                content = {
                    "entities": [
                        {
                            "name": "Alice Smith",
                            "type": "PERSON",
                            "description": "Alice Smith is present.",
                            "source_chunk_id": "chunk_1",
                            "confidence": 1,
                            "aliases": [],
                        }
                    ],
                    "relations": [],
                }
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": __import__("json").dumps(content)}}]},
                request=request,
            )
        if url.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
                request=request,
            )
        raise AssertionError(f"Unexpected URL: {url}")


class _StructuredRetryMockClient:
    """Return malformed JSON once, then valid strict output during regeneration.

    Class-level counting spans the adapter's two client contexts.
    """

    call_count = 0

    def __init__(self, timeout: int) -> None:
        """Record the adapter timeout for compatibility with the HTTP client API.

        No real network resources are allocated.
        """

        self.timeout = timeout

    def __enter__(self) -> _StructuredRetryMockClient:
        """Expose this deterministic client through the adapter's context-manager protocol.

        The same fixture instance handles each request context.
        """

        return self

    def __exit__(self, *_args: object) -> None:
        """Close the mock context without suppressing exceptions from adapter logic.

        There are no resources requiring teardown.
        """

        return None

    def post(self, url: str, headers: dict[str, str], json: dict[str, object]) -> httpx.Response:
        """Validate strict transport and alternate malformed then valid response content.

        Both responses retain an ordinary HTTP response shape.
        """

        _StructuredRetryMockClient.call_count += 1
        response_format = json["response_format"]
        assert isinstance(response_format, dict)
        assert response_format["type"] == "json_schema"
        content = "{broken" if self.call_count == 1 else '{"records": []}'
        request = httpx.Request("POST", url, headers=headers)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )


class _VllmMockClient:
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    def __enter__(self) -> _VllmMockClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, headers: dict[str, str], json: dict[str, object]) -> httpx.Response:
        request = httpx.Request("POST", url, headers=headers)
        assert url == "http://localhost:8000/v1/chat/completions"
        assert "Authorization" not in headers
        assert json["model"] == "qwen2.5"
        assert json["max_tokens"] == 16384
        assert json["chat_template_kwargs"] == {"enable_thinking": False}
        content = {
            "entities": [
                {
                    "name": "Alice Smith",
                    "type": "PERSON",
                    "description": "Alice Smith is present.",
                    "source_chunk_id": "chunk_1",
                    "confidence": 1,
                    "aliases": [],
                }
            ],
            "relations": [],
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": __import__("json").dumps(content)}}]},
            request=request,
        )


class _AzureMockClient:
    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    def __enter__(self) -> _AzureMockClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, headers: dict[str, str], json: dict[str, object]) -> httpx.Response:
        request = httpx.Request("POST", url, headers=headers)
        assert "api-version=2025-01-01-preview" in url
        assert headers["api-key"] == "secret"
        if url.endswith("/chat/completions?api-version=2025-01-01-preview"):
            assert json["max_tokens"] == 8192
            assert json["reasoning_effort"] == "none"
            messages = json["messages"]
            assert isinstance(messages, list)
            system_message = messages[0]
            assert isinstance(system_message, dict)
            system_content = str(system_message["content"])
            content: dict[str, object]
            if "merge observed descriptions" in system_content:
                content = {"description": "Alice Smith works at Acme Corp."}
            elif "knowledge-graph community" in system_content:
                content = {
                    "title": "Acme",
                    "summary": "Acme community",
                    "rating": "6.5",
                    "rating_explanation": "Important because the relation is high weight.",
                    "findings": [{"summary": "Finding", "explanation": "Because"}],
                    "suggested_questions": ["What does Acme connect to?"],
                }
            else:
                content = {
                    "entities": [
                        {
                            "name": "Alice Smith",
                            "type": "PERSON",
                            "description": "Alice Smith is present.",
                            "source_chunk_id": "chunk_1",
                            "confidence": 1,
                            "aliases": [],
                        }
                    ],
                    "relations": [],
                }
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": __import__("json").dumps(content)}}]},
                request=request,
            )
        if url.endswith("/embeddings?api-version=2025-01-01-preview"):
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
                request=request,
            )
        raise AssertionError(f"Unexpected URL: {url}")


class _AzureCapabilityNegotiationMockClient:
    """Require the newer token field and default temperature before succeeding."""

    payloads: list[dict[str, object]] = []

    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    def __enter__(self) -> _AzureCapabilityNegotiationMockClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, headers: dict[str, str], json: dict[str, object]) -> httpx.Response:
        """Return the same structured capability errors emitted by newer models."""

        self.__class__.payloads.append(dict(json))
        request = httpx.Request("POST", url, headers=headers)
        if "max_tokens" in json:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "unsupported_parameter",
                        "param": "max_tokens",
                    }
                },
                request=request,
            )
        if "temperature" in json:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "unsupported_value",
                        "param": "temperature",
                    }
                },
                request=request,
            )
        if "reasoning_effort" in json:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "unsupported_parameter",
                        "param": "reasoning_effort",
                    }
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
            request=request,
        )


def test_openai_compatible_llm_runs_structured_and_enrichment_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _MockClient.instances = 0
    monkeypatch.setattr(httpx, "Client", _MockClient)
    provider = OpenAICompatibleLlmProvider("https://example.test/v1", "secret", model="community")
    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="entities",
            model="graph",
            system="Extract entities.",
            user="Alice Smith",
            json_schema={"type": "object"},
            max_tokens=16384,
        )
    )
    summary = provider.summarize_community(
        CommunitySummaryRequest(title_seed="Acme", members=["Acme"], relations=[])
    )
    description = provider.merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice Smith",
            entity_type="PERSON",
            descriptions=["Alice is present.", "Alice works at Acme Corp."],
            evidence=["Alice Smith works at Acme Corp."],
        )
    )

    assert result.payload["entities"][0]["name"] == "Alice Smith"
    assert result.provider_metadata["provider"] == "openai_compatible"
    assert result.provider_metadata["task_name"] == "entities"
    assert summary.title == "Acme"
    assert summary.rating_explanation == "Important because Alice is connected to Acme."
    assert summary.suggested_questions == ["How is Alice connected to Acme?"]
    assert summary.provider_metadata["provider"] == "openai_compatible"
    assert summary.provider_metadata["model"] == "community"
    assert summary.provider_metadata["prompt_name"] == "community_report"
    assert description.description == "Alice Smith works at Acme Corp."
    assert description.provider_metadata["provider"] == "openai_compatible"
    assert description.provider_metadata["model"] == "community"
    assert description.provider_metadata["prompt_name"] == "entity_description_merge"
    assert _MockClient.instances == 1


def test_openai_compatible_llm_requires_explicit_default_model() -> None:
    with pytest.raises(ValueError, match="explicit model"):
        OpenAICompatibleLlmProvider("https://example.test/v1", "secret", model="")


def test_openai_structured_completion_retries_malformed_transport_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure malformed transport JSON triggers exactly one schema-preserving regeneration.

    Repair attempts must appear in provider metadata.
    """

    _StructuredRetryMockClient.call_count = 0
    monkeypatch.setattr(httpx, "Client", _StructuredRetryMockClient)
    provider = OpenAICompatibleLlmProvider("https://example.test/v1", "secret", "model")

    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="records",
            model="model",
            system="Return records.",
            user="Input",
            json_schema={
                "type": "object",
                "properties": {"records": {"type": "array", "items": {"type": "string"}}},
                "required": ["records"],
            },
        )
    )

    assert result.payload == {"records": []}
    assert result.provider_metadata["format_repair_attempts"] == 1
    assert _StructuredRetryMockClient.call_count == 2


def test_openai_structured_completion_retries_empty_message_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry an empty choice message that fails before JSON parsing begins."""

    responses: list[str | ValueError] = [ValueError("empty message"), '{"records":[]}']
    provider = OpenAICompatibleLlmProvider("https://example.test/v1", "secret", "model")

    def chat_content(*_args: object, **_kwargs: object) -> ChatCompletion:
        """Return one transport failure followed by a complete JSON response."""

        response = responses.pop(0)
        if isinstance(response, ValueError):
            raise response
        return ChatCompletion(content=response)

    monkeypatch.setattr(provider, "_chat_content", chat_content)
    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="records",
            model="model",
            system="Return records.",
            user="Input",
            json_schema={"type": "object", "properties": {"records": {"type": "array"}}},
        )
    )

    assert result.payload == {"records": []}
    assert result.provider_metadata["format_repair_attempts"] == 1


def test_vllm_local_llm_uses_openai_compatible_chat_without_required_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "Client", _VllmMockClient)
    provider = VllmLocalLlmProvider("http://localhost:8000/v1", model="qwen2.5")
    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="entities",
            model="qwen2.5",
            system="Extract entities.",
            user="Alice Smith",
            json_schema={"type": "object"},
            max_tokens=16384,
        )
    )

    assert result.payload["entities"][0]["name"] == "Alice Smith"
    assert result.provider_metadata["provider"] == "vllm_local"
    assert provider.capabilities().supports_seed is True
    assert provider.capabilities().max_output_tokens == 16384


def test_openai_compatible_embeddings_validate_dimension(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(httpx, "Client", _MockClient)
    provider = OpenAICompatibleEmbeddingProvider("https://example.test/v1", "secret")

    vectors = provider.embed(["Alice"], EmbedOptions(model="embed", dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]


def test_azure_openai_llm_uses_deployment_url(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(httpx, "Client", _AzureMockClient)
    provider = AzureOpenAILlmProvider(
        "https://example.test",
        "secret",
        "2025-01-01-preview",
        "gpt-4.1-mini",
    )
    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="entities",
            model="gpt-4.1-mini",
            system="Extract entities.",
            user="Alice Smith",
            json_schema={"type": "object"},
            max_tokens=8192,
        )
    )
    description = provider.merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice Smith",
            entity_type="PERSON",
            descriptions=["Alice is present.", "Alice works at Acme Corp."],
            evidence=["Alice Smith works at Acme Corp."],
        )
    )
    summary = provider.summarize_community(
        CommunitySummaryRequest(title_seed="Acme", members=["Acme"], relations=[])
    )

    assert result.payload["entities"][0]["name"] == "Alice Smith"
    assert description.description == "Alice Smith works at Acme Corp."
    assert description.provider_metadata["provider"] == "azure_openai"
    assert description.provider_metadata["model"] == "gpt-4.1-mini"
    assert description.provider_metadata["prompt_name"] == "entity_description_merge"
    assert summary.title == "Acme"
    assert summary.rating == 6.5
    assert summary.rating_explanation == "Important because the relation is high weight."
    assert summary.findings == [("Finding", "Because")]
    assert summary.suggested_questions == ["What does Acme connect to?"]
    assert summary.provider_metadata["provider"] == "azure_openai"
    assert summary.provider_metadata["model"] == "gpt-4.1-mini"
    assert summary.provider_metadata["prompt_name"] == "community_report"


def test_azure_openai_advertises_dense_relation_output_budget() -> None:
    """Allow complete relation objects that exceed the legacy 8K ceiling."""

    provider = AzureOpenAILlmProvider(
        "https://example.openai.azure.com",
        "secret",
        "2025-01-01-preview",
        "deployment",
    )

    assert provider.capabilities().max_output_tokens == 16384


def test_azure_openai_negotiates_newer_chat_parameters(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Cache model capability errors so only the first request requires retries."""

    _AzureCapabilityNegotiationMockClient.payloads = []
    monkeypatch.setattr(httpx, "Client", _AzureCapabilityNegotiationMockClient)
    provider = AzureOpenAILlmProvider(
        "https://example.test",
        "secret",
        "2025-01-01-preview",
        "new-model",
    )
    request = StructuredCompletionRequest(
        task_name="new_model",
        model="new-model",
        system="Return JSON.",
        user="Confirm.",
        json_schema={"type": "object"},
        max_tokens=128,
    )

    first = provider.complete_structured(request)
    second = provider.complete_structured(request)

    assert first.payload == {"ok": True}
    assert second.payload == {"ok": True}
    assert len(_AzureCapabilityNegotiationMockClient.payloads) == 5
    assert "max_tokens" in _AzureCapabilityNegotiationMockClient.payloads[0]
    assert "max_completion_tokens" in _AzureCapabilityNegotiationMockClient.payloads[1]
    assert "temperature" in _AzureCapabilityNegotiationMockClient.payloads[1]
    assert "temperature" not in _AzureCapabilityNegotiationMockClient.payloads[2]
    assert "reasoning_effort" in _AzureCapabilityNegotiationMockClient.payloads[2]
    assert "reasoning_effort" not in _AzureCapabilityNegotiationMockClient.payloads[3]
    assert "reasoning_effort" not in _AzureCapabilityNegotiationMockClient.payloads[4]


def test_azure_openai_embeddings_use_deployment_url(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(httpx, "Client", _AzureMockClient)
    provider = AzureOpenAIEmbeddingProvider(
        "https://example.test",
        "secret",
        "2025-01-01-preview",
    )

    vectors = provider.embed(["Alice"], EmbedOptions(model="text-embedding-3-small", dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        (
            OpenAICompatibleEmbeddingProvider("https://example.test/v1", "secret"),
            "text-embedding-3-small",
        ),
        (
            AzureOpenAIEmbeddingProvider("https://example.test", "secret", "2025-01-01-preview"),
            "ada-deployment",
        ),
    ],
)
def test_embedding_adapters_retry_without_unsupported_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    provider: OpenAICompatibleEmbeddingProvider,
    model: str,
) -> None:
    class _DimensionsFallbackClient:
        payloads: list[dict[str, object]] = []

        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def post(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, object],
        ) -> httpx.Response:
            del headers
            self.__class__.payloads.append(dict(json))
            request = httpx.Request("POST", url)
            if "dimensions" in json:
                return httpx.Response(
                    400, json={"error": "unsupported dimensions"}, request=request
                )
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
                request=request,
            )

    _DimensionsFallbackClient.payloads = []
    monkeypatch.setattr(httpx, "Client", _DimensionsFallbackClient)

    vectors = provider.embed(["Alice"], EmbedOptions(model=model, dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]
    assert "dimensions" in _DimensionsFallbackClient.payloads[0]
    assert "dimensions" not in _DimensionsFallbackClient.payloads[1]
    provider.close()


def test_embedding_adapters_keep_the_requested_width_after_an_unrelated_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One rejected batch must not narrow every later vector in the process."""

    class _OversizedBatchClient:
        payloads: list[dict[str, object]] = []

        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def post(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, object],
        ) -> httpx.Response:
            del headers
            self.__class__.payloads.append(dict(json))
            request = httpx.Request("POST", url)
            inputs = json["input"]
            assert isinstance(inputs, list)
            if len(inputs) > 1:
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "code": "context_length_exceeded",
                            "message": "This model's maximum context length is 8192 tokens",
                            "param": "input",
                        }
                    },
                    request=request,
                )
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
                request=request,
            )

    _OversizedBatchClient.payloads = []
    monkeypatch.setattr(httpx, "Client", _OversizedBatchClient)
    provider = OpenAICompatibleEmbeddingProvider("https://example.test/v1", "secret")
    options = EmbedOptions(model="text-embedding-3-small", dimension=3)

    with pytest.raises(httpx.HTTPStatusError):
        provider.embed(["Alice", "Acme"], options)

    assert provider.embed(["Alice"], options) == [[0.1, 0.2, 0.3]]
    assert all("dimensions" in payload for payload in _OversizedBatchClient.payloads)
    provider.close()


def test_generic_embedding_omits_dimensions_for_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StrictCompatibleClient:
        payloads: list[dict[str, object]] = []

        def __init__(self, timeout: int) -> None:
            self.timeout = timeout

        def post(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, object],
        ) -> httpx.Response:
            del headers
            self.__class__.payloads.append(dict(json))
            request = httpx.Request("POST", url)
            if "dimensions" in json:
                return httpx.Response(
                    400,
                    json={"error": "extra fields are forbidden"},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
                request=request,
            )

    monkeypatch.setattr(httpx, "Client", _StrictCompatibleClient)
    provider = OpenAICompatibleEmbeddingProvider("https://example.test/v1", "secret")

    vectors = provider.embed(["Alice"], EmbedOptions(model="custom-model", dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]
    assert _StrictCompatibleClient.payloads == [{"model": "custom-model", "input": ["Alice"]}]
    provider.close()
