from __future__ import annotations

import httpx
import pytest

from kg_processor.adapters.embeddings.azure_openai import AzureOpenAIEmbeddingProvider
from kg_processor.adapters.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from kg_processor.adapters.llm.azure_openai import AzureOpenAILlmProvider
from kg_processor.adapters.llm.openai_compatible import OpenAICompatibleLlmProvider
from kg_processor.adapters.llm.vllm_local import VllmLocalLlmProvider
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractionResult
from kg_processor.ports.embeddings import EmbedOptions
from kg_processor.ports.llm import CommunitySummaryRequest, DescriptionMergeRequest, LlmOptions


class _MockClient:
    def __init__(self, timeout: int) -> None:
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
        assert json["max_tokens"] == 8192
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


class _RepairMockClient:
    call_count = 0

    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    def __enter__(self) -> _RepairMockClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, headers: dict[str, str], json: dict[str, object]) -> httpx.Response:
        request = httpx.Request("POST", url, headers=headers)
        assert url.endswith("/chat/completions")
        assert json["max_tokens"] == 8192
        _RepairMockClient.call_count += 1
        if _RepairMockClient.call_count == 1:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not-json"}}]},
                request=request,
            )
        messages = json["messages"]
        assert isinstance(messages, list)
        system_message = messages[0]
        user_message = messages[1]
        assert isinstance(system_message, dict)
        assert isinstance(user_message, dict)
        assert "repair malformed knowledge graph extraction output" in str(
            system_message["content"]
        )
        assert '"invalid_response": "not-json"' in str(user_message["content"])
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


class _PreviousResultMockClient:
    user_messages: list[str] = []

    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    def __enter__(self) -> _PreviousResultMockClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(self, url: str, headers: dict[str, str], json: dict[str, object]) -> httpx.Response:
        request = httpx.Request("POST", url, headers=headers)
        assert url.endswith("/chat/completions")
        assert json["max_tokens"] == 8192
        messages = json["messages"]
        assert isinstance(messages, list)
        user_message = messages[1]
        assert isinstance(user_message, dict)
        _PreviousResultMockClient.user_messages.append(str(user_message["content"]))
        content = {
            "entities": [
                {
                    "name": "Acme Corp",
                    "type": "ORGANIZATION",
                    "description": "Acme Corp is present.",
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


def test_openai_compatible_llm_extracts_json(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(httpx, "Client", _MockClient)
    provider = OpenAICompatibleLlmProvider("https://example.test/v1", "secret", model="community")
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith",
        start_offset=0,
        end_offset=11,
        token_count=2,
        content_hash="hash",
    )

    result = provider.extract_graph(
        [chunk],
        LlmOptions(model="graph", entity_types=["PERSON"]),
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

    assert result.entities[0].name == "Alice Smith"
    assert result.provider_metadata["provider"] == "openai_compatible"
    assert result.provider_metadata["prompt_name"] == "graph_extraction"
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


def test_openai_compatible_llm_requires_explicit_default_model() -> None:
    with pytest.raises(ValueError, match="explicit model"):
        OpenAICompatibleLlmProvider("https://example.test/v1", "secret", model="")


def test_vllm_local_llm_uses_openai_compatible_chat_without_required_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "Client", _VllmMockClient)
    provider = VllmLocalLlmProvider("http://localhost:8000/v1", model="qwen2.5")
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith",
        start_offset=0,
        end_offset=11,
        token_count=2,
        content_hash="hash",
    )

    result = provider.extract_graph(
        [chunk],
        LlmOptions(model="qwen2.5", entity_types=["PERSON"]),
    )

    assert result.entities[0].name == "Alice Smith"
    assert result.provider_metadata["provider"] == "vllm_local"


def test_openai_compatible_llm_repairs_malformed_extraction_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RepairMockClient.call_count = 0
    monkeypatch.setattr(httpx, "Client", _RepairMockClient)
    provider = OpenAICompatibleLlmProvider("https://example.test/v1", "secret", model="graph")
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith",
        start_offset=0,
        end_offset=11,
        token_count=2,
        content_hash="hash",
    )

    result = provider.extract_graph(
        [chunk],
        LlmOptions(model="graph", entity_types=["PERSON"]),
    )

    assert _RepairMockClient.call_count == 2
    assert result.entities[0].name == "Alice Smith"
    assert result.provider_metadata["repair_attempts"] == 1
    assert result.provider_metadata["repair_prompt_name"] == "graph_repair"


def test_openai_compatible_llm_includes_previous_observations_in_gleaning_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _PreviousResultMockClient.user_messages = []
    monkeypatch.setattr(httpx, "Client", _PreviousResultMockClient)
    provider = OpenAICompatibleLlmProvider("https://example.test/v1", "secret", model="graph")
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith works at Acme Corp.",
        start_offset=0,
        end_offset=31,
        token_count=6,
        content_hash="hash",
    )

    provider.extract_graph(
        [chunk],
        LlmOptions(
            model="graph",
            entity_types=["PERSON", "ORGANIZATION"],
            extraction_pass=1,
            previous_result=ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Alice is present.",
                        source_chunk_id="chunk_1",
                    )
                ]
            ),
        ),
    )

    assert len(_PreviousResultMockClient.user_messages) == 1
    assert "Previously accepted observations" in _PreviousResultMockClient.user_messages[0]
    assert "Alice Smith" in _PreviousResultMockClient.user_messages[0]


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
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith",
        start_offset=0,
        end_offset=11,
        token_count=2,
        content_hash="hash",
    )

    result = provider.extract_graph(
        [chunk],
        LlmOptions(model="gpt-4.1-mini", entity_types=["PERSON"]),
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

    assert result.entities[0].name == "Alice Smith"
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


def test_azure_openai_embeddings_use_deployment_url(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(httpx, "Client", _AzureMockClient)
    provider = AzureOpenAIEmbeddingProvider(
        "https://example.test",
        "secret",
        "2025-01-01-preview",
    )

    vectors = provider.embed(["Alice"], EmbedOptions(model="text-embedding-3-small", dimension=3))

    assert vectors == [[0.1, 0.2, 0.3]]
