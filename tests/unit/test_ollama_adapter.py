"""Contracts for Ollama's native structured chat transport."""

from __future__ import annotations

from typing import Any

import httpx

from kg_processor.adapters.llm.ollama import OllamaLlmProvider
from kg_processor.ports.llm import StructuredCompletionRequest


class _OllamaMockClient:
    """Capture one native request and return a schema-conforming response."""

    request_payload: dict[str, Any] = {}
    request_url = ""

    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    def __enter__(self) -> _OllamaMockClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def post(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> httpx.Response:
        """Record transport fields and return native Ollama message content."""

        _ = headers
        self.__class__.request_url = url
        self.__class__.request_payload = json
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": '{"ok":true}'}},
            request=request,
        )


def test_ollama_uses_native_schema_and_disables_thinking(
    monkeypatch: Any,
) -> None:
    """Reserve output tokens for structured content rather than reasoning traces."""

    monkeypatch.setattr("kg_processor.adapters.http.httpx.Client", _OllamaMockClient)
    provider = OllamaLlmProvider(
        endpoint="http://localhost:11434/v1",
        model="qwen3.8:27b-q4_K_M",
    )
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="ollama_contract",
            model="qwen3.8:27b-q4_K_M",
            system="Return JSON.",
            user="Confirm.",
            json_schema=schema,
            max_tokens=128,
        )
    )

    payload = _OllamaMockClient.request_payload
    assert _OllamaMockClient.request_url == "http://localhost:11434/api/chat"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"] == schema
    assert payload["options"] == {
        "temperature": 0.0,
        "num_predict": 128,
        "num_ctx": 32768,
    }
    assert result.payload == {"ok": True}
    assert result.provider_metadata["provider"] == "ollama"
