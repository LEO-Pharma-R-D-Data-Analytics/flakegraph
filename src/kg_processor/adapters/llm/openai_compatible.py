"""OpenAI-compatible chat adapter for graph extraction and report prompts."""

from __future__ import annotations

import json

import httpx
from pydantic import ValidationError

from kg_processor.application.prompt_registry import (
    community_report_prompt,
    entity_description_merge_prompt,
    graph_extraction_prompt,
    graph_repair_prompt,
    prompt_metadata,
)
from kg_processor.domain.graph import Chunk, ExtractionResult
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    GraphRepairRequest,
    LlmOptions,
)

_CHAT_MAX_TOKENS = 8192


class OpenAICompatibleLlmProvider:
    """Executes graph prompts against an OpenAI-compatible chat API."""

    def __init__(
        self,
        endpoint: str,
        api_key: str | None,
        model: str,
        provider_name: str = "openai_compatible",
    ) -> None:
        if not model.strip():
            raise ValueError("OpenAI-compatible LLM requires an explicit model")
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider_name = provider_name

    def extract_graph(self, chunks: list[Chunk], options: LlmOptions) -> ExtractionResult:
        """Run graph extraction and attach provider prompt/model metadata."""

        prompt = graph_extraction_prompt(
            chunks,
            options,
            options.extraction_pass,
            options.previous_result,
        )
        raw_content = self._chat_content(
            model=options.model,
            timeout_seconds=options.timeout_seconds,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        )
        result, repair_attempts = self._parse_extraction_or_repair(
            raw_content,
            chunks,
            options,
        )
        metadata = dict(result.provider_metadata)
        metadata.update(
            {
                "provider": self.provider_name,
                "model": options.model,
                "repair_attempts": repair_attempts,
                **prompt_metadata(prompt),
            }
        )
        result.provider_metadata = metadata
        return result

    def repair_graph_extraction(self, request: GraphRepairRequest) -> ExtractionResult:
        """Ask the chat API to repair an invalid extraction payload."""

        repair_prompt = graph_repair_prompt(
            request.chunks,
            request.options,
            json.dumps(request.invalid_result.model_dump(mode="json"), sort_keys=True),
            request.validation_error,
        )
        repair_payload = self._chat(
            model=request.options.model,
            timeout_seconds=request.options.timeout_seconds,
            messages=[
                {"role": "system", "content": repair_prompt.system},
                {"role": "user", "content": repair_prompt.user},
            ],
        )
        result = ExtractionResult.model_validate(repair_payload)
        result.provider_metadata = {
            "provider": self.provider_name,
            "model": request.options.model,
            "repair_validation_error": request.validation_error,
            **{f"repair_{key}": value for key, value in prompt_metadata(repair_prompt).items()},
        }
        return result

    def _parse_extraction_or_repair(
        self,
        raw_content: str,
        chunks: list[Chunk],
        options: LlmOptions,
    ) -> tuple[ExtractionResult, int]:
        try:
            return ExtractionResult.model_validate(_parse_json_object(raw_content)), 0
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            repair_prompt = graph_repair_prompt(chunks, options, raw_content, str(exc))
        repair_payload = self._chat(
            model=options.model,
            timeout_seconds=options.timeout_seconds,
            messages=[
                {"role": "system", "content": repair_prompt.system},
                {"role": "user", "content": repair_prompt.user},
            ],
        )
        result = ExtractionResult.model_validate(repair_payload)
        result.provider_metadata = {
            f"repair_{key}": value for key, value in prompt_metadata(repair_prompt).items()
        }
        return result, 1

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Merge repeated entity descriptions through the configured model."""

        prompt = entity_description_merge_prompt(request)
        payload = self._chat(
            model=self.model,
            timeout_seconds=120,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        )
        return DescriptionMergeResult(
            description=str(payload.get("description", "")),
            provider_metadata={
                "provider": self.provider_name,
                "model": self.model,
                **prompt_metadata(prompt),
            },
        )

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        """Summarize a detected community through the configured model."""

        prompt = community_report_prompt(request)
        payload = self._chat(
            model=self.model,
            timeout_seconds=120,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        )
        raw_findings = payload.get("findings", [])
        findings = raw_findings if isinstance(raw_findings, list) else []
        rating = payload.get("rating", 0.0)
        # Providers sometimes return optional community fields as the wrong
        # scalar type. Coercion is intentionally narrow: keep grounded strings
        # and drop malformed lists rather than guessing new content.
        return CommunitySummaryResult(
            title=str(payload.get("title", request.title_seed)),
            summary=str(payload.get("summary", "")),
            rating=_coerce_rating(rating),
            rating_explanation=str(payload.get("rating_explanation", "")),
            findings=[
                (str(item.get("summary", "")), str(item.get("explanation", "")))
                for item in findings
                if isinstance(item, dict)
            ],
            suggested_questions=_coerce_string_list(payload.get("suggested_questions", [])),
            provider_metadata={
                "provider": self.provider_name,
                "model": self.model,
                **prompt_metadata(prompt),
            },
        )

    def _chat(
        self,
        model: str,
        timeout_seconds: int,
        messages: list[dict[str, str]],
    ) -> dict[str, object]:
        return _parse_json_object(self._chat_content(model, timeout_seconds, messages))

    def _chat_content(
        self,
        model: str,
        timeout_seconds: int,
        messages: list[dict[str, str]],
    ) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(
                f"{self.endpoint}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "temperature": 0,
                    # Graph extraction asks for structured JSON rather than a
                    # short chat answer, so give compatible servers enough room
                    # to finish the object before parsing or repair kicks in.
                    "max_tokens": _CHAT_MAX_TOKENS,
                    "response_format": {"type": "json_object"},
                    "messages": messages,
                },
            )
            response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])


def _coerce_rating(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]


def _parse_json_object(content: str) -> dict[str, object]:
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    return parsed
