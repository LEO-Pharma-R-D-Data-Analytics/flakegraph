"""Snowflake Cortex LLM adapter using structured `AI_COMPLETE` calls."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    SnowflakeConnectionConfig,
    as_json_object,
    connect_snowflake,
    scalar_from_first_row,
)
from kg_processor.application.extraction_schema import extraction_response_format
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

_SNOWFLAKE_COMPLETE_SQL = (
    "SELECT AI_COMPLETE(?, ?, PARSE_JSON(?)::OBJECT, PARSE_JSON(?)::OBJECT, TRUE)"
)


class SnowflakeCortexLlmProvider:
    """Executes graph prompts through Snowflake Cortex structured completion."""

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        default_model: str,
        connector_factory: ConnectorFactory | None = None,
    ) -> None:
        self.config = config
        self.default_model = default_model
        self.connector_factory = connector_factory

    def extract_graph(self, chunks: list[Chunk], options: LlmOptions) -> ExtractionResult:
        """Run structured graph extraction and one validation repair if needed."""

        prompt = graph_extraction_prompt(
            chunks,
            options,
            options.extraction_pass,
            options.previous_result,
        )
        payload, metadata = self._complete_structured(
            model=options.model,
            prompt=f"{prompt.system}\n\n{prompt.user}",
            model_parameters={
                "temperature": options.temperature,
                "max_tokens": 4096,
            },
            response_format=extraction_response_format(),
        )
        repair_attempts = 0
        try:
            result = ExtractionResult.model_validate(payload)
        except ValidationError as exc:
            repair_prompt = graph_repair_prompt(
                chunks,
                options,
                json.dumps(payload, sort_keys=True),
                str(exc),
            )
            payload, metadata = self._complete_structured(
                model=options.model,
                prompt=f"{repair_prompt.system}\n\n{repair_prompt.user}",
                model_parameters={
                    "temperature": options.temperature,
                    "max_tokens": 4096,
                },
                response_format=extraction_response_format(),
            )
            result = ExtractionResult.model_validate(payload)
            result.provider_metadata = {
                f"repair_{key}": value for key, value in prompt_metadata(repair_prompt).items()
            }
            repair_attempts = 1
        metadata_from_result = dict(result.provider_metadata)
        result.provider_metadata = {
            **metadata_from_result,
            "provider": "snowflake_cortex",
            "model": metadata.get("model", options.model),
            "usage": metadata.get("usage"),
            "repair_attempts": repair_attempts,
            **prompt_metadata(prompt),
        }
        return result

    def repair_graph_extraction(self, request: GraphRepairRequest) -> ExtractionResult:
        """Ask Cortex to repair a payload that failed extraction validation."""

        repair_prompt = graph_repair_prompt(
            request.chunks,
            request.options,
            json.dumps(request.invalid_result.model_dump(mode="json"), sort_keys=True),
            request.validation_error,
        )
        payload, metadata = self._complete_structured(
            model=request.options.model,
            prompt=f"{repair_prompt.system}\n\n{repair_prompt.user}",
            model_parameters={
                "temperature": request.options.temperature,
                "max_tokens": 4096,
            },
            response_format=extraction_response_format(),
        )
        result = ExtractionResult.model_validate(payload)
        result.provider_metadata = {
            "provider": "snowflake_cortex",
            "model": metadata.get("model", request.options.model),
            "usage": metadata.get("usage"),
            "repair_validation_error": request.validation_error,
            **{f"repair_{key}": value for key, value in prompt_metadata(repair_prompt).items()},
        }
        return result

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Merge entity descriptions with Cortex structured output."""

        prompt = entity_description_merge_prompt(request)
        payload, metadata = self._complete_structured(
            model=self.default_model,
            prompt=f"{prompt.system}\n\n{prompt.user}",
            model_parameters={
                "temperature": 0,
                "max_tokens": 1024,
            },
            response_format=_description_merge_response_format(),
        )
        return DescriptionMergeResult(
            description=str(payload.get("description", "")),
            provider_metadata={
                "provider": "snowflake_cortex",
                "model": metadata.get("model", self.default_model),
                "usage": metadata.get("usage"),
                **prompt_metadata(prompt),
            },
        )

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        """Generate a community report with Cortex structured output."""

        prompt = community_report_prompt(request)
        payload, metadata = self._complete_structured(
            model=self.default_model,
            prompt=f"{prompt.system}\n\n{prompt.user}",
            model_parameters={
                "temperature": 0,
                "max_tokens": 2048,
            },
            response_format=_community_response_format(),
        )
        raw_findings = payload.get("findings", [])
        findings = raw_findings if isinstance(raw_findings, list) else []
        return CommunitySummaryResult(
            title=str(payload.get("title", request.title_seed)),
            summary=str(payload.get("summary", "")),
            rating=_coerce_rating(payload.get("rating", 0.0)),
            rating_explanation=str(payload.get("rating_explanation", "")),
            findings=[
                (str(item.get("summary", "")), str(item.get("explanation", "")))
                for item in findings
                if isinstance(item, dict)
            ],
            suggested_questions=_coerce_string_list(payload.get("suggested_questions", [])),
            provider_metadata={
                "provider": "snowflake_cortex",
                "model": metadata.get("model", self.default_model),
                "usage": metadata.get("usage"),
                **prompt_metadata(prompt),
            },
        )

    def _complete_structured(
        self,
        model: str,
        prompt: str,
        model_parameters: dict[str, object],
        response_format: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, Any]]:
        # The response format is supplied by the shared prompt contract. Cortex
        # metadata stays separate from the normalized payload so traces can show
        # usage/model details without leaking provider-specific response shapes.
        connection = connect_snowflake(self.config, self.connector_factory)
        cursor = connection.cursor()
        try:
            cursor.execute(
                _SNOWFLAKE_COMPLETE_SQL,
                [
                    model,
                    prompt,
                    json.dumps(model_parameters, sort_keys=True),
                    json.dumps(response_format, sort_keys=True),
                ],
            )
            raw_result = scalar_from_first_row(cursor)
        finally:
            cursor.close()
            connection.close()
        details = _snowflake_result_to_object(raw_result)
        payload = _structured_payload(details)
        metadata = {key: value for key, value in details.items() if key != "structured_output"}
        return payload, metadata


def _description_merge_response_format() -> dict[str, object]:
    return {
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
            },
            "required": ["description"],
        },
    }


def _community_response_format() -> dict[str, object]:
    return {
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "rating": {"type": "number"},
                "rating_explanation": {"type": "string"},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["summary", "explanation"],
                    },
                },
                "suggested_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "title",
                "summary",
                "rating",
                "rating_explanation",
                "findings",
                "suggested_questions",
            ],
        },
    }


def _snowflake_result_to_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return as_json_object(value)
    raise ValueError("Snowflake AI_COMPLETE returned an unsupported result type")


def _structured_payload(details: dict[str, Any]) -> dict[str, object]:
    structured_output = details.get("structured_output")
    if isinstance(structured_output, dict):
        return _object_payload(structured_output)
    if isinstance(structured_output, list) and structured_output:
        first_item = structured_output[0]
        if isinstance(first_item, dict):
            raw_message = first_item.get("raw_message")
            if isinstance(raw_message, dict):
                return _object_payload(raw_message)
            return _object_payload(first_item)
    choices = details.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("messages")
        if isinstance(message, str):
            parsed = json.loads(message)
            return _object_payload(parsed)
        if isinstance(message, dict):
            return _object_payload(message)
    return _object_payload(details)


def _object_payload(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    raise ValueError("Snowflake AI_COMPLETE structured output must be a JSON object")


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
