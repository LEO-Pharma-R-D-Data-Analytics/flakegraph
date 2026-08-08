"""Snowflake Cortex LLM adapter using structured `AI_COMPLETE` calls."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from kg_processor.adapters.llm.openai_common import (
    ChatCompletion,
    coerce_rating,
    complete_json_object_with_retry,
    complete_structured_with_retry,
)
from kg_processor.adapters.snowflake import (
    ConnectorFactory,
    ReusableSnowflakeConnections,
    SnowflakeConnectionConfig,
    as_json_object,
    scalar_from_first_row,
)
from kg_processor.application.prompt_registry import (
    community_report_prompt,
    entity_description_merge_prompt,
    prompt_metadata,
)
from kg_processor.application.structured_output import strict_json_schema
from kg_processor.domain.consumption import TokenUsage
from kg_processor.ports.llm import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    LlmCapabilities,
    StructuredCompletionRequest,
    StructuredCompletionResult,
)

_SNOWFLAKE_COMPLETE_SQL = (
    "SELECT AI_COMPLETE("
    "model => ?, prompt => ?, "
    "model_parameters => PARSE_JSON(?)::OBJECT, "
    "response_format => PARSE_JSON(?)::OBJECT, "
    "show_details => TRUE, return_error_details => TRUE)"
)

# Snowflake's structured-output implementation supports JSON Schema structure,
# enums, and references, but deliberately rejects these validation constraints.
# FlakeGraph removes them only from the transport schema and still validates the
# response against its complete Pydantic contract after the provider call.
_UNSUPPORTED_SNOWFLAKE_SCHEMA_KEYWORDS = {
    "contains",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "maxContains",
    "maximum",
    "maxItems",
    "maxLength",
    "maxProperties",
    "minContains",
    "minimum",
    "minItems",
    "minLength",
    "minProperties",
    "multipleOf",
    "patternProperties",
    "propertyNames",
    "uniqueItems",
}


class SnowflakeCortexLlmProvider:
    """Execute provider-neutral LLM tasks through Snowflake Cortex ``AI_COMPLETE``.

    The adapter translates shared JSON schemas and prompt contracts into bound
    Snowflake SQL parameters, then separates normalized payloads from Cortex
    usage metadata for auditable downstream traces.
    """

    def __init__(
        self,
        config: SnowflakeConnectionConfig,
        default_model: str,
        connector_factory: ConnectorFactory | None = None,
        timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        """Configure the session, default model, and enrichment request timeout.

        Extraction requests carry their own timeout through the shared port
        contract. Graph enrichment calls do not, so they use this value.
        """

        if timeout_seconds <= 0:
            raise ValueError("Snowflake Cortex LLM timeout must be positive")
        self.config = config
        self.default_model = default_model
        self.connector_factory = connector_factory
        self.timeout_seconds = timeout_seconds
        self._connections = ReusableSnowflakeConnections(config, connector_factory)

    def close(self) -> None:
        """Release retained Snowflake sessions for embedded callers."""

        self._connections.close()

    def capabilities(self) -> LlmCapabilities:
        """Describe the Cortex capabilities available to extraction orchestration.

        Native strict output is enabled through ``AI_COMPLETE`` response formats;
        seed support is omitted because Cortex model behavior is model-specific.
        """

        return LlmCapabilities(
            strict_json_schema=True,
            native_structured_output=True,
            supports_seed=False,
            max_output_tokens=8192,
        )

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Execute a shared strict request and normalize Cortex output metadata.

        Cortex model-output validation failures participate in the same bounded,
        schema-preserving repair loop as other LLM adapters. Permanent Snowflake
        input, access, or configuration errors fail immediately. Model usage from
        the successful attempt remains in provider metadata while only the
        schema-conforming object is returned as the task payload.
        """

        successful_metadata: dict[str, Any] = {}

        def send(
            messages: list[dict[str, str]],
            _response_format: dict[str, object],
            max_tokens: int,
            _seed: int | None,
        ) -> ChatCompletion:
            """Run one Cortex attempt and expose its object to the shared parser."""

            payload, metadata = self._complete_structured(
                model=request.model,
                prompt="\n\n".join(message["content"] for message in messages),
                model_parameters={
                    "temperature": request.temperature,
                    "max_tokens": max_tokens,
                },
                response_format={
                    "type": "json",
                    "schema": _snowflake_json_schema(request.json_schema),
                },
                timeout_seconds=request.timeout_seconds,
            )
            successful_metadata.clear()
            successful_metadata.update(metadata)
            return ChatCompletion(
                content=json.dumps(payload),
                usage=TokenUsage.from_provider_payload(metadata.get("usage")),
            )

        result = complete_structured_with_retry(
            request,
            provider_name="snowflake_cortex",
            max_output_tokens=self.capabilities().max_output_tokens,
            supports_seed=False,
            send=send,
        )
        return StructuredCompletionResult(
            payload=result.payload,
            usage=result.usage,
            provider_metadata={
                **result.provider_metadata,
                "model": successful_metadata.get("model", request.model),
                "usage": successful_metadata.get("usage"),
            },
        )

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        """Merge entity descriptions with Cortex structured output."""

        prompt = entity_description_merge_prompt(request)
        payload, metadata = self._complete_enrichment_with_retry(
            prompt=f"{prompt.system}\n\n{prompt.user}",
            max_tokens=1024,
            response_format=_description_merge_response_format(),
        )
        return DescriptionMergeResult(
            description=_optional_string(payload.get("description")),
            usage=_metadata_usage(metadata),
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
        payload, metadata = self._complete_enrichment_with_retry(
            prompt=f"{prompt.system}\n\n{prompt.user}",
            max_tokens=2048,
            response_format=_community_response_format(),
        )
        raw_findings = payload.get("findings", [])
        findings = raw_findings if isinstance(raw_findings, list) else []
        return CommunitySummaryResult(
            title=_optional_string(payload.get("title")) or request.title_seed,
            summary=_optional_string(payload.get("summary")),
            rating=_coerce_rating(payload.get("rating", 0.0)),
            rating_explanation=_optional_string(payload.get("rating_explanation")),
            findings=[
                (
                    _optional_string(item.get("summary")),
                    _optional_string(item.get("explanation")),
                )
                for item in findings
                if isinstance(item, dict)
            ],
            suggested_questions=_coerce_string_list(payload.get("suggested_questions", [])),
            usage=_metadata_usage(metadata),
            provider_metadata={
                "provider": "snowflake_cortex",
                "model": metadata.get("model", self.default_model),
                "usage": metadata.get("usage"),
                **prompt_metadata(prompt),
            },
        )

    def _complete_enrichment_with_retry(
        self,
        *,
        prompt: str,
        max_tokens: int,
        response_format: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, Any]]:
        """Retry transient Cortex structured-output validation failures."""

        metadata: dict[str, Any] = {}

        def send() -> ChatCompletion:
            payload, attempt_metadata = self._complete_structured(
                model=self.default_model,
                prompt=prompt,
                model_parameters={"temperature": 0, "max_tokens": max_tokens},
                response_format=response_format,
                timeout_seconds=self.timeout_seconds,
            )
            metadata.clear()
            metadata.update(attempt_metadata)
            return ChatCompletion(
                content=json.dumps(payload),
                usage=TokenUsage.from_provider_payload(attempt_metadata.get("usage")),
            )

        payload, usage = complete_json_object_with_retry(send, model=self.default_model)
        return payload, {**metadata, "token_usage": usage}

    def _complete_structured(
        self,
        model: str,
        prompt: str,
        model_parameters: dict[str, object],
        response_format: dict[str, object],
        timeout_seconds: int,
    ) -> tuple[dict[str, object], dict[str, Any]]:
        # The response format is supplied by the shared prompt contract. Cortex
        # metadata stays separate from the normalized payload so traces can show
        # usage/model details without leaking provider-specific response shapes.
        #
        # The bound is applied per statement rather than through a session-level
        # STATEMENT_TIMEOUT_IN_SECONDS: the session is retained and shared by
        # every call this adapter makes, while the timeout belongs to one
        # request. The connector cancels the running query when the bound
        # elapses, which is what releases the extraction thread; without it a
        # stalled AI_COMPLETE holds that thread for the whole retry budget.
        connection = self._connections.get()
        cursor = connection.cursor()
        try:
            cast(Any, cursor).execute(
                _SNOWFLAKE_COMPLETE_SQL,
                [
                    model,
                    prompt,
                    json.dumps(model_parameters, sort_keys=True),
                    json.dumps(response_format, sort_keys=True),
                ],
                timeout=timeout_seconds,
            )
            raw_result = scalar_from_first_row(cursor)
        except Exception:
            self._connections.invalidate()
            raise
        finally:
            cursor.close()
        result = _snowflake_result_to_object(raw_result)
        details = _unwrap_completion_result(result)
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
    """Normalize the connector's VARIANT representation into a Python object.

    Depending on connector settings and Snowflake's selected overload, a VARIANT
    may arrive as a native mapping or as serialized JSON. ``None`` means the AI
    function failed without error details, so the exception explains how to make
    the underlying provider failure observable.
    """

    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return as_json_object(value)
    if value is None:
        raise RuntimeError(
            "Snowflake AI_COMPLETE returned NULL without error details; "
            "enable return_error_details in the AI_COMPLETE call"
        )
    raise ValueError(
        f"Snowflake AI_COMPLETE returned an unsupported result type: {type(value).__name__}"
    )


def _unwrap_completion_result(result: dict[str, Any]) -> dict[str, Any]:
    """Unwrap Snowflake's ``return_error_details`` envelope or raise its error.

    Snowflake normally turns row-level inference failures into SQL ``NULL``.
    Requesting the envelope keeps a failed document diagnosable while allowing
    successful calls to retain the same structured-output normalization path.
    """

    if "value" not in result and "error" not in result:
        return result
    error = result.get("error")
    if error is not None and str(error).strip():
        message = str(error).strip()
        if message.lower().startswith("json mode output validation error"):
            # A constrained model can occasionally emit malformed JSON despite
            # a valid schema. Raising ValueError routes only this model-output
            # condition through the shared structured-response repair loop.
            raise ValueError(f"Snowflake AI_COMPLETE failed: {message}")
        raise RuntimeError(f"Snowflake AI_COMPLETE failed: {message}")
    return _object_payload(result.get("value"))


def _snowflake_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Create a strict schema containing only constraints Snowflake supports.

    The shared schema is copied before normalization because it is also used for
    local validation, cache fingerprints, and other provider adapters. This
    provider-specific projection prevents Snowflake limitations from weakening
    those contracts globally.
    """

    normalized = strict_json_schema(schema)
    _remove_unsupported_snowflake_constraints(normalized)
    return normalized


def _remove_unsupported_snowflake_constraints(value: object) -> None:
    """Recursively remove documented unsupported Cortex schema keywords in place."""

    if isinstance(value, dict):
        schema_maps = ("properties", "$defs", "definitions", "patternProperties")
        for map_name in schema_maps:
            schema_map = value.get(map_name)
            if isinstance(schema_map, dict):
                for child_schema in schema_map.values():
                    _remove_unsupported_snowflake_constraints(child_schema)
        for keyword in _UNSUPPORTED_SNOWFLAKE_SCHEMA_KEYWORDS:
            value.pop(keyword, None)
        for key, child in value.items():
            if key in schema_maps:
                continue
            _remove_unsupported_snowflake_constraints(child)
    elif isinstance(value, list):
        for child in value:
            _remove_unsupported_snowflake_constraints(child)


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
    raise ValueError("Snowflake AI_COMPLETE response did not contain structured_output or choices")


def _object_payload(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    raise ValueError("Snowflake AI_COMPLETE structured output must be a JSON object")


def _coerce_rating(value: object) -> float:
    """Normalize Cortex ratings through the shared inclusive 0-10 policy."""

    return coerce_rating(value)


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if isinstance(item, str) and (text := item.strip())]


def _optional_string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _metadata_usage(metadata: Mapping[str, Any]) -> TokenUsage:
    """Read the usage accumulated across an enrichment call's retries."""

    usage = metadata.get("token_usage")
    return usage if isinstance(usage, TokenUsage) else TokenUsage()
