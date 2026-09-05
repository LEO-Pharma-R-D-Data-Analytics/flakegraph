from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from kg_processor.adapters.llm.snowflake_cortex import (
    SnowflakeCortexLlmProvider,
    _snowflake_json_schema,
    _snowflake_result_to_object,
    _structured_payload,
    _unwrap_completion_result,
)
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    DescriptionMergeRequest,
    StructuredCompletionRequest,
)


class FakeCursor:
    """Record bound SQL calls and return deterministic connector rows."""

    def __init__(self, rows: list[Sequence[object]]) -> None:
        self.rows = rows
        self.index = 0
        self.executed: list[tuple[str, Sequence[object] | None]] = []
        self.timeouts: list[int | None] = []
        self.closed = False

    def execute(
        self,
        sql: str,
        params: Sequence[object] | None = None,
        *,
        timeout: int | None = None,
    ) -> object:
        self.executed.append((sql, params))
        self.timeouts.append(timeout)
        return None

    def fetchone(self) -> Sequence[object] | None:
        row = self.rows[self.index]
        self.index += 1
        return row

    def fetchall(self) -> list[object]:
        return []

    def close(self) -> object:
        self.closed = True
        return None


class FakeConnection:
    """Expose the cursor methods used by the Snowflake adapter."""

    def __init__(self, rows: list[Sequence[object]]) -> None:
        self.cursor_instance = FakeCursor(rows)
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> object:
        return None

    def rollback(self) -> object:
        return None

    def close(self) -> object:
        self.closed = True
        return None


def test_snowflake_cortex_executes_provider_neutral_structured_completion() -> None:
    """Cortex should receive bound prompts, model parameters, and a strict schema."""

    connection = FakeConnection(
        [[{"structured_output": {"records": ["Alice"]}, "usage": {"total_tokens": 12}}]]
    )
    provider = _provider(connection)

    result = provider.complete_structured(
        StructuredCompletionRequest(
            task_name="entity_extraction",
            model="llama3.3-70b",
            system="Extract grounded entities.",
            user="Document text",
            json_schema={
                "type": "object",
                "properties": {"records": {"type": "array", "items": {"type": "string"}}},
                "required": ["records"],
            },
            prompt_metadata={"prompt_revision": "entity@1"},
        )
    )

    sql, params = connection.cursor_instance.executed[0]
    assert sql == (
        "SELECT AI_COMPLETE(model => ?, prompt => ?, "
        "model_parameters => PARSE_JSON(?)::OBJECT, "
        "response_format => PARSE_JSON(?)::OBJECT, "
        "show_details => TRUE, return_error_details => TRUE)"
    )
    assert params is not None
    assert params[0] == "llama3.3-70b"
    assert params[1] == "Extract grounded entities.\n\nDocument text"
    assert json.loads(str(params[2])) == {"max_tokens": 4096, "temperature": 0.0}
    assert json.loads(str(params[3]))["schema"]["required"] == ["records"]
    assert result.payload == {"records": ["Alice"]}
    assert result.provider_metadata["usage"] == {"total_tokens": 12}
    assert result.provider_metadata["prompt_revision"] == "entity@1"
    assert connection.cursor_instance.closed
    assert not connection.closed
    provider.close()
    assert connection.closed


def test_snowflake_cortex_applies_the_requested_completion_timeout() -> None:
    """The port contract's per-request timeout must reach the statement.

    A stalled AI_COMPLETE otherwise holds an extraction thread for the whole
    retry budget instead of failing the one attempt.
    """

    connection = FakeConnection([[{"structured_output": {"records": []}}]])
    provider = _provider(connection)

    provider.complete_structured(
        StructuredCompletionRequest(
            task_name="entity_extraction",
            model="llama3.3-70b",
            system="Extract grounded entities.",
            user="Document text",
            json_schema={"type": "object", "properties": {"records": {"type": "array"}}},
            timeout_seconds=42,
        )
    )

    assert connection.cursor_instance.timeouts == [42]


def test_snowflake_cortex_bounds_enrichment_calls_without_a_request_timeout() -> None:
    """Enrichment carries no per-request timeout, so the adapter's own applies."""

    connection = FakeConnection([[{"structured_output": {"description": "Grounded."}}]])

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexLlmProvider(
        _config(),
        "llama3.3-70b",
        connector_factory=factory,
        timeout_seconds=77,
    )

    provider.merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice",
            entity_type="PERSON",
            descriptions=["Alice is present."],
            evidence=[],
        )
    )

    assert connection.cursor_instance.timeouts == [77]


def test_snowflake_cortex_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        SnowflakeCortexLlmProvider(_config(), "llama3.3-70b", timeout_seconds=0)


def test_snowflake_cortex_summarizes_community() -> None:
    connection = FakeConnection(
        [
            [
                {
                    "structured_output": {
                        "title": "Acme",
                        "summary": "Acme community",
                        "rating": 8,
                        "rating_explanation": "Important because Alice anchors Acme.",
                        "findings": [{"summary": "Finding", "explanation": "Because"}],
                        "suggested_questions": ["Who anchors Acme?"],
                    },
                    "model": "llama3.3-70b",
                }
            ]
        ]
    )

    summary = _provider(connection).summarize_community(
        CommunitySummaryRequest(title_seed="Acme", members=["Alice", "Acme"], relations=[])
    )

    _sql, params = connection.cursor_instance.executed[0]
    assert params is not None
    assert json.loads(str(params[3]))["schema"]["required"] == [
        "title",
        "summary",
        "rating",
        "rating_explanation",
        "findings",
        "suggested_questions",
    ]
    assert summary.title == "Acme"
    assert summary.rating == 8
    assert summary.findings == [("Finding", "Because")]
    assert summary.suggested_questions == ["Who anchors Acme?"]
    assert summary.provider_metadata["prompt_name"] == "community_report"


def test_snowflake_cortex_clamps_out_of_range_community_rating() -> None:
    """Do not let one provider rating violate the Community domain contract."""

    connection = FakeConnection(
        [
            [
                {
                    "structured_output": {
                        "title": "Acme",
                        "summary": "Acme community",
                        "rating": 99,
                        "rating_explanation": "Provider exceeded the scale.",
                        "findings": [],
                        "suggested_questions": [],
                    }
                }
            ]
        ]
    )

    summary = _provider(connection).summarize_community(
        CommunitySummaryRequest(title_seed="Acme", members=["Acme"], relations=[])
    )

    assert summary.rating == 10.0


def test_snowflake_cortex_merges_entity_description() -> None:
    connection = FakeConnection(
        [
            [
                {
                    "structured_output": {"description": "Alice Smith works at Acme Corp."},
                    "model": "llama3.3-70b",
                }
            ]
        ]
    )

    result = _provider(connection).merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice Smith",
            entity_type="PERSON",
            descriptions=["Alice is present.", "Alice works at Acme Corp."],
            evidence=["Alice Smith works at Acme Corp."],
        )
    )

    _sql, params = connection.cursor_instance.executed[0]
    assert params is not None
    assert "merge observed descriptions" in str(params[1])
    assert json.loads(str(params[3]))["schema"]["required"] == ["description"]
    assert result.description == "Alice Smith works at Acme Corp."
    assert result.provider_metadata["prompt_name"] == "entity_description_merge"


def test_snowflake_cortex_drops_null_enrichment_strings() -> None:
    description_connection = FakeConnection([[{"structured_output": {"description": None}}]])
    description = _provider(description_connection).merge_entity_description(
        DescriptionMergeRequest(
            entity_name="Alice",
            entity_type="PERSON",
            descriptions=["Alice is present."],
            evidence=[],
        )
    )
    summary_connection = FakeConnection(
        [
            [
                {
                    "structured_output": {
                        "title": None,
                        "summary": None,
                        "rating": None,
                        "rating_explanation": None,
                        "findings": [{"summary": None, "explanation": None}],
                        "suggested_questions": [None, "Valid question?"],
                    }
                }
            ]
        ]
    )
    summary = _provider(summary_connection).summarize_community(
        CommunitySummaryRequest(title_seed="Fallback", members=[], relations=[])
    )

    assert description.description == ""
    assert summary.title == "Fallback"
    assert summary.summary == ""
    assert summary.rating_explanation == ""
    assert summary.findings == [("", "")]
    assert summary.suggested_questions == ["Valid question?"]


def test_snowflake_cortex_normalizes_choices_message_string_payload() -> None:
    payload = _structured_payload(
        {
            "choices": [
                {
                    "messages": json.dumps(
                        {
                            "title": "Acme",
                            "summary": "Acme community",
                            "rating": 7,
                            "rating_explanation": "Useful community.",
                            "findings": [],
                            "suggested_questions": ["What is Acme?"],
                        }
                    )
                }
            ]
        }
    )

    assert payload["title"] == "Acme"
    assert payload["suggested_questions"] == ["What is Acme?"]


def test_snowflake_cortex_normalizes_json_string_result() -> None:
    result = _snowflake_result_to_object(
        json.dumps({"structured_output": {"description": "Alice works at Acme."}})
    )

    assert result == {"structured_output": {"description": "Alice works at Acme."}}


def test_snowflake_cortex_rejects_non_object_result() -> None:
    with pytest.raises(ValueError, match="unsupported result type"):
        _snowflake_result_to_object(42)


def test_snowflake_cortex_explains_null_result() -> None:
    """A bare NULL should identify missing error details instead of a type mismatch."""

    with pytest.raises(RuntimeError, match="returned NULL without error details"):
        _snowflake_result_to_object(None)


def test_snowflake_cortex_unwraps_successful_error_details_envelope() -> None:
    """Successful detailed calls should expose the ordinary completion response."""

    details = _unwrap_completion_result(
        {
            "value": {"structured_output": {"description": "Grounded description."}},
            "error": None,
        }
    )

    assert details == {"structured_output": {"description": "Grounded description."}}


def test_snowflake_cortex_surfaces_provider_error_details() -> None:
    """Inference failures should retain Snowflake's actionable provider message."""

    with pytest.raises(RuntimeError, match="input schema validation error"):
        _unwrap_completion_result(
            {"value": None, "error": "input schema validation error: unsupported maxLength"}
        )


def test_snowflake_cortex_marks_model_output_validation_as_retryable() -> None:
    """Malformed model JSON should enter the shared structured repair loop."""

    with pytest.raises(ValueError, match="json mode output validation error"):
        _unwrap_completion_result(
            {
                "value": None,
                "error": "json mode output validation error: invalid JSON",
            }
        )


def test_snowflake_cortex_removes_only_unsupported_schema_constraints() -> None:
    """Project full contracts onto Cortex's documented JSON Schema subset."""

    schema = _snowflake_json_schema(
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "maxLength": 160},
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "items": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {"type": "string"},
                },
            },
            "required": ["name", "score", "items"],
        }
    )

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["name", "score", "items"]
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["score"] == {"type": "number"}
    assert schema["properties"]["items"] == {
        "type": "array",
        "items": {"type": "string"},
    }


def _provider(connection: FakeConnection) -> SnowflakeCortexLlmProvider:
    """Build an adapter bound to a deterministic fake connection."""

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    return SnowflakeCortexLlmProvider(_config(), "llama3.3-70b", connector_factory=factory)


def _config() -> SnowflakeConnectionConfig:
    """Return the complete connection settings required by the adapter."""

    return SnowflakeConnectionConfig(
        account="account",
        host=None,
        user="user",
        password="password",
        authenticator=None,
        private_key_path=None,
        database="DB",
        schema_name="SCHEMA",
        role="ROLE",
        warehouse="WH",
    )
