from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from kg_processor.adapters.llm.snowflake_cortex import (
    SnowflakeCortexLlmProvider,
    _snowflake_result_to_object,
    _structured_payload,
)
from kg_processor.adapters.snowflake import SnowflakeConnectionConfig
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractionResult
from kg_processor.ports.llm import CommunitySummaryRequest, DescriptionMergeRequest, LlmOptions


class FakeCursor:
    def __init__(self, rows: list[Sequence[object]]) -> None:
        self.rows = rows
        self.index = 0
        self.executed: list[tuple[str, Sequence[object] | None]] = []
        self.closed = False

    def execute(self, sql: str, params: Sequence[object] | None = None) -> object:
        self.executed.append((sql, params))
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


def test_snowflake_cortex_llm_extracts_structured_output() -> None:
    connection = FakeConnection(
        [
            [
                {
                    "structured_output": [
                        {
                            "raw_message": {
                                "entities": [
                                    {
                                        "name": "Alice Smith",
                                        "type": "PERSON",
                                        "description": "Alice is present.",
                                        "source_chunk_id": "chunk_1",
                                        "confidence": 1,
                                        "aliases": [],
                                    }
                                ],
                                "relations": [],
                            },
                            "type": "json",
                        }
                    ],
                    "model": "llama3.3-70b",
                    "usage": {"total_tokens": 100},
                }
            ]
        ]
    )

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexLlmProvider(_config(), "llama3.3-70b", connector_factory=factory)

    result = provider.extract_graph(
        [_chunk()],
        LlmOptions(
            model="llama3.3-70b",
            entity_types=["PERSON", "ORGANIZATION"],
            extraction_pass=1,
            previous_result=ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Acme Corp",
                        type="ORGANIZATION",
                        description="Acme is present.",
                        source_chunk_id="chunk_1",
                    )
                ]
            ),
        ),
    )

    sql, params = connection.cursor_instance.executed[0]
    assert sql == "SELECT AI_COMPLETE(?, ?, PARSE_JSON(?)::OBJECT, PARSE_JSON(?)::OBJECT, TRUE)"
    assert params is not None
    assert params[0] == "llama3.3-70b"
    assert "Valid source_chunk_id values are: chunk_1" in str(params[1])
    assert "Previously accepted observations" in str(params[1])
    assert "Acme Corp" in str(params[1])
    model_parameters = json.loads(str(params[2]))
    assert model_parameters["temperature"] == 0.0
    response_format = json.loads(str(params[3]))
    assert response_format["type"] == "json"
    assert result.entities[0].name == "Alice Smith"
    assert result.provider_metadata["usage"] == {"total_tokens": 100}


def test_snowflake_cortex_llm_repairs_invalid_structured_output() -> None:
    connection = FakeConnection(
        [
            [
                {
                    "structured_output": {
                        "entities": "not-a-list",
                        "relations": [],
                    },
                    "model": "llama3.3-70b",
                }
            ],
            [
                {
                    "structured_output": {
                        "entities": [
                            {
                                "name": "Alice Smith",
                                "type": "PERSON",
                                "description": "Alice is present.",
                                "source_chunk_id": "chunk_1",
                                "confidence": 1,
                                "aliases": [],
                            }
                        ],
                        "relations": [],
                    },
                    "model": "llama3.3-70b",
                }
            ],
        ]
    )

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexLlmProvider(_config(), "llama3.3-70b", connector_factory=factory)

    result = provider.extract_graph(
        [_chunk()],
        LlmOptions(model="llama3.3-70b", entity_types=["PERSON"]),
    )

    assert len(connection.cursor_instance.executed) == 2
    _sql, repair_params = connection.cursor_instance.executed[1]
    assert repair_params is not None
    assert "repair malformed knowledge graph extraction output" in str(repair_params[1])
    assert "not-a-list" in str(repair_params[1])
    assert result.entities[0].source_chunk_id == "chunk_1"
    assert result.provider_metadata["repair_attempts"] == 1
    assert result.provider_metadata["repair_prompt_name"] == "graph_repair"


def test_snowflake_cortex_llm_summarizes_community() -> None:
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

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexLlmProvider(_config(), "llama3.3-70b", connector_factory=factory)

    summary = provider.summarize_community(
        CommunitySummaryRequest(title_seed="Acme", members=["Alice", "Acme"], relations=[])
    )
    _sql, params = connection.cursor_instance.executed[0]
    assert params is not None
    response_format = json.loads(str(params[3]))
    assert response_format["schema"]["required"] == [
        "title",
        "summary",
        "rating",
        "rating_explanation",
        "findings",
        "suggested_questions",
    ]

    assert summary.title == "Acme"
    assert summary.rating == 8
    assert summary.rating_explanation == "Important because Alice anchors Acme."
    assert summary.findings == [("Finding", "Because")]
    assert summary.suggested_questions == ["Who anchors Acme?"]
    assert summary.provider_metadata["provider"] == "snowflake_cortex"
    assert summary.provider_metadata["model"] == "llama3.3-70b"
    assert summary.provider_metadata["prompt_name"] == "community_report"


def test_snowflake_cortex_llm_merges_entity_description() -> None:
    connection = FakeConnection(
        [
            [
                {
                    "structured_output": {
                        "description": "Alice Smith works at Acme Corp.",
                    },
                    "model": "llama3.3-70b",
                }
            ]
        ]
    )

    def factory(**_kwargs: object) -> FakeConnection:
        return connection

    provider = SnowflakeCortexLlmProvider(_config(), "llama3.3-70b", connector_factory=factory)

    result = provider.merge_entity_description(
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
    response_format = json.loads(str(params[3]))
    assert response_format["schema"]["required"] == ["description"]
    assert result.description == "Alice Smith works at Acme Corp."
    assert result.provider_metadata["provider"] == "snowflake_cortex"
    assert result.provider_metadata["model"] == "llama3.3-70b"
    assert result.provider_metadata["prompt_name"] == "entity_description_merge"


def test_snowflake_cortex_llm_normalizes_choices_message_string_payload() -> None:
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

    assert payload == {
        "title": "Acme",
        "summary": "Acme community",
        "rating": 7,
        "rating_explanation": "Useful community.",
        "findings": [],
        "suggested_questions": ["What is Acme?"],
    }


def test_snowflake_cortex_llm_normalizes_json_string_result() -> None:
    result = _snowflake_result_to_object(
        json.dumps({"structured_output": {"description": "Alice works at Acme."}})
    )

    assert result == {"structured_output": {"description": "Alice works at Acme."}}


def test_snowflake_cortex_llm_rejects_non_object_result() -> None:
    with pytest.raises(ValueError, match="unsupported result type"):
        _snowflake_result_to_object(42)


def _chunk() -> Chunk:
    return Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith works at Acme.",
        start_offset=0,
        end_offset=26,
        token_count=5,
        content_hash="hash",
    )


def _config() -> SnowflakeConnectionConfig:
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
