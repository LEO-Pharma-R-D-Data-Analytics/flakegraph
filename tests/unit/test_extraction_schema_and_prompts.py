from __future__ import annotations

import pytest

from kg_processor.application.extraction_schema import (
    EXTRACTION_SCHEMA_REVISION,
    extraction_response_format,
    validate_extraction_result,
)
from kg_processor.application.graph_extraction import (
    ExtractionBatchCheckpoint,
    extract_graph_from_chunks,
)
from kg_processor.application.prompt_registry import (
    EXTRACTION_PROMPT_NAMES,
    PROMPT_REVISIONS,
    community_report_prompt,
    entity_description_merge_prompt,
    get_prompt_template,
    graph_extraction_prompt,
    graph_repair_prompt,
    prompt_fingerprints,
)
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation, ExtractionResult
from kg_processor.domain.ids import stable_id
from kg_processor.ports.llm import (
    CommunitySummaryRequest,
    CommunitySummaryResult,
    DescriptionMergeRequest,
    DescriptionMergeResult,
    GraphRepairRequest,
    LlmOptions,
)


def test_extraction_response_format_is_generated_from_domain_schema() -> None:
    response_format = extraction_response_format()

    assert response_format["type"] == "json"
    schema = response_format["schema"]
    assert schema["properties"]["entities"]["items"]["$ref"] == "#/$defs/ExtractedEntity"
    assert schema["properties"]["relations"]["items"]["$ref"] == "#/$defs/ExtractedRelation"
    entity_schema = schema["$defs"]["ExtractedEntity"]["properties"]
    assert "quote" in entity_schema
    assert "start_offset" in entity_schema
    assert "end_offset" in entity_schema


def test_graph_extraction_prompt_records_template_metadata() -> None:
    chunk = _chunk()
    prompt = graph_extraction_prompt(
        [chunk],
        LlmOptions(model="graph", entity_types=["PERSON"]),
    )
    template = get_prompt_template("graph_extraction")

    assert prompt.template.revision == PROMPT_REVISIONS["graph_extraction"]
    assert prompt.template.checksum == template.checksum
    assert "Valid source_chunk_id values are: chunk_1" in prompt.user
    assert "--- CHUNK id=chunk_1 ---" in prompt.user
    assert "Return no prose outside the JSON object" in prompt.system


def test_prompt_registry_has_explicit_revisions_and_fingerprints() -> None:
    fingerprints = prompt_fingerprints(EXTRACTION_PROMPT_NAMES)

    assert set(PROMPT_REVISIONS) == {
        "graph_extraction",
        "graph_gleaning",
        "graph_repair",
        "entity_description_merge",
        "community_report",
    }
    assert all(not revision.endswith("current") for revision in PROMPT_REVISIONS.values())
    assert set(fingerprints) == set(EXTRACTION_PROMPT_NAMES)
    assert all(fingerprint["revision"] for fingerprint in fingerprints.values())
    assert all(len(fingerprint["checksum"]) == 64 for fingerprint in fingerprints.values())


def test_prompt_registry_rejects_unknown_templates() -> None:
    with pytest.raises(ValueError, match="Unknown prompt template"):
        get_prompt_template("missing_prompt")


def test_graph_extraction_prompt_includes_configured_caps_and_thresholds() -> None:
    prompt = graph_extraction_prompt(
        [_chunk()],
        LlmOptions(
            model="graph",
            entity_types=["PERSON"],
            max_entities_per_batch=5,
            max_relations_per_batch=7,
            min_entity_confidence=0.75,
            min_relation_confidence=0.65,
        ),
    )

    assert "Return at most 5 entities and 7 relations" in prompt.user
    assert "below confidence 0.75" in prompt.user
    assert "below confidence 0.65" in prompt.user


def test_graph_extraction_prompt_requests_quote_and_span_grounding() -> None:
    prompt = graph_extraction_prompt(
        [_chunk()],
        LlmOptions(model="graph", entity_types=["PERSON"]),
    )

    assert "short exact `quote`" in prompt.system
    assert "chunk-local `start_offset` and `end_offset`" in prompt.system


def test_validate_extraction_result_accepts_grounded_quote_and_span() -> None:
    result = validate_extraction_result(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Alice Smith",
                    type="PERSON",
                    description="Alice is present.",
                    source_chunk_id="chunk_1",
                    quote="Alice Smith",
                    start_offset=0,
                    end_offset=11,
                )
            ],
            relations=[
                ExtractedRelation(
                    source_name="Alice Smith",
                    target_name="Acme Corp",
                    relation_type="works_at",
                    description="Alice works at Acme.",
                    source_chunk_id="chunk_1",
                    quote="works at",
                    start_offset=12,
                    end_offset=20,
                )
            ],
        ),
        [_chunk()],
        LlmOptions(model="graph", entity_types=["PERSON"], relation_types=["works_at"]),
        pass_index=0,
    )

    assert result.entities[0].quote == "Alice Smith"
    assert result.relations[0].start_offset == 12


def test_validate_extraction_result_normalizes_grounded_quote_with_bad_span() -> None:
    result = validate_extraction_result(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Alice Smith",
                    type="PERSON",
                    description="Alice is present.",
                    source_chunk_id="chunk_1",
                    quote="Alice Smith",
                    start_offset=12,
                    end_offset=20,
                )
            ],
            relations=[
                ExtractedRelation(
                    source_name="Alice Smith",
                    target_name="Acme Corp",
                    relation_type="works_at",
                    description="Alice works at Acme.",
                    source_chunk_id="chunk_1",
                    quote="Acme Corp",
                    start_offset=0,
                    end_offset=5,
                )
            ],
        ),
        [_chunk()],
        LlmOptions(model="graph", entity_types=["PERSON"], relation_types=["works_at"]),
        pass_index=0,
    )

    assert result.entities[0].start_offset == 0
    assert result.entities[0].end_offset == 11
    assert result.relations[0].start_offset == 21
    assert result.relations[0].end_offset == 30
    assert result.provider_metadata["evidence_offsets_normalized"] == 2


def test_validate_extraction_result_rejects_unsupported_quote() -> None:
    with pytest.raises(ValueError, match="quote is not present in source chunk"):
        validate_extraction_result(
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Alice is present.",
                        source_chunk_id="chunk_1",
                        quote="Unsupported quote",
                        start_offset=12,
                        end_offset=20,
                    )
                ]
            ),
            [_chunk()],
            LlmOptions(model="graph", entity_types=["PERSON"]),
            pass_index=0,
        )


def test_graph_repair_prompt_records_invalid_response_and_allowed_chunks() -> None:
    prompt = graph_repair_prompt(
        [_chunk()],
        LlmOptions(model="graph", entity_types=["PERSON"]),
        invalid_response="not-json",
        validation_error="Expecting value",
    )
    template = get_prompt_template("graph_repair")

    assert prompt.template.revision == PROMPT_REVISIONS["graph_repair"]
    assert prompt.template.checksum == template.checksum
    assert "Return one JSON object" in prompt.system
    assert '"invalid_response": "not-json"' in prompt.user
    assert '"valid_source_chunk_ids": ["chunk_1"]' in prompt.user
    assert "Expecting value" in prompt.user


def test_community_report_prompt_records_template_metadata() -> None:
    request = CommunitySummaryRequest(
        title_seed="Acme",
        members=["Alice Smith [PERSON] degree=2"],
        relations=["Alice Smith --works_at--> Acme Corp weight=9.0"],
    )
    prompt = community_report_prompt(request)
    template = get_prompt_template("community_report")

    assert prompt.template.revision == PROMPT_REVISIONS["community_report"]
    assert prompt.template.checksum == template.checksum
    assert "Use only supplied member and relation text" in prompt.system
    assert "rating_explanation" in prompt.system
    assert "suggested_questions" in prompt.system
    assert '"title_seed": "Acme"' in prompt.user
    assert "works_at" in prompt.user


def test_entity_description_merge_prompt_records_template_metadata() -> None:
    request = DescriptionMergeRequest(
        entity_name="Alice Smith",
        entity_type="PERSON",
        descriptions=["Alice is a researcher.", "Alice works at Acme Corp."],
        evidence=["Alice Smith works at Acme Corp."],
    )
    prompt = entity_description_merge_prompt(request)
    template = get_prompt_template("entity_description_merge")

    assert prompt.template.revision == PROMPT_REVISIONS["entity_description_merge"]
    assert prompt.template.checksum == template.checksum
    assert "Use only the supplied descriptions and evidence snippets" in prompt.system
    assert '"entity_name": "Alice Smith"' in prompt.user
    assert "Acme Corp" in prompt.user


def test_extract_graph_from_chunks_rejects_invalid_chunk_citations() -> None:
    provider = _StaticProvider(
        [
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Alice is present.",
                        source_chunk_id="missing_chunk",
                    )
                ]
            )
        ]
    )

    with pytest.raises(ValueError, match="invalid source_chunk_id"):
        extract_graph_from_chunks(
            [_chunk()],
            provider,
            GraphSettings(entity_types=["PERSON"], gleaning_max_passes=0),
            "graph",
            120,
        )


def test_extract_graph_from_chunks_repairs_invalid_chunk_citations() -> None:
    provider = _StaticProvider(
        [
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Alice is present.",
                        source_chunk_id="missing_chunk",
                    )
                ],
                provider_metadata={"provider": "test"},
            )
        ],
        repair_results=[
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Alice is present.",
                        source_chunk_id="chunk_1",
                    )
                ],
                provider_metadata={"provider": "test_repair"},
            )
        ],
    )

    result = extract_graph_from_chunks(
        [_chunk()],
        provider,
        GraphSettings(entity_types=["PERSON"], gleaning_max_passes=0),
        "graph",
        120,
    )

    assert result.entities[0].source_chunk_id == "chunk_1"
    assert len(provider.repair_requests) == 1
    repair_request = provider.repair_requests[0]
    assert repair_request.invalid_result.entities[0].source_chunk_id == "missing_chunk"
    assert "invalid source_chunk_id" in repair_request.validation_error
    pass_metadata = result.provider_metadata["batches"][0]["passes"][0]
    assert pass_metadata["validation_repair_attempts"] == 1
    assert pass_metadata["pass_index"] == 0
    assert pass_metadata["provider"] == "test_repair"
    assert pass_metadata["pre_repair_provider_metadata"] == {"provider": "test"}


def test_extract_graph_from_chunks_drops_ungrounded_relations_after_failed_repair() -> None:
    provider = _StaticProvider(
        [
            ExtractionResult(
                relations=[
                    ExtractedRelation(
                        source_name="Alice Smith",
                        target_name="Acme Corp",
                        relation_type="works_at",
                        description="Alice works at Acme.",
                        source_chunk_id="chunk_1",
                        quote="not present in the chunk",
                    )
                ],
                provider_metadata={"provider": "test"},
            )
        ],
        repair_results=[
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Alice is present.",
                        source_chunk_id="chunk_1",
                        quote="Alice Smith",
                    )
                ],
                relations=[
                    ExtractedRelation(
                        source_name="Alice Smith",
                        target_name="Acme Corp",
                        relation_type="works_at",
                        description="Alice works at Acme.",
                        source_chunk_id="chunk_1",
                        quote="still not present in the chunk",
                    )
                ],
                provider_metadata={"provider": "test_repair"},
            )
        ],
    )

    result = extract_graph_from_chunks(
        [_chunk()],
        provider,
        GraphSettings(entity_types=["PERSON"], gleaning_max_passes=0),
        "graph",
        120,
    )

    assert [entity.name for entity in result.entities] == ["Alice Smith"]
    assert result.relations == []
    assert len(provider.repair_requests) == 1
    pass_metadata = result.provider_metadata["batches"][0]["passes"][0]
    assert pass_metadata["validation_repair_attempts"] == 1
    assert "quote is not present" in pass_metadata["validation_repair_error"]
    assert "quote is not present" in pass_metadata["validation_repair_failed_error"]
    assert pass_metadata["dropped_entities"] == 0
    assert pass_metadata["dropped_relations"] == 1
    assert pass_metadata["dropped_observations_by_reason"] == {"ungrounded_quote": 1}


def test_extract_graph_from_chunks_rejects_over_limit_batches() -> None:
    provider = _StaticProvider(
        [
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Alice is present.",
                        source_chunk_id="chunk_1",
                    ),
                    ExtractedEntity(
                        name="Bob Jones",
                        type="PERSON",
                        description="Bob is present.",
                        source_chunk_id="chunk_1",
                    ),
                ],
                relations=[
                    ExtractedRelation(
                        source_name="Alice Smith",
                        target_name="Bob Jones",
                        relation_type="knows",
                        description="Alice knows Bob.",
                        source_chunk_id="chunk_1",
                    )
                ],
            )
        ]
    )

    with pytest.raises(ValueError, match="too many entities"):
        extract_graph_from_chunks(
            [_chunk()],
            provider,
            GraphSettings(
                entity_types=["PERSON"],
                max_entities_per_batch=1,
                gleaning_max_passes=0,
            ),
            "graph",
            120,
        )


def test_extract_graph_from_chunks_dedupes_gleaned_records() -> None:
    provider = _StaticProvider(
        [
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Alice is present.",
                        source_chunk_id="chunk_1",
                    )
                ]
            ),
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Duplicate Alice.",
                        source_chunk_id="chunk_1",
                    ),
                    ExtractedEntity(
                        name="Acme Corp",
                        type="ORGANIZATION",
                        description="Acme is present.",
                        source_chunk_id="chunk_1",
                    ),
                ]
            ),
        ]
    )

    result = extract_graph_from_chunks(
        [_chunk()],
        provider,
        GraphSettings(
            entity_types=["PERSON", "ORGANIZATION"],
            gleaning_max_passes=1,
            gleaning_saturation_threshold=1,
        ),
        "graph",
        120,
    )

    assert [entity.name for entity in result.entities] == ["Alice Smith", "Acme Corp"]
    assert provider.passes == [0, 1]
    assert result.provider_metadata["schema_revision"] == EXTRACTION_SCHEMA_REVISION
    assert result.provider_metadata["chunk_count"] == 1
    batch_metadata = result.provider_metadata["batches"][0]
    assert batch_metadata["batch_id"] == stable_id("extraction_batch", "chunk_1")
    assert batch_metadata["batch_index"] == 0
    assert [pass_["pass_index"] for pass_ in batch_metadata["passes"]] == [0, 1]


def test_extract_graph_from_chunks_runs_gleaning_after_low_recall_initial_pass() -> None:
    provider = _StaticProvider(
        [
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Alice is present.",
                        source_chunk_id="chunk_1",
                    )
                ]
            ),
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Acme Corp",
                        type="ORGANIZATION",
                        description="Acme is present.",
                        source_chunk_id="chunk_1",
                    )
                ],
                relations=[
                    ExtractedRelation(
                        source_name="Alice Smith",
                        target_name="Acme Corp",
                        relation_type="works_at",
                        description="Alice works at Acme.",
                        source_chunk_id="chunk_1",
                    )
                ],
            ),
        ]
    )

    result = extract_graph_from_chunks(
        [_chunk()],
        provider,
        GraphSettings(
            entity_types=["PERSON", "ORGANIZATION"],
            gleaning_max_passes=2,
            gleaning_saturation_threshold=10,
        ),
        "graph",
        120,
    )

    assert [entity.name for entity in result.entities] == ["Alice Smith", "Acme Corp"]
    assert [relation.relation_type for relation in result.relations] == ["works_at"]
    assert provider.passes == [0, 1]
    assert provider.previous_results[0] is None
    assert provider.previous_results[1] is not None
    assert [entity.name for entity in provider.previous_results[1].entities] == ["Alice Smith"]


def test_extract_graph_from_chunks_records_stable_batch_metadata() -> None:
    provider = _StaticProvider(
        [
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Alice Smith",
                        type="PERSON",
                        description="Alice is present.",
                        source_chunk_id="chunk_1",
                    )
                ]
            ),
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Bob Jones",
                        type="PERSON",
                        description="Bob is present.",
                        source_chunk_id="chunk_2",
                    )
                ]
            ),
        ]
    )

    result = extract_graph_from_chunks(
        [_chunk(), _chunk("chunk_2", "file_2", "Bob Jones works at Beta Corp.")],
        provider,
        GraphSettings(
            entity_types=["PERSON"],
            max_chunks_per_llm_call=1,
            gleaning_max_passes=0,
        ),
        "graph",
        120,
    )

    batches = result.provider_metadata["batches"]
    assert [batch["batch_index"] for batch in batches] == [0, 1]
    assert [batch["batch_id"] for batch in batches] == [
        stable_id("extraction_batch", "chunk_1"),
        stable_id("extraction_batch", "chunk_2"),
    ]
    assert [batch["chunk_ids"] for batch in batches] == [["chunk_1"], ["chunk_2"]]
    assert [batch["passes"][0]["pass_index"] for batch in batches] == [0, 0]


def test_extract_graph_from_chunks_resumes_from_batch_checkpoint_after_failure() -> None:
    checkpoints: dict[str, ExtractionResult] = {}

    def get_checkpoint(batch: list[Chunk]) -> ExtractionBatchCheckpoint:
        cache_id = _checkpoint_id(batch)
        return ExtractionBatchCheckpoint(cache_id=cache_id, result=checkpoints.get(cache_id))

    def put_checkpoint(batch: list[Chunk], result: ExtractionResult) -> str:
        cache_id = _checkpoint_id(batch)
        checkpoints[cache_id] = result
        return cache_id

    first_provider = _FailingSecondBatchProvider()

    with pytest.raises(RuntimeError, match="batch 2 failed"):
        extract_graph_from_chunks(
            [_chunk(), _chunk("chunk_2", "file_2", "Bob Jones works at Beta Corp.")],
            first_provider,
            GraphSettings(
                entity_types=["PERSON"],
                max_chunks_per_llm_call=1,
                gleaning_max_passes=0,
            ),
            "graph",
            120,
            batch_checkpoint_get=get_checkpoint,
            batch_checkpoint_put=put_checkpoint,
        )

    assert sorted(checkpoints) == [_checkpoint_id([_chunk()])]
    second_provider = _StaticProvider(
        [
            ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Bob Jones",
                        type="PERSON",
                        description="Bob is present.",
                        source_chunk_id="chunk_2",
                    )
                ]
            )
        ]
    )

    result = extract_graph_from_chunks(
        [_chunk(), _chunk("chunk_2", "file_2", "Bob Jones works at Beta Corp.")],
        second_provider,
        GraphSettings(
            entity_types=["PERSON"],
            max_chunks_per_llm_call=1,
            gleaning_max_passes=0,
        ),
        "graph",
        120,
        batch_checkpoint_get=get_checkpoint,
        batch_checkpoint_put=put_checkpoint,
    )

    assert [entity.name for entity in result.entities] == ["Alice Smith", "Bob Jones"]
    assert second_provider.passes == [0]
    batches = result.provider_metadata["batches"]
    assert batches[0]["checkpoint_cache_hit"] is True
    assert batches[1]["checkpoint_cache_hit"] is False


class _StaticProvider:
    def __init__(
        self,
        results: list[ExtractionResult],
        repair_results: list[ExtractionResult] | None = None,
    ) -> None:
        self.results = results
        self.repair_results = repair_results or []
        self.passes: list[int] = []
        self.previous_results: list[ExtractionResult | None] = []
        self.repair_requests: list[GraphRepairRequest] = []

    def extract_graph(self, chunks: list[Chunk], options: LlmOptions) -> ExtractionResult:
        self.passes.append(options.extraction_pass)
        self.previous_results.append(options.previous_result)
        return self.results.pop(0)

    def repair_graph_extraction(self, request: GraphRepairRequest) -> ExtractionResult:
        self.repair_requests.append(request)
        if self.repair_results:
            return self.repair_results.pop(0)
        return request.invalid_result

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        return DescriptionMergeResult(description=request.descriptions[0])

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        return CommunitySummaryResult(
            title=request.title_seed,
            summary="",
            rating=0,
            findings=[],
        )


class _FailingSecondBatchProvider:
    def __init__(self) -> None:
        self.calls = 0

    def extract_graph(self, chunks: list[Chunk], options: LlmOptions) -> ExtractionResult:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("batch 2 failed")
        return ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Alice Smith",
                    type="PERSON",
                    description="Alice is present.",
                    source_chunk_id=chunks[0].id,
                )
            ]
        )

    def repair_graph_extraction(self, request: GraphRepairRequest) -> ExtractionResult:
        return request.invalid_result

    def merge_entity_description(
        self,
        request: DescriptionMergeRequest,
    ) -> DescriptionMergeResult:
        return DescriptionMergeResult(description=request.descriptions[0])

    def summarize_community(self, request: CommunitySummaryRequest) -> CommunitySummaryResult:
        return CommunitySummaryResult(
            title=request.title_seed,
            summary="",
            rating=0,
            findings=[],
        )


def _chunk(
    chunk_id: str = "chunk_1",
    file_id: str = "file_1",
    content: str = "Alice Smith works at Acme Corp.",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        file_id=file_id,
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=6,
        content_hash=stable_id("content", content),
    )


def _checkpoint_id(batch: list[Chunk]) -> str:
    return stable_id("checkpoint", *[chunk.id for chunk in batch])
