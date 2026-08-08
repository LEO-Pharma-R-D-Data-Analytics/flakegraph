from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg_processor.adapters.embeddings.hash import HashEmbeddingProvider
from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.application.llm_extractors import LlmRelationExtractor
from kg_processor.application.ontology import load_ontology
from kg_processor.application.two_pass_extraction import (
    _relation_completion_inventory,
    _relation_inventory_for_window,
    _run_window_stage,
    extract_graph_observations,
    extract_graph_two_pass,
    is_reference_only_window,
)
from kg_processor.application.window_errors import is_systemic_provider_error
from kg_processor.config.settings import GraphSettings
from kg_processor.domain.extraction import EntityMention, ExtractionWindow
from kg_processor.domain.graph import Chunk
from kg_processor.ports.embeddings import EmbedOptions
from kg_processor.ports.llm import StructuredCompletionRequest, StructuredCompletionResult


class AuthenticationError(RuntimeError):
    """Model a provider-wide credential rejection."""


class _HttpError(RuntimeError):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = type("Response", (), {"status_code": status_code})()


@pytest.mark.parametrize("status_code", [400, 422])
def test_request_specific_http_errors_are_not_systemic(status_code: int) -> None:
    assert not is_systemic_provider_error(_HttpError(status_code))


@pytest.mark.parametrize("status_code", [401, 403])
def test_auth_http_errors_are_systemic(status_code: int) -> None:
    assert is_systemic_provider_error(_HttpError(status_code))


def test_window_stage_reraises_systemic_authentication_failures() -> None:
    def reject(_window: ExtractionWindow) -> object:
        raise AuthenticationError("invalid API key")

    with pytest.raises(AuthenticationError, match="invalid API key"):
        _run_window_stage(  # noqa: SLF001
            [_window("Alice works at Acme.")],
            GraphSettings(extraction_parallelism=1),
            reject,  # type: ignore[arg-type]
            None,
        )


class _MixedRecordLlm(FakeLlmProvider):
    """Return mixed valid and invalid sibling records across every extraction stage.

    This fixture exercises record-local failure isolation.
    """

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Construct mixed stage payloads while delegating unrelated tasks normally.

        Provider metadata identifies the controlled fixture path.
        """

        if request.task_name == "entity_extraction":
            payload = {
                "entities": [
                    {**_entity("Invalid confidence", "PERSON"), "confidence": -1.0},
                    _entity("Jigoro Kano", "PERSON"),
                    _entity("Kodokan", "SCHOOL"),
                    _entity("Tokyo", "LOCATION"),
                ]
            }
        elif request.task_name == "relation_extraction":
            input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
            ids = {item["name"]: item["id"] for item in input_payload["entities"]}
            quote = "Jigoro Kano founded the Kodokan in Tokyo."
            payload = {
                "relations": [
                    {
                        "source_entity_id": ids["Kodokan"],
                        "target_entity_id": ids["Jigoro Kano"],
                        "source_surface": "Kodokan",
                        "target_surface": "Jigoro Kano",
                        "relation_type": "FOUNDED_BY",
                        "description": "Schema-invalid confidence.",
                        "source_chunk_id": "chunk-1",
                        "quote": quote,
                        "confidence": -1.0,
                    },
                    {
                        # Deliberately reversed: the ontology signature should
                        # repair PERSON -> SCHOOL into SCHOOL -> PERSON.
                        "source_entity_id": ids["Jigoro Kano"],
                        "target_entity_id": ids["Kodokan"],
                        "source_surface": "Jigoro Kano",
                        "target_surface": "Kodokan",
                        "relation_type": "FOUNDED_BY",
                        "description": "Jigoro Kano founded the Kodokan.",
                        "source_chunk_id": "chunk-1",
                        "quote": quote,
                        "confidence": 0.98,
                    },
                    {
                        "source_entity_id": ids["Kodokan"],
                        "target_entity_id": ids["Tokyo"],
                        "source_surface": "Kodokan",
                        "target_surface": "Tokyo",
                        "relation_type": "LOCATED_IN",
                        "description": "The Kodokan was established in Tokyo.",
                        "source_chunk_id": "chunk-1",
                        "quote": quote,
                        "confidence": 0.98,
                    },
                    {
                        "source_entity_id": ids["Kodokan"],
                        "target_entity_id": ids["Kodokan"],
                        "source_surface": "Kodokan",
                        "target_surface": "Kodokan",
                        "relation_type": "FOUNDED_BY",
                        "description": "Invalid reflexive record.",
                        "source_chunk_id": "chunk-1",
                        "quote": quote,
                        "confidence": 0.9,
                    },
                    {
                        "source_entity_id": "missing-id",
                        "target_entity_id": ids["Jigoro Kano"],
                        "source_surface": "Kodokan",
                        "target_surface": "Jigoro Kano",
                        "relation_type": "FOUNDED_BY",
                        "description": "Invalid endpoint record.",
                        "source_chunk_id": "chunk-1",
                        "quote": quote,
                        "confidence": 0.9,
                    },
                ]
            }
        elif request.task_name == "relation_verification":
            input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
            payload = {
                "decisions": [
                    {
                        "relation_id": item["id"],
                        "verdict": "supported",
                        "confidence": 0.97,
                        "explanation": "The quoted sentence directly states the claim.",
                    }
                    for item in input_payload["relations"]
                ]
                + [
                    {
                        "relation_id": input_payload["relations"][0]["id"],
                        "verdict": "supported",
                        "confidence": -1.0,
                        "explanation": "Schema-invalid verifier sibling.",
                    }
                ]
            }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(payload=payload, provider_metadata={"provider": "mixed"})


class _CoverageAuditLlm(FakeLlmProvider):
    """Reveal one additional entity and relation only during continuation passes."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return partial first passes and complementary audit-pass records."""

        input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        if request.task_name == "entity_extraction":
            previous = input_payload["previous_entities"]
            names = (
                [("roda", "PRACTICE")]
                if previous
                else [
                    ("Capoeira", "MARTIAL_ART"),
                    ("UNESCO", "ORGANIZATION"),
                ]
            )
            payload = {"entities": [_entity(name, entity_type) for name, entity_type in names]}
        elif request.task_name == "relation_extraction":
            ids = {item["name"]: item["id"] for item in input_payload["entities"]}
            previous = input_payload["previous_relations"]
            if previous:
                payload = {
                    "relations": [
                        _relation(
                            ids["roda"],
                            ids["UNESCO"],
                            "roda",
                            "UNESCO",
                            "RECOGNIZED_BY",
                            "The roda was recognized by UNESCO.",
                        )
                    ]
                }
            else:
                payload = {
                    "relations": [
                        _relation(
                            ids["Capoeira"],
                            ids["roda"],
                            "Capoeira",
                            "roda",
                            "HAS_PRACTICE",
                            "Capoeira includes the roda.",
                        )
                    ]
                }
        elif request.task_name == "relation_verification":
            payload = {
                "decisions": [
                    {
                        "relation_id": item["id"],
                        "verdict": "supported",
                        "confidence": 1.0,
                        "explanation": "The exact source sentence states the triple.",
                    }
                    for item in input_payload["relations"]
                ]
            }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(payload=payload, provider_metadata={"provider": "audit"})


class _LocalSurfaceLlm(FakeLlmProvider):
    """Map a shortened source mention onto its canonical inventory entity ID."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return one relation whose target surface is shorter than its inventory name."""

        input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in input_payload["entities"]}
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids["Capoeira"],
                        ids["Capoeira roda"],
                        "Capoeira",
                        "roda",
                        "HAS_PRACTICE",
                        "Capoeira includes the roda.",
                    )
                ]
            },
            provider_metadata={"provider": "local-surface"},
        )


class _DocumentSubjectRelationLlm(FakeLlmProvider):
    """Connect an explicit document pronoun to its pre-extracted focal paper."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return one context-grounded contribution relation."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids["Adam: A Method for Stochastic Optimization"],
                        ids["Adam"],
                        "We",
                        "Adam",
                        "INTRODUCES",
                        "We introduce Adam, an algorithm for stochastic optimization.",
                    )
                    | {"source_chunk_id": "reference-chunk"}
                ]
            }
        )


class _ImplicitDocumentSubjectRelationLlm(FakeLlmProvider):
    """Attribute an authorial technical statement through document provenance."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return a paper-to-method relation whose quote omits the paper surface."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids["Learning representations by back-propagating errors"],
                        ids["Gradient descent"],
                        "source paper",
                        "gradient descent",
                        "USES_METHOD",
                        "To minimize E by gradient descent, it is necessary to "
                        "compute the derivative of E.",
                    )
                    | {"source_chunk_id": "reference-chunk"}
                ]
            }
        )


class _ImplicitMethodSubjectRelationLlm(FakeLlmProvider):
    """Try the provenance exception with a non-paper context entity."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return a method relation whose proposed source surface is absent."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids["Back-propagation"],
                        ids["Gradient descent"],
                        "the method",
                        "gradient descent",
                        "USES_METHOD",
                        "To minimize E by gradient descent, derivatives are computed efficiently.",
                    )
                    | {"source_chunk_id": "reference-chunk"}
                ]
            }
        )


class _NestedTitleRelationLlm(FakeLlmProvider):
    """Return a paper-to-task claim whose target is nested in the paper title."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Build the overlap case that must rely on document provenance."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        title = "Understanding the Difficulty of Training Deep Feedforward Neural Networks"
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids[title],
                        ids["Training Deep Feedforward Neural Networks"],
                        title,
                        "Training Deep Feedforward Neural Networks",
                        "ADDRESSES",
                        title,
                    )
                    | {"source_chunk_id": "reference-chunk"}
                ]
            }
        )


class _NormalizedEndpointSurfaceLlm(FakeLlmProvider):
    """Return a correct endpoint ID with a shortened surface absent from the OCR text."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Exercise canonical-inventory fallback while preserving an exact source quote."""

        payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        ids = {item["name"]: item["id"] for item in payload["entities"]}
        quote = (
            "The Stochastic Gradient Variational Bayes estimator uses the reparameterization trick."
        )
        return StructuredCompletionResult(
            payload={
                "relations": [
                    _relation(
                        ids["Stochastic Gradient Variational Bayes estimator"],
                        ids["reparameterization trick"],
                        "SGVB estimator",
                        "reparameterization trick",
                        "USES_METHOD",
                        quote,
                    )
                    | {"source_chunk_id": "reference-chunk"}
                ]
            }
        )


class _CueRecoveryLlm(FakeLlmProvider):
    """Omit a relation during generation so deterministic cue evidence recovers it."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return entities but deliberately omit the directly asserted relation."""

        input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        if request.task_name == "entity_extraction":
            payload = {
                "entities": [
                    _entity("UFC 1", "EVENT"),
                    _entity("Ultimate Fighting Championship", "ORGANIZATION"),
                ]
            }
        elif request.task_name == "relation_extraction":
            payload = {"relations": []}
        elif request.task_name == "relation_verification":
            names = {item["id"]: item["name"] for item in input_payload["entities"]}
            payload = {
                "decisions": [
                    {
                        "relation_id": item["id"],
                        "verdict": (
                            "supported"
                            if names[item["source_entity_id"]] == "UFC 1"
                            and names[item["target_entity_id"]] == "Ultimate Fighting Championship"
                            and item["relation_type"] == "PART_OF"
                            else "insufficient"
                        ),
                        "confidence": 1.0,
                        "explanation": "Only the correctly oriented part-of claim is stated.",
                    }
                    for item in input_payload["relations"]
                ]
            }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(payload=payload, provider_metadata={"provider": "cue"})


class _CompoundSubjectAuditLlm(FakeLlmProvider):
    """Miss a compound subject initially and recover it from the focused audit."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Return the full compound only when orchestration supplies its cue sentence."""

        input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        if request.task_name == "entity_extraction":
            entities = [
                _entity("Capoeira", "MARTIAL_ART"),
                _entity("UNESCO", "ORGANIZATION"),
            ]
            if str(input_payload["window_id"]).startswith("cue_entity_audit_window"):
                entities = [_entity("capoeira roda", "PRACTICE")]
            payload = {"entities": entities}
        elif request.task_name == "relation_extraction":
            payload = {"relations": []}
        elif request.task_name == "relation_verification":
            payload = {
                "decisions": [
                    {
                        "relation_id": item["id"],
                        "verdict": "insufficient",
                        "confidence": 1.0,
                        "explanation": "Reject any remaining ambiguous candidate.",
                    }
                    for item in input_payload["relations"]
                ]
            }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(
            payload=payload, provider_metadata={"provider": "compound-audit"}
        )


class _IncompleteEndpointAuditLlm(FakeLlmProvider):
    """Recover a relation target after the ordinary entity pass omits it."""

    def complete_structured(
        self, request: StructuredCompletionRequest
    ) -> StructuredCompletionResult:
        """Expose HiddenSet only to the cue-focused entity audit."""

        input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
        if request.task_name == "document_context_extraction":
            payload: object = {"entities": []}
        elif request.task_name == "entity_extraction":
            entities = [_entity("Method Alpha", "MODEL")]
            if str(input_payload["window_id"]).startswith("cue_entity_audit_window"):
                entities = [_entity("HiddenSet", "DATASET")]
            payload = {"entities": entities}
        elif request.task_name == "relation_extraction":
            ids = {item["name"]: item["id"] for item in input_payload["entities"]}
            payload = (
                {
                    "relations": [
                        _relation(
                            ids["Method Alpha"],
                            ids["HiddenSet"],
                            "Method Alpha",
                            "HiddenSet",
                            "EVALUATED_ON",
                            "Method Alpha was evaluated on HiddenSet.",
                        )
                    ]
                }
                if {"Method Alpha", "HiddenSet"} <= ids.keys()
                else {"relations": []}
            )
        elif request.task_name == "relation_verification":
            payload = {
                "decisions": [
                    {
                        "relation_id": item["id"],
                        "verdict": "supported",
                        "confidence": 1.0,
                        "explanation": "The sentence explicitly states the evaluation dataset.",
                    }
                    for item in input_payload["relations"]
                ]
            }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(
            payload=payload,
            provider_metadata={"provider": "incomplete-endpoint-audit"},
        )


def test_two_pass_keeps_valid_records_when_siblings_are_invalid() -> None:
    """Ensure record-local validation preserves good records across all extraction stages.

    Rejection reasons and direction repairs must remain auditable.
    """

    chunk = Chunk(
        id="chunk-1",
        file_id="file-1",
        document_id="document-1",
        page_number=1,
        chunk_index=0,
        content="Jigoro Kano founded the Kodokan in Tokyo.",
        start_offset=0,
        end_offset=43,
        token_count=7,
        content_hash="hash",
    )
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)
    settings = GraphSettings(
        gleaning_max_passes=0,
        entity_resolution_enabled=False,
        verification_min_confidence=0.75,
    )

    result = extract_graph_two_pass(
        [chunk],
        _MixedRecordLlm(),
        HashEmbeddingProvider(),
        EmbedOptions(model="hash", dimension=8, batch_size=8),
        settings,
        ontology.profile,
        ontology.checksum,
        ontology.source,
        model="test-model",
        timeout_seconds=30,
    )

    triples = [
        (item.source_name, item.relation_type, item.target_name) for item in result.relations
    ]
    assert triples == [
        ("Kodokan", "FOUNDED_BY", "Jigoro Kano"),
        ("Kodokan", "LOCATED_IN", "Tokyo"),
    ]
    assert result.relations[0].quote == chunk.content
    relation_trace = next(
        item for item in result.provider_metadata["trace"] if item["stage"] == "relation_extraction"
    )
    assert relation_trace["record_actions"] == {
        "cue_not_found": 1,
        "invalid_endpoint": 1,
        "invalid_schema": 1,
        "repaired_direction": 1,
        "self_loop": 1,
    }
    entity_trace = next(
        item for item in result.provider_metadata["trace"] if item["stage"] == "entity_extraction"
    )
    assert entity_trace["record_actions"] == {"invalid_schema": 1}
    verification_trace = next(
        item
        for item in result.provider_metadata["trace"]
        if item["stage"] == "relation_verification"
    )
    assert verification_trace["record_actions"] == {"invalid_schema": 1}


def test_gleaning_audits_partially_covered_chunks_for_missed_facts() -> None:
    """Revisit chunks containing some results because presence is not completeness."""

    content = "Capoeira includes the roda. The roda was recognized by UNESCO."
    chunk = Chunk(
        id="chunk-1",
        file_id="file-1",
        document_id="document-1",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=10,
        content_hash="hash",
    )
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)
    settings = GraphSettings(
        gleaning_max_passes=1,
        gleaning_min_uncovered_tokens=1,
        entity_resolution_enabled=False,
    )

    result = extract_graph_two_pass(
        [chunk],
        _CoverageAuditLlm(),
        HashEmbeddingProvider(),
        EmbedOptions(model="hash", dimension=8, batch_size=8),
        settings,
        ontology.profile,
        ontology.checksum,
        ontology.source,
        model="test-model",
        timeout_seconds=30,
    )

    assert {item.name for item in result.entities} == {"Capoeira", "UNESCO", "roda"}
    assert {
        (item.source_name, item.relation_type, item.target_name) for item in result.relations
    } == {
        ("Capoeira", "HAS_PRACTICE", "roda"),
        ("roda", "RECOGNIZED_BY", "UNESCO"),
    }
    trace = result.provider_metadata["trace"]
    assert sum(item["stage"] == "entity_extraction" for item in trace) == 2
    assert sum(item["stage"] == "relation_extraction" for item in trace) == 2


def test_relation_grounding_accepts_verified_local_endpoint_surfaces() -> None:
    """Ground shortened contextual mentions without weakening canonical inventory IDs."""

    content = "The Capoeira roda is recognized by UNESCO. Capoeira includes the roda."
    chunk = Chunk(
        id="chunk-1",
        file_id="file-1",
        document_id="document-1",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=11,
        content_hash="hash",
    )
    window = ExtractionWindow(
        id="window-1",
        document_id="document-1",
        chunks=[chunk],
        token_count=chunk.token_count,
    )
    entities = [
        EntityMention(
            id="capoeira",
            name="Capoeira",
            type="MARTIAL_ART",
            description="A martial art.",
            source_chunk_id=chunk.id,
            quote="Capoeira",
        ),
        EntityMention(
            id="roda",
            name="Capoeira roda",
            type="PRACTICE",
            description="The capoeira circle practice.",
            source_chunk_id=chunk.id,
            quote="Capoeira roda",
        ),
    ]
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)

    outcome = LlmRelationExtractor(_LocalSurfaceLlm()).extract(
        window,
        entities,
        ontology.profile,
        model="test-model",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    assert outcome.relations[0].source_surface == "Capoeira"
    assert outcome.relations[0].target_surface == "roda"
    assert outcome.relations[0].quote == "Capoeira includes the roda."


def test_ontology_cues_recover_relations_missed_by_generation() -> None:
    """Retain unambiguous typed cue evidence without a stochastic verifier veto."""

    content = "UFC 1 was part of the Ultimate Fighting Championship."
    chunk = Chunk(
        id="chunk-1",
        file_id="file-1",
        document_id="document-1",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=9,
        content_hash="hash",
    )
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)

    result = extract_graph_two_pass(
        [chunk],
        _CueRecoveryLlm(),
        HashEmbeddingProvider(),
        EmbedOptions(model="hash", dimension=8, batch_size=8),
        GraphSettings(gleaning_max_passes=0, entity_resolution_enabled=False),
        ontology.profile,
        ontology.checksum,
        ontology.source,
        model="test-model",
        timeout_seconds=30,
    )

    assert [
        (item.source_name, item.relation_type, item.target_name) for item in result.relations
    ] == [("UFC 1", "PART_OF", "Ultimate Fighting Championship")]
    cue_trace = next(
        item
        for item in result.provider_metadata["trace"]
        if item["stage"] == "cue_candidate_generation"
    )
    assert cue_trace["additional_records"] >= 1
    assert cue_trace["direct_evidence_records"] >= 1


def test_cue_entity_audit_recovers_compound_relation_subject() -> None:
    """Recover a named compound instead of attaching its relation to a substring."""

    content = (
        "A capoeira roda is a recurring community practice. "
        "The capoeira roda was recognized by UNESCO in 2014."
    )
    chunk = Chunk(
        id="chunk-1",
        file_id="file-1",
        document_id="document-1",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=17,
        content_hash="hash",
    )
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)

    result = extract_graph_two_pass(
        [chunk],
        _CompoundSubjectAuditLlm(),
        HashEmbeddingProvider(),
        EmbedOptions(model="hash", dimension=8, batch_size=8),
        GraphSettings(gleaning_max_passes=0, entity_resolution_enabled=False),
        ontology.profile,
        ontology.checksum,
        ontology.source,
        model="test-model",
        timeout_seconds=30,
    )

    assert any(entity.name == "capoeira roda" for entity in result.entities)
    assert [
        (item.source_name, item.relation_type, item.target_name) for item in result.relations
    ] == [("capoeira roda", "RECOGNIZED_BY", "UNESCO")]


def test_cue_entity_audit_recovers_a_completely_missing_endpoint() -> None:
    """Revisit cue sentences whose first entity pass cannot form any relation."""

    content = "Method Alpha was evaluated on HiddenSet."
    chunk = Chunk(
        id="chunk-1",
        file_id="file-1",
        document_id="document-1",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=7,
        content_hash="hash",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None)

    result = extract_graph_two_pass(
        [chunk],
        _IncompleteEndpointAuditLlm(),
        HashEmbeddingProvider(),
        EmbedOptions(model="hash", dimension=8, batch_size=8),
        GraphSettings(gleaning_max_passes=0, entity_resolution_enabled=False),
        ontology.profile,
        ontology.checksum,
        ontology.source,
        model="test-model",
        timeout_seconds=30,
    )

    assert {entity.name for entity in result.entities} == {"Method Alpha", "HiddenSet"}
    assert [
        (item.source_name, item.relation_type, item.target_name) for item in result.relations
    ] == [("Method Alpha", "EVALUATED_ON", "HiddenSet")]


def test_reference_filter_skips_bibliographies_but_keeps_cited_prose() -> None:
    """Require several reference entries before suppressing provider extraction."""

    bibliography = """References
[19] S. Li et al. Composing simple image descriptions. 2011.
[20] T. Lin et al. Microsoft COCO. 2014.
[21] J. Mao et al. Explain images with neural networks. 2014.
[22] T. Mikolov et al. Word representations. 2013.
"""
    numbered_bibliography = """
3. Konorski, J. Integrative Activity of the Brain. 1967.
4. Logothetis, N. K. Visual object recognition. 1996.
5. Riesenhuber, M. Neural mechanisms of recognition. 2002.
6. Young, M. P. Sparse population coding. 1992.
"""
    author_year_bibliography = """
Muller-Budack, E., Pustu-Iren, K., and Ewerth, R. Geolocation estimation.
In Proceedings of ECCV, 2018.
Murty, S., Koh, P. W., and Liang, P. Expbert. arXiv preprint arXiv:2005.01932, 2020.
Narasimhan, K., Kulkarni, T., and Barzilay, R. Language understanding. arXiv:1506.08941, 2015.
Netzer, Y., Wang, T., Coates, A., and Ng, A. Reading digits. In a conference workshop, 2011.
"""
    cited_prose = "Our method follows prior work [19] but introduces a new objective."
    historical_prose = """
A. Smith described the apparatus used in our study.
B. Jones repeated the experiment with a larger sample.
C. García calibrated the instrument before each trial.
D. Chen independently reviewed every measurement.
The experiments ran from 2018 through 2021.
"""

    assert is_reference_only_window(_window(bibliography))
    assert is_reference_only_window(_window(numbered_bibliography))
    assert is_reference_only_window(_window(author_year_bibliography))
    assert not is_reference_only_window(_window(cited_prose))
    assert not is_reference_only_window(_window(historical_prose))


def test_relation_extraction_grounds_document_context_through_explicit_pronoun() -> None:
    """Let independent windows attach an authored claim to their source document."""

    content = "We introduce Adam, an algorithm for stochastic optimization."
    window = _window(content)
    paper = EntityMention(
        id="paper",
        name="Adam: A Method for Stochastic Optimization",
        type="PAPER",
        description="The source paper.",
        source_chunk_id="front-matter",
        quote="Adam: A Method for Stochastic Optimization",
        is_document_context=True,
        contextual_surfaces=["we", "this paper"],
    )
    method = EntityMention(
        id="method",
        name="Adam",
        type="METHOD",
        description="An optimization method.",
        source_chunk_id="reference-chunk",
        quote="Adam",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = LlmRelationExtractor(_DocumentSubjectRelationLlm()).extract(
        window,
        [paper, method],
        ontology,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    relation = outcome.relations[0]
    assert relation.source_entity_id == paper.id
    assert relation.target_entity_id == method.id
    assert relation.source_surface == "We"
    assert relation.relation_type == "INTRODUCES"


def test_relation_extraction_uses_implicit_source_paper_provenance() -> None:
    """Ground an authorial paper relation without requiring its title in every sentence."""

    content = "To minimize E by gradient descent, it is necessary to compute the derivative of E."
    window = _window(content)
    paper = EntityMention(
        id="paper",
        name="Learning representations by back-propagating errors",
        type="PAPER",
        description="The source paper.",
        source_chunk_id="front-matter",
        quote="Learning representations by back-propagating errors",
        is_document_context=True,
        contextual_surfaces=["we", "this paper"],
    )
    method = EntityMention(
        id="gradient-descent",
        name="Gradient descent",
        type="METHOD",
        description="An optimization method.",
        source_chunk_id="chunk-1",
        quote="gradient descent",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = LlmRelationExtractor(_ImplicitDocumentSubjectRelationLlm()).extract(
        window,
        [paper, method],
        ontology,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    assert outcome.relations[0].source_entity_id == paper.id
    assert outcome.relations[0].target_entity_id == method.id
    assert outcome.relations[0].source_surface == paper.name
    assert outcome.trace["record_actions"]["implicit_document_source"] == 1
    assert outcome.trace["record_actions"]["repaired_source_surface"] == 1


def test_relation_extraction_uses_provenance_for_target_nested_in_paper_title() -> None:
    """Allow a title claim when the nested task cannot be a second text span."""

    title = "Understanding the Difficulty of Training Deep Feedforward Neural Networks"
    window = _window(title)
    paper = EntityMention(
        id="paper",
        name=title,
        type="PAPER",
        description="The source paper.",
        source_chunk_id="reference-chunk",
        quote=title,
        is_document_context=True,
        contextual_surfaces=["we", "this paper"],
    )
    task = EntityMention(
        id="task",
        name="Training Deep Feedforward Neural Networks",
        type="TASK",
        description="The training task named by the title.",
        source_chunk_id="reference-chunk",
        quote="Training Deep Feedforward Neural Networks",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = LlmRelationExtractor(_NestedTitleRelationLlm()).extract(
        window,
        [paper, task],
        ontology,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    assert outcome.relations[0].source_entity_id == paper.id
    assert outcome.relations[0].target_entity_id == task.id
    assert outcome.trace["record_actions"]["implicit_document_source"] == 1


def test_relation_extraction_does_not_infer_implicit_non_paper_source() -> None:
    """Keep method and model endpoints subject to exact local surface grounding."""

    content = "To minimize E by gradient descent, derivatives are computed efficiently."
    window = _window(content)
    source = EntityMention(
        id="back-propagation",
        name="Back-propagation",
        type="METHOD",
        description="A differentiation method.",
        source_chunk_id="front-matter",
        quote="Back-propagation",
        is_document_context=True,
        contextual_surfaces=["the method"],
    )
    target = EntityMention(
        id="gradient-descent",
        name="Gradient descent",
        type="METHOD",
        description="An optimization method.",
        source_chunk_id="chunk-1",
        quote="gradient descent",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = LlmRelationExtractor(_ImplicitMethodSubjectRelationLlm()).extract(
        window,
        [source, target],
        ontology,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert outcome.relations == []
    assert outcome.trace["record_actions"]["ungrounded_quote"] == 1


def test_relation_grounding_recovers_normalized_surface_from_inventory() -> None:
    """Use a canonical identity fallback when a model-normalized local label is absent."""

    content = (
        "The Stochastic Gradient Variational Bayes estimator uses the reparameterization trick."
    )
    window = _window(content)
    source = EntityMention(
        id="sgvb",
        name="Stochastic Gradient Variational Bayes estimator",
        type="METHOD",
        description="A gradient estimator.",
        source_chunk_id="reference-chunk",
        quote="Stochastic Gradient Variational Bayes estimator",
    )
    target = EntityMention(
        id="reparameterization",
        name="reparameterization trick",
        type="METHOD",
        description="A gradient-estimation method.",
        source_chunk_id="reference-chunk",
        quote="reparameterization trick",
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    outcome = LlmRelationExtractor(_NormalizedEndpointSurfaceLlm()).extract(
        window,
        [source, target],
        ontology,
        model="fake",
        timeout_seconds=30,
        max_relations=10,
    )

    assert len(outcome.relations) == 1
    assert outcome.relations[0].quote == content
    assert outcome.relations[0].source_surface == "Stochastic Gradient Variational Bayes estimator"
    assert outcome.trace["record_actions"]["repaired_source_surface"] == 1


def test_single_document_window_inherits_context_grounded_in_another_chunk() -> None:
    """Preserve front-matter entities when distributed workers receive body-only windows.

    A queue task contains a bounded subset of a document, while its reusable paper
    or model context is normally grounded in an earlier front-matter chunk. The
    context must remain in the observation inventory so body pronouns can reference
    it and finalization cannot silently drop the focal source entity.
    """

    window = _window("The body describes a generally useful optimization method.")
    paper = EntityMention(
        id="paper",
        name="A General Optimization Method",
        type="PAPER",
        description="The source paper.",
        source_chunk_id="front-matter-outside-this-window",
        quote="A General Optimization Method",
        is_document_context=True,
        contextual_surfaces=["we", "this paper"],
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    observations = extract_graph_observations(
        window.chunks,
        FakeLlmProvider(),
        extraction_window_tokens=1_000,
        max_chunks_per_llm_call=2,
        graph_settings=GraphSettings(extraction_parallelism=1, gleaning_max_passes=0),
        ontology=ontology,
        model="fake",
        timeout_seconds=30,
        document_context_entities=[paper],
    )

    assert any(entity.id == paper.id for entity in observations.entities)


class _CrossWindowRelationLlm(FakeLlmProvider):
    """Omit a repeated endpoint locally so the document inventory is required."""

    def complete_structured(
        self,
        request: StructuredCompletionRequest,
    ) -> StructuredCompletionResult:
        """Return one entity per window and a relation that needs both windows."""

        if request.task_name == "entity_extraction":
            if "Introduces Method Alpha" in request.user:
                entities = [
                    {
                        **_entity("Method Alpha", "METHOD"),
                        "source_chunk_id": "chunk-alpha",
                        "quote": "Introduces Method Alpha.",
                    }
                ]
            else:
                entities = [
                    {
                        **_entity("Method Beta", "METHOD"),
                        "source_chunk_id": "chunk-beta",
                        "quote": "Method Beta builds on Method Alpha.",
                    }
                ]
            payload: object = {"entities": entities}
        elif request.task_name == "relation_extraction":
            input_payload = json.loads(request.user.split("INPUT_JSON:\n", 1)[1])
            ids = {item["name"]: item["id"] for item in input_payload["entities"]}
            if "Method Alpha" not in ids or "Method Beta" not in ids:
                payload = {"relations": []}
            else:
                payload = {
                    "relations": [
                        {
                            "source_entity_id": ids["Method Beta"],
                            "target_entity_id": ids["Method Alpha"],
                            "source_surface": "Method Beta",
                            "target_surface": "Method Alpha",
                            "relation_type": "BUILDS_ON",
                            "description": "Method Beta builds on Method Alpha.",
                            "source_chunk_id": "chunk-beta",
                            "quote": "Method Beta builds on Method Alpha.",
                            "confidence": 0.99,
                        }
                    ]
                }
        else:
            return super().complete_structured(request)
        return StructuredCompletionResult(
            payload=payload,
            raw_text=json.dumps(payload),
            provider_metadata={"provider": "cross-window-test"},
        )


def test_relation_phase_uses_entities_discovered_in_other_document_windows() -> None:
    """Connect endpoints across windows without serializing either provider phase."""

    first = (
        _window("Introduces Method Alpha.")
        .chunks[0]
        .model_copy(update={"id": "chunk-alpha", "chunk_index": 0})
    )
    second = (
        _window("Method Beta builds on Method Alpha.")
        .chunks[0]
        .model_copy(update={"id": "chunk-beta", "chunk_index": 1})
    )
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    observations = extract_graph_observations(
        [first, second],
        _CrossWindowRelationLlm(),
        extraction_window_tokens=1_000,
        max_chunks_per_llm_call=1,
        graph_settings=GraphSettings(
            extraction_parallelism=2,
            gleaning_max_passes=0,
            verify_relations=False,
        ),
        ontology=ontology,
        model="fake",
        timeout_seconds=30,
    )

    names = {entity.name for entity in observations.entities}
    assert names == {"Method Alpha", "Method Beta"}
    assert len(observations.relations) == 1
    assert observations.relations[0].relation_type == "BUILDS_ON"


def test_relation_inventory_compacts_duplicates_and_omits_ungrounded_entities() -> None:
    """Keep cross-window aliases while excluding identities unusable in this window."""

    window = _window("Local Method uses its LM abbreviation.")
    duplicate_first = EntityMention(
        id="local-first",
        name="Local Method",
        type="METHOD",
        aliases=[],
        description="The first source observation.",
        source_chunk_id="outside-first",
        quote="Local Method",
    )
    duplicate_second = duplicate_first.model_copy(
        update={
            "id": "local-second",
            "aliases": ["LM"],
            "source_chunk_id": "outside-second",
        }
    )
    remote = EntityMention(
        id="remote",
        name="Remote Dataset",
        type="DATASET",
        description="An endpoint established in another window.",
        source_chunk_id="outside-remote",
        quote="Remote Dataset",
    )

    inventory = _relation_inventory_for_window(
        [remote, duplicate_first, duplicate_second],
        window,
    )

    assert [entity.id for entity in inventory] == ["local-first"]
    assert inventory[0].aliases == ["LM"]


def test_relation_completion_inventory_keeps_only_uncovered_cue_typed_endpoints() -> None:
    """Focus the second pass without allowing a cue to invent incompatible pairs."""

    window = _window("Residual Network is trained on ImageNet. Alice Smith observes.")
    entities = [
        EntityMention(
            id="model",
            name="Residual Network",
            type="MODEL",
            description="A trained model.",
            source_chunk_id="reference-chunk",
            quote="Residual Network",
        ),
        EntityMention(
            id="dataset",
            name="ImageNet",
            type="DATASET",
            description="A training dataset.",
            source_chunk_id="reference-chunk",
            quote="ImageNet",
        ),
        EntityMention(
            id="person",
            name="Alice Smith",
            type="PERSON",
            description="An observer unrelated to the training predicate.",
            source_chunk_id="reference-chunk",
            quote="Alice Smith",
        ),
    ]
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    completion = _relation_completion_inventory(window, entities, ontology, [])

    assert [entity.id for entity in completion] == ["model", "dataset"]


def test_relation_completion_inventory_keeps_endpoints_for_cueless_relations() -> None:
    """Do not let an unrelated lexical cue suppress layout-defined authorship."""

    window = _window("Research Paper. Alice Smith. Residual Network is trained on ImageNet.")
    entities = [
        EntityMention(
            id="paper",
            name="Research Paper",
            type="PAPER",
            description="The focal publication.",
            source_chunk_id="reference-chunk",
            quote="Research Paper",
            is_document_context=True,
        ),
        EntityMention(
            id="author",
            name="Alice Smith",
            type="PERSON",
            description="A front-matter author.",
            source_chunk_id="reference-chunk",
            quote="Alice Smith",
        ),
        EntityMention(
            id="model",
            name="Residual Network",
            type="MODEL",
            description="A trained model.",
            source_chunk_id="reference-chunk",
            quote="Residual Network",
        ),
        EntityMention(
            id="dataset",
            name="ImageNet",
            type="DATASET",
            description="A training dataset.",
            source_chunk_id="reference-chunk",
            quote="ImageNet",
        ),
    ]
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None).profile

    completion = _relation_completion_inventory(window, entities, ontology, [])

    assert [entity.id for entity in completion] == ["paper", "author", "model", "dataset"]


def _window(content: str) -> ExtractionWindow:
    """Build one deterministic extraction window for reference-filter tests."""

    chunk = Chunk(
        id="reference-chunk",
        file_id="reference-file",
        document_id="reference-document",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=max(1, len(content.split())),
        content_hash="reference-hash",
    )
    return ExtractionWindow(
        id="reference-window",
        document_id=chunk.document_id,
        chunks=[chunk],
        token_count=chunk.token_count,
    )


def _entity(name: str, entity_type: str) -> dict[str, object]:
    """Build one strict entity candidate used by the mixed-record provider fixture.

    Confidence and source fields remain valid by default.
    """

    return {
        "name": name,
        "type": entity_type,
        "description": f"{name} appears in the source.",
        "source_chunk_id": "chunk-1",
        "quote": name,
        "confidence": 1.0,
        "aliases": [],
    }


def _relation(
    source_entity_id: str,
    target_entity_id: str,
    source_surface: str,
    target_surface: str,
    relation_type: str,
    quote: str,
) -> dict[str, object]:
    """Build one strict relation candidate for the coverage-audit fixture."""

    return {
        "source_entity_id": source_entity_id,
        "target_entity_id": target_entity_id,
        "source_surface": source_surface,
        "target_surface": target_surface,
        "relation_type": relation_type,
        "description": quote,
        "source_chunk_id": "chunk-1",
        "quote": quote,
        "confidence": 1.0,
    }
