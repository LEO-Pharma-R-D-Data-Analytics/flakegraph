from __future__ import annotations

from kg_processor.application.graph_filter import (
    filter_entities,
    filter_entities_with_decisions,
    filter_relations,
    filter_relations_with_decisions,
)
from kg_processor.application.graph_merge import (
    assemble_graph,
    assemble_graph_with_decisions,
    normalize_entity_name,
    normalize_relation_type,
)
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation


def test_normalization_helpers() -> None:
    assert normalize_entity_name("Next.js") == "nextjs"
    assert normalize_relation_type(" Located In ") == "located_in"


def test_assemble_graph_deduplicates_entities_and_tracks_evidence() -> None:
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith works at Acme Corp.",
        start_offset=0,
        end_offset=32,
        token_count=6,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith is a person.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith works at Acme Corp.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Acme Corp is an organization.",
            source_chunk_id=chunk.id,
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            relation_type="works at",
            description="Alice Smith works at Acme Corp.",
            source_chunk_id=chunk.id,
            weight=12,
        )
    ]

    filtered_entities = filter_entities(entities, {chunk.id: chunk})
    filtered_relations = filter_relations(relations, filtered_entities)
    nodes, edges, evidence, sources = assemble_graph(
        "graph", [chunk], filtered_entities, filtered_relations, relation_weight_max=10
    )

    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0].weight == 10
    assert len(evidence) >= 3
    assert len(sources) == 2


def test_assemble_graph_records_merge_decisions() -> None:
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith works at Acme Corp.",
        start_offset=0,
        end_offset=32,
        token_count=6,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith is present.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith works at Acme Corp.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Acme is present.",
            source_chunk_id=chunk.id,
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            relation_type="works at",
            description="First observation.",
            source_chunk_id=chunk.id,
            weight=7,
        ),
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            relation_type="works at",
            description="Longer duplicate relation observation.",
            source_chunk_id=chunk.id,
            weight=7,
        ),
        ExtractedRelation(
            source_name="Ghost",
            target_name="Acme Corp",
            relation_type="mentions",
            description="Source node is missing.",
            source_chunk_id=chunk.id,
        ),
    ]

    result = assemble_graph_with_decisions("graph", [chunk], entities, relations, 10)

    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    assert result.edges[0].weight == 10
    assert result.decision_reason_counts() == {
        "canonical_edge_created": 1,
        "canonical_node_created": 1,
        "entity_observations_merged": 1,
        "relation_evidence_aggregated_weight_clamped": 1,
        "source_node_missing": 1,
    }
    merge_events = [decision.to_trace_event() for decision in result.decisions]
    assert any(
        event["kind"] == "entity"
        and event["action"] == "merged"
        and event["reason"] == "entity_observations_merged"
        and event["observation_count"] == 2
        for event in merge_events
    )
    assert any(
        event["kind"] == "relation"
        and event["action"] == "aggregated"
        and event["new_weight"] == 10
        for event in merge_events
    )
    assert any(
        event["kind"] == "relation"
        and event["action"] == "dropped"
        and event["reason"] == "source_node_missing"
        for event in merge_events
    )


def test_assemble_graph_uses_extracted_quote_spans_for_evidence() -> None:
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith works at Acme Corp.",
        start_offset=100,
        end_offset=132,
        token_count=6,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith is present.",
            source_chunk_id=chunk.id,
            quote="Alice Smith",
            start_offset=0,
            end_offset=11,
        ),
        ExtractedEntity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Acme is present.",
            source_chunk_id=chunk.id,
            quote="Acme Corp",
            start_offset=21,
            end_offset=30,
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            relation_type="works at",
            description="Alice works at Acme.",
            source_chunk_id=chunk.id,
            quote="works at",
            start_offset=12,
            end_offset=20,
        )
    ]

    nodes, edges, evidence, _sources = assemble_graph(
        "graph",
        [chunk],
        entities,
        relations,
        relation_weight_max=10,
    )

    node_by_name = {node.name: node for node in nodes}
    alice_evidence = next(
        row for row in evidence if row.subject_id == node_by_name["Alice Smith"].id
    )
    edge_evidence = next(row for row in evidence if row.subject_id == edges[0].id)
    assert alice_evidence.quote == "Alice Smith"
    assert alice_evidence.start_offset == 100
    assert alice_evidence.end_offset == 111
    assert edge_evidence.quote == "works at"
    assert edge_evidence.start_offset == 112
    assert edge_evidence.end_offset == 120


def test_filter_entities_applies_confidence_name_quality_and_blocklist() -> None:
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith works at Acme Corp.",
        start_offset=0,
        end_offset=32,
        token_count=6,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice is present.",
            source_chunk_id=chunk.id,
            confidence=0.95,
        ),
        ExtractedEntity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Acme is present.",
            source_chunk_id=chunk.id,
            confidence=0.2,
        ),
        ExtractedEntity(
            name="Q",
            type="CONCEPT",
            description="Too short.",
            source_chunk_id=chunk.id,
            confidence=0.99,
        ),
        ExtractedEntity(
            name="Document",
            type="CONCEPT",
            description="Blocked document furniture.",
            source_chunk_id=chunk.id,
            confidence=0.99,
        ),
    ]

    result = filter_entities_with_decisions(
        entities,
        {chunk.id: chunk},
        min_confidence=0.8,
        min_name_length=2,
        blocklist=["document"],
    )

    assert [entity.name for entity in result.kept] == ["Alice Smith"]
    assert result.dropped_reason_counts() == {
        "blocklisted_entity": 1,
        "low_confidence": 1,
        "name_too_short": 1,
    }
    assert [
        decision.to_trace_event()["reason"]
        for decision in result.decisions
        if decision.action == "dropped"
    ] == ["low_confidence", "name_too_short", "blocklisted_entity"]


def test_filter_relations_applies_confidence_and_endpoint_grounding() -> None:
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith works at Acme Corp.",
        start_offset=0,
        end_offset=32,
        token_count=6,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice is present.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Acme is present.",
            source_chunk_id=chunk.id,
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            relation_type="works_at",
            description="Alice works at Acme.",
            source_chunk_id=chunk.id,
            confidence=0.9,
        ),
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Missing Org",
            relation_type="works_at",
            description="Ungrounded endpoint.",
            source_chunk_id=chunk.id,
            confidence=0.9,
        ),
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Acme Corp",
            relation_type="mentions",
            description="Low confidence.",
            source_chunk_id=chunk.id,
            confidence=0.2,
        ),
    ]

    result = filter_relations_with_decisions(
        relations,
        entities,
        {chunk.id: chunk},
        min_confidence=0.8,
        require_endpoint_grounding=True,
    )

    assert [relation.relation_type for relation in result.kept] == ["works_at"]
    assert result.dropped_reason_counts() == {
        "low_confidence": 1,
        "target_entity_missing": 1,
    }


def test_filter_relations_records_endpoint_grounding_reasons() -> None:
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice Smith works at Acme Corp.",
        start_offset=0,
        end_offset=32,
        token_count=6,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice is present.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="Copenhagen",
            type="LOCATION",
            description="Known from another accepted observation.",
            source_chunk_id=chunk.id,
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice Smith",
            target_name="Copenhagen",
            relation_type="located_in",
            description="Endpoint is known but not grounded in this chunk.",
            source_chunk_id=chunk.id,
        )
    ]

    result = filter_relations_with_decisions(
        relations,
        entities,
        {chunk.id: chunk},
        require_endpoint_grounding=True,
    )

    assert result.kept == []
    assert result.dropped_reason_counts() == {"target_endpoint_ungrounded": 1}
