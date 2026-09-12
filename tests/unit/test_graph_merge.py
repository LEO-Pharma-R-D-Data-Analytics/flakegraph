from __future__ import annotations

from kg_processor.application.graph_filter import (
    filter_entities,
    filter_entities_with_decisions,
    filter_relations,
    filter_relations_with_decisions,
)
from kg_processor.application.graph_merge import (
    _best_description,
    _casefold_span,
    assemble_graph,
    assemble_graph_with_decisions,
    normalize_entity_name,
    normalize_relation_type,
    prune_isolated_entities,
)
from kg_processor.domain.graph import Chunk, ExtractedEntity, ExtractedRelation


def test_normalization_helpers() -> None:
    assert normalize_entity_name("Next.js") == "nextjs"
    assert normalize_entity_name("Müller") == "müller"
    assert normalize_entity_name("Mller") == "mller"
    assert normalize_entity_name("東京") == "東京"
    assert normalize_relation_type(" Located In ") == "located_in"
    assert normalize_relation_type("Works-At") == normalize_relation_type("works at")
    assert _best_description(["equal value B", "equal value A"]) == "equal value B"
    assert _best_description(["equal value A", "equal value B"]) == "equal value B"


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
            aliases=["A. Smith"],
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
    assert next(node for node in nodes if node.name == "Alice Smith").aliases == ["A. Smith"]
    assert len(edges) == 1
    assert edges[0].weight == 10
    assert len(evidence) >= 3
    assert len(sources) == 2


def test_prune_isolated_entities_removes_dependent_provenance_but_keeps_edges() -> None:
    """Disconnected observations must not survive as expensive graph entities."""

    chunk = Chunk(
        id="chunk",
        file_id="file",
        page_number=1,
        chunk_index=0,
        content="Alice works at Acme. Incidental appears separately.",
        start_offset=0,
        end_offset=52,
        token_count=7,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Alice", type="PERSON", description="A person.", source_chunk_id=chunk.id
        ),
        ExtractedEntity(
            name="Acme",
            type="ORGANIZATION",
            description="An organization.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="Incidental",
            type="CONCEPT",
            description="An incidental concept.",
            source_chunk_id=chunk.id,
        ),
    ]
    relation = ExtractedRelation(
        source_name="Alice",
        source_type="PERSON",
        target_name="Acme",
        target_type="ORGANIZATION",
        relation_type="WORKS_AT",
        description="Alice works at Acme.",
        source_chunk_id=chunk.id,
    )
    assembly = assemble_graph_with_decisions("graph", [chunk], entities, [relation], 10)

    pruned = prune_isolated_entities(assembly)

    assert {node.name for node in pruned.nodes} == {"Alice", "Acme"}
    assert len(pruned.edges) == 1
    assert {source.node_id for source in pruned.entity_sources} == {
        node.id for node in pruned.nodes
    }
    assert all(
        item.subject_kind != "node" or item.subject_id in {node.id for node in pruned.nodes}
        for item in pruned.evidence
    )
    assert any(
        decision.reason == "isolated_entity_without_relation" for decision in pruned.decisions
    )


def test_assemble_graph_keeps_same_name_with_distinct_types_addressable() -> None:
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Jordan teaches grappling at North Hall.",
        start_offset=0,
        end_offset=39,
        token_count=6,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Jordan",
            type="PERSON",
            description="Jordan is a person.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="Jordan",
            type="ORGANIZATION",
            description="Jordan is also mislabeled as an organization.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="North Hall",
            type="LOCATION",
            description="North Hall is a location.",
            source_chunk_id=chunk.id,
        ),
    ]
    relations = [
        ExtractedRelation(
            source_name="Jordan",
            target_name="North Hall",
            source_type="PERSON",
            target_type="LOCATION",
            relation_type="teaches at",
            description="Jordan teaches at North Hall.",
            source_chunk_id=chunk.id,
        )
    ]

    nodes, edges, _evidence, _sources = assemble_graph(
        "graph",
        [chunk],
        entities,
        relations,
        relation_weight_max=10,
    )

    jordan = next(node for node in nodes if node.name == "Jordan" and node.primary_type == "PERSON")
    assert len(nodes) == 3
    assert jordan.primary_type == "PERSON"
    assert jordan.types == ["PERSON"]
    assert edges[0].source_node_id == jordan.id
    assert jordan.degree == 1


def test_assemble_graph_drops_loop_created_by_entity_normalization() -> None:
    """Do not let punctuation variants become an invalid canonical self-loop."""

    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Wal-Mart renamed Walmart.",
        start_offset=0,
        end_offset=25,
        token_count=3,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name=name,
            type="ORGANIZATION",
            description=name,
            source_chunk_id=chunk.id,
        )
        for name in ("Wal-Mart", "Walmart")
    ]
    relation = ExtractedRelation(
        source_name="Wal-Mart",
        target_name="Walmart",
        relation_type="renamed to",
        description=chunk.content,
        source_chunk_id=chunk.id,
    )

    result = assemble_graph_with_decisions("graph", [chunk], entities, [relation], 10)

    assert len(result.nodes) == 1
    assert result.edges == []
    assert result.decision_reason_counts()["self_loop_after_entity_merge"] == 1


def test_assemble_graph_records_merge_decisions() -> None:
    """Ensure graph assembly emits explicit decisions for created and merged observations.

    The trace should explain graph cardinality changes.
    """

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
    # Duplicate model output for the same edge/chunk must not amplify weight.
    assert result.edges[0].weight == 7
    assert len(result.edge_observations) == 1
    assert result.decision_reason_counts() == {
        "canonical_edge_created": 1,
        "canonical_node_created": 1,
        "entity_observations_merged": 1,
        "duplicate_edge_observation": 1,
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
        and event["action"] == "dropped"
        and event["reason"] == "duplicate_edge_observation"
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


def test_assemble_graph_quote_search_preserves_unicode_casefold_offsets() -> None:
    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="İ Alice Smith works at Acme Corp.",
        start_offset=500,
        end_offset=532,
        token_count=7,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Alice Smith",
            type="PERSON",
            description="Alice Smith is present.",
            source_chunk_id=chunk.id,
            quote="Alice Smith",
        )
    ]

    nodes, _edges, evidence, _sources = assemble_graph(
        "graph",
        [chunk],
        entities,
        [],
        relation_weight_max=10,
    )

    alice_evidence = next(row for row in evidence if row.subject_id == nodes[0].id)
    assert alice_evidence.quote == "Alice Smith"
    assert alice_evidence.start_offset == 502
    assert alice_evidence.end_offset == 513


def test_casefold_span_maps_nfc_match_back_to_original_nfd_offsets() -> None:
    content = "Mu\u0308ller Straße München"

    start, end = _casefold_span(content, "STRASSE")

    assert (start, end) == (content.index("Straße"), content.index("Straße") + len("Straße"))
    assert content[start:end] == "Straße"


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


def test_filter_entities_accepts_a_grounded_alias_after_canonical_resolution() -> None:
    """Keep a mention when its canonical spelling comes from another source chunk."""

    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="The paper studies image recognition.",
        start_offset=0,
        end_offset=36,
        token_count=6,
        content_hash="hash",
    )
    entity = ExtractedEntity(
        name="Image classification",
        type="TASK",
        description="A visual recognition task.",
        source_chunk_id=chunk.id,
        aliases=["image recognition"],
        confidence=0.99,
    )

    result = filter_entities_with_decisions([entity], {chunk.id: chunk})

    assert result.kept == [entity]
    assert result.decisions[0].reason == "grounded_alias"


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


def test_filter_relations_grounds_verified_local_endpoint_surfaces() -> None:
    """Preserve alias and document-context evidence after canonical resolution."""

    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="We evaluate LSTM on the benchmark.",
        start_offset=0,
        end_offset=34,
        token_count=7,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Sequence Learning Paper",
            type="PAPER",
            description="The source paper.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="Long Short-Term Memory",
            type="MODEL",
            description="A recurrent architecture.",
            source_chunk_id=chunk.id,
            aliases=["LSTM"],
        ),
    ]
    relation = ExtractedRelation(
        source_name="Sequence Learning Paper",
        target_name="Long Short-Term Memory",
        source_surface="We",
        target_surface="LSTM",
        relation_type="USES_METHOD",
        description="The paper evaluates LSTM.",
        source_chunk_id=chunk.id,
    )

    result = filter_relations_with_decisions(
        [relation],
        entities,
        {chunk.id: chunk},
        require_endpoint_grounding=True,
    )

    assert result.kept == [relation]
    assert result.dropped_reason_counts() == {}


def test_resolved_alias_remains_a_relation_endpoint_through_assembly() -> None:
    """Keep local relation surfaces after entity resolution canonicalizes a name."""

    chunk = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="LSTM uses an input gate.",
        start_offset=0,
        end_offset=24,
        token_count=5,
        content_hash="hash",
    )
    entities = [
        ExtractedEntity(
            name="Long Short-Term Memory",
            type="MODEL",
            aliases=["LSTM"],
            description="A recurrent architecture.",
            source_chunk_id=chunk.id,
        ),
        ExtractedEntity(
            name="input gate",
            type="METHOD",
            description="A multiplicative gate.",
            source_chunk_id=chunk.id,
        ),
    ]
    relation = ExtractedRelation(
        source_name="LSTM",
        source_type="MODEL",
        target_name="input gate",
        target_type="METHOD",
        relation_type="USES_METHOD",
        description="LSTM uses an input gate.",
        source_chunk_id=chunk.id,
    )

    filtered = filter_relations_with_decisions([relation], entities)
    assembly = assemble_graph_with_decisions(
        "graph",
        [chunk],
        entities,
        filtered.kept,
        relation_weight_max=10,
    )

    assert filtered.kept == [relation]
    assert len(assembly.edges) == 1
    source = next(node for node in assembly.nodes if node.id == assembly.edges[0].source_node_id)
    assert source.name == "Long Short-Term Memory"


def test_ambiguous_resolved_alias_is_not_used_as_a_relation_endpoint() -> None:
    """Avoid assigning a shared acronym to an arbitrary same-typed entity."""

    entities = [
        ExtractedEntity(
            name=name,
            type="MODEL",
            aliases=["SRN"],
            description=name,
            source_chunk_id="chunk",
        )
        for name in ("Simple Recurrent Network", "Scene Representation Network")
    ]
    entities.append(
        ExtractedEntity(
            name="benchmark",
            type="DATASET",
            description="A benchmark.",
            source_chunk_id="chunk",
        )
    )
    relation = ExtractedRelation(
        source_name="SRN",
        source_type="MODEL",
        target_name="benchmark",
        target_type="DATASET",
        relation_type="EVALUATED_ON",
        description="SRN is evaluated on the benchmark.",
        source_chunk_id="chunk",
    )

    result = filter_relations_with_decisions([relation], entities)

    assert result.kept == []
    assert result.dropped_reason_counts() == {"source_entity_missing": 1}


def test_assemble_graph_aggregates_cross_file_assertions_into_one_edge() -> None:
    """Ensure cross-file assertions share topology while preserving source provenance.

    Each assertion must remain independently removable during reindexing.
    """

    first = Chunk(
        id="chunk_1",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content="Alice works at Acme.",
        start_offset=0,
        end_offset=20,
        token_count=4,
        content_hash="one",
    )
    second = first.model_copy(update={"id": "chunk_2", "file_id": "file_2", "content_hash": "two"})
    entities = [
        ExtractedEntity(
            name=name,
            type=entity_type,
            description=name,
            source_chunk_id=chunk.id,
            quote=name,
        )
        for chunk in (first, second)
        for name, entity_type in (("Alice", "PERSON"), ("Acme", "ORGANIZATION"))
    ]
    relations = [
        ExtractedRelation(
            source_name="Alice",
            target_name="Acme",
            relation_type="works_at",
            description=chunk.content,
            source_chunk_id=chunk.id,
            quote=chunk.content,
        )
        for chunk in (first, second)
    ]

    result = assemble_graph_with_decisions(
        "graph", [first, second], entities, relations, relation_weight_max=10
    )

    assert len(result.edges) == 1
    assert result.edges[0].source_file_ids == ["file_1", "file_2"]
    assert result.edges[0].source_chunk_ids == ["chunk_1", "chunk_2"]
    assert result.edges[0].evidence_count == 2
    assert result.edges[0].weight == 2.0
    assert len(result.edge_observations) == 2
