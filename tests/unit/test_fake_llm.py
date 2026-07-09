from __future__ import annotations

from kg_processor.adapters.llm.fake import FakeLlmProvider
from kg_processor.domain.graph import Chunk
from kg_processor.ports.llm import LlmOptions


def test_fake_llm_does_not_join_entity_names_across_lines() -> None:
    chunk = Chunk(
        id="chunk_1",
        graph_id="graph",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content=(
            "Alice Smith works at Acme Corp in Copenhagen.\n"
            "Acme Corp collaborates with Contoso Health on knowledge graph tooling.\n"
            "Copenhagen teams evaluate MinerU OCR and Azure OpenAI."
        ),
        start_offset=0,
        end_offset=173,
        token_count=24,
        content_hash="hash",
    )
    options = LlmOptions(
        model="fake",
        entity_types=["PERSON", "ORGANIZATION", "CONCEPT"],
    )

    result = FakeLlmProvider().extract_graph([chunk], options)

    assert [entity.name for entity in result.entities] == [
        "Alice Smith",
        "Acme Corp",
        "Copenhagen",
        "Contoso Health",
        "MinerU OCR",
        "Azure OpenAI",
    ]
    assert all("\n" not in entity.name for entity in result.entities)
    assert all("\n" not in relation.source_name for relation in result.relations)
    assert all("\n" not in relation.target_name for relation in result.relations)


def test_fake_llm_respects_batch_record_limits() -> None:
    chunk = Chunk(
        id="chunk_1",
        graph_id="graph",
        file_id="file_1",
        page_number=1,
        chunk_index=0,
        content=(
            "Alice Smith works with Bob Jones, Carol White, David Black, "
            "Eve Green, Frank Brown, and Grace Stone at Acme Corp."
        ),
        start_offset=0,
        end_offset=122,
        token_count=21,
        content_hash="hash",
    )
    options = LlmOptions(
        model="fake",
        entity_types=["PERSON", "ORGANIZATION", "CONCEPT"],
        max_entities_per_batch=3,
        max_relations_per_batch=2,
    )

    result = FakeLlmProvider().extract_graph([chunk], options)

    assert [entity.name for entity in result.entities] == [
        "Alice Smith",
        "Bob Jones",
        "Carol White",
    ]
    assert [(relation.source_name, relation.target_name) for relation in result.relations] == [
        ("Alice Smith", "Bob Jones"),
        ("Bob Jones", "Carol White"),
    ]
    assert result.provider_metadata["max_entities_per_batch"] == 3
    assert result.provider_metadata["max_relations_per_batch"] == 2
