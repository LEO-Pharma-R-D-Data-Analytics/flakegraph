from pathlib import Path

from kg_processor.application.ontology import load_ontology
from kg_processor.application.relation_candidates import discover_cue_relation_candidates
from kg_processor.domain.extraction import EntityMention, ExtractionWindow
from kg_processor.domain.graph import Chunk


def test_cue_candidates_ground_shortened_numeric_event_names() -> None:
    """Recover chronology when official event names appear as stable year prefixes."""

    content = "The Seoul 1988 event preceded the Sydney 2000 event."
    chunk = Chunk(
        id="chunk",
        file_id="file",
        document_id="document",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=12,
        content_hash="hash",
    )
    window = ExtractionWindow(
        id="window",
        document_id="document",
        chunks=[chunk],
        token_count=chunk.token_count,
    )
    entities = [
        _mention("seoul", "Seoul 1988 Olympic Games", chunk.id),
        _mention("sydney", "Sydney 2000 Olympic Games", chunk.id),
    ]
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)

    candidates = discover_cue_relation_candidates(window, entities, ontology.profile)

    preceded = [candidate for candidate in candidates if candidate.relation_type == "PRECEDED"]
    assert len(preceded) == 1
    assert preceded[0].source_entity_id == "seoul"
    assert preceded[0].target_entity_id == "sydney"
    assert preceded[0].source_surface == "Seoul 1988"
    assert preceded[0].target_surface == "Sydney 2000"
    assert preceded[0].verification_required is True


def test_possessive_target_keeps_semantic_verification() -> None:
    """Do not treat a named possessor as the direct object of a cue."""

    content = "The Seoul event preceded taekwondo's medal programme."
    chunk = Chunk(
        id="chunk",
        file_id="file",
        document_id="document",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=7,
        content_hash="hash",
    )
    window = ExtractionWindow(id="window", document_id="document", chunks=[chunk], token_count=7)
    entities = [
        _mention("seoul", "Seoul", chunk.id),
        EntityMention(
            id="taekwondo",
            name="taekwondo",
            type="MARTIAL_ART",
            description="taekwondo",
            source_chunk_id=chunk.id,
            quote="taekwondo",
        ),
    ]
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)

    candidates = discover_cue_relation_candidates(window, entities, ontology.profile)

    preceded = [candidate for candidate in candidates if candidate.relation_type == "PRECEDED"]
    assert len(preceded) == 1
    assert preceded[0].verification_required is True


def test_endpoint_types_disambiguate_shared_ontology_cues() -> None:
    """Treat a shared lexical cue as direct when only one predicate accepts the types."""

    content = "Mixed Martial Arts uses wrestling in regulated competition."
    chunk = Chunk(
        id="chunk",
        file_id="file",
        document_id="document",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=8,
        content_hash="hash",
    )
    window = ExtractionWindow(id="window", document_id="document", chunks=[chunk], token_count=8)
    entities = [
        _typed_mention("mma", "Mixed Martial Arts", "MARTIAL_ART", chunk.id),
        _typed_mention("wrestling", "wrestling", "TECHNIQUE", chunk.id),
    ]
    ontology = load_ontology(Path("data/martial_arts/ontology.yaml"), [], None)

    candidates = discover_cue_relation_candidates(window, entities, ontology.profile)

    uses = [candidate for candidate in candidates if candidate.relation_type == "USES_TECHNIQUE"]
    assert len(uses) == 1
    assert uses[0].source_entity_id == "mma"
    assert uses[0].target_entity_id == "wrestling"
    assert uses[0].verification_required is False


def test_cue_candidates_use_unique_document_context_surface_as_source() -> None:
    """Recover an explicit paper relation stated through a configured discourse phrase."""

    content = "In this paper, we address the degradation problem."
    chunk = Chunk(
        id="chunk",
        file_id="file",
        document_id="document",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=9,
        content_hash="hash",
    )
    window = ExtractionWindow(id="window", document_id="document", chunks=[chunk], token_count=9)
    paper = _typed_mention("paper", "Example Paper", "PAPER", chunk.id).model_copy(
        update={
            "is_document_context": True,
            "contextual_surfaces": ["this paper", "we"],
        }
    )
    task = _typed_mention("task", "degradation problem", "TASK", chunk.id)
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None)

    candidates = discover_cue_relation_candidates(window, [paper, task], ontology.profile)

    addresses = [candidate for candidate in candidates if candidate.relation_type == "ADDRESSES"]
    assert len(addresses) == 1
    assert addresses[0].source_entity_id == "paper"
    assert addresses[0].source_surface == "we"
    assert addresses[0].target_entity_id == "task"
    assert addresses[0].verification_required is False


def test_cue_candidates_enumerate_coordinated_targets() -> None:
    """Recover every dataset in a coordinated training statement without auto-accepting all."""

    content = "The model was trained on ImageNet and CIFAR-10."
    chunk = Chunk(
        id="chunk",
        file_id="file",
        document_id="document",
        page_number=1,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        token_count=9,
        content_hash="hash",
    )
    window = ExtractionWindow(id="window", document_id="document", chunks=[chunk], token_count=9)
    entities = [
        _typed_mention("model", "model", "MODEL", chunk.id),
        _typed_mention("imagenet", "ImageNet", "DATASET", chunk.id),
        _typed_mention("cifar", "CIFAR-10", "DATASET", chunk.id),
    ]
    ontology = load_ontology(Path("data/deep_learning_papers/ontology.yaml"), [], None)

    candidates = discover_cue_relation_candidates(window, entities, ontology.profile)

    trained_on = [candidate for candidate in candidates if candidate.relation_type == "TRAINED_ON"]
    endpoint_pairs = [
        (candidate.source_entity_id, candidate.target_entity_id) for candidate in trained_on
    ]
    assert endpoint_pairs == [
        ("model", "imagenet"),
        ("model", "cifar"),
    ]
    assert trained_on[0].verification_required is False
    assert trained_on[1].verification_required is True


def _mention(entity_id: str, name: str, chunk_id: str) -> EntityMention:
    """Build one event inventory mention for cue-discovery testing."""

    return _typed_mention(entity_id, name, "EVENT", chunk_id)


def _typed_mention(
    entity_id: str,
    name: str,
    entity_type: str,
    chunk_id: str,
) -> EntityMention:
    """Build one typed inventory mention for cue-discovery testing."""

    return EntityMention(
        id=entity_id,
        name=name,
        type=entity_type,
        description=name,
        source_chunk_id=chunk_id,
        quote=name,
    )
