"""Replaceable ports for entity, relation, and semantic verification stages."""

from __future__ import annotations

from typing import Protocol

from kg_processor.domain.extraction import (
    EntityExtractionOutcome,
    EntityMention,
    ExtractionWindow,
    RelationExtractionOutcome,
    RelationObservation,
    VerificationOutcome,
)
from kg_processor.domain.ontology import OntologyProfile


class EntityExtractor(Protocol):
    """Define the replaceable port for grounded entity mention recognition.

    Implementations may use an LLM or local model but must return exact mention
    evidence and provider diagnostics in the shared domain contract.
    """

    def extract(
        self,
        window: ExtractionWindow,
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_entities: int,
        previous_entities: list[EntityMention] | None = None,
    ) -> EntityExtractionOutcome:
        """Return exact mentions from one document-scoped extraction window.

        Previous mentions are supplied only for targeted gleaning and deduplication.
        """
        ...


class RelationExtractor(Protocol):
    """Define the replaceable port for relations between accepted mention IDs.

    Implementations must honor ontology constraints and return grounded observation
    records rather than unconstrained endpoint names.
    """

    def extract(
        self,
        window: ExtractionWindow,
        entities: list[EntityMention],
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
        max_relations: int,
        previous_relations: list[RelationObservation] | None = None,
    ) -> RelationExtractionOutcome:
        """Return grounded observations that reference supplied entity mention IDs.

        Previous observations support targeted relation gleaning without duplication.
        """
        ...


class RelationVerifier(Protocol):
    """Define an independent semantic entailment verifier for candidate triples.

    Verification cannot create relations; it may only judge supplied observation IDs.
    """

    def verify(
        self,
        window: ExtractionWindow,
        entities: list[EntityMention],
        relations: list[RelationObservation],
        ontology: OntologyProfile,
        *,
        model: str,
        timeout_seconds: int,
    ) -> VerificationOutcome:
        """Return one evidence-based verdict for every supplied relation observation.

        Missing or malformed provider decisions are reconciled conservatively by callers.
        """
        ...
