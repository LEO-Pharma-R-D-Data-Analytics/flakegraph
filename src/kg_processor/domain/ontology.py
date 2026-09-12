"""Provider-neutral ontology contracts for typed graph extraction.

Ontology profiles describe what an entity or relation means without coupling
that definition to a particular LLM, OCR engine, or graph writer.  The same
profile is therefore enforced across local, hosted, and Snowflake providers.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EntityTypeDefinition(BaseModel):
    """Define one allowed entity category and language explaining its scope.

    Aliases and examples guide models while the canonical name remains stable for
    validation and persisted graph typing.
    """

    name: str
    description: str
    aliases: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    document_context: bool = False
    contextual_surfaces: list[str] = Field(default_factory=list)


class RelationTypeDefinition(BaseModel):
    """Define one directed predicate and all structural/evidence constraints.

    Domain/range types, inverses, symmetry, self-loop policy, aliases, examples,
    and grounding cues form a reviewable contract independent from any model.
    """

    name: str
    description: str
    source_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    inverse: str | None = None
    symmetric: bool = False
    allow_self_loop: bool = False
    examples: list[str] = Field(default_factory=list)
    evidence_cues: list[str] = Field(default_factory=list)


class OntologyProfile(BaseModel):
    """Represent a complete, reviewable extraction vocabulary and enforcement mode.

    The same profile drives prompt schemas, record validation, graph quality, and
    run provenance across every provider implementation.
    """

    name: str
    description: str
    mode: Literal["closed", "open", "hybrid"] = "hybrid"
    entity_types: list[EntityTypeDefinition]
    relation_types: list[RelationTypeDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def definitions_must_be_unique_and_referentially_valid(self) -> OntologyProfile:
        """Reject ambiguous names, broken references, and invalid cues before extraction.

        Early profile validation prevents expensive provider calls under an ontology
        that could never produce internally consistent graph rows.
        """

        entity_names = [definition.name for definition in self.entity_types]
        relation_names = [definition.name for definition in self.relation_types]
        normalized_entity_names = [normalize_ontology_label(name) for name in entity_names]
        normalized_relation_labels = [
            normalize_ontology_label(label)
            for definition in self.relation_types
            for label in [definition.name, *definition.aliases]
        ]
        if len(set(normalized_entity_names)) != len(normalized_entity_names):
            raise ValueError("ontology entity type names must be unique")
        if len(set(normalized_relation_labels)) != len(normalized_relation_labels):
            raise ValueError(
                "ontology relation names and aliases must be unique after normalization"
            )
        known_entities = set(entity_names)
        known_relations = set(relation_names)
        contextual_surface_owners: dict[str, str] = {}
        for entity in self.entity_types:
            if entity.contextual_surfaces and not entity.document_context:
                raise ValueError(
                    f"ontology entity {entity.name} declares contextual surfaces without "
                    "enabling document_context"
                )
            if any(not surface.strip() for surface in entity.contextual_surfaces):
                raise ValueError(
                    f"ontology entity {entity.name} contains a blank contextual surface"
                )
            for surface in entity.contextual_surfaces:
                normalized_surface = " ".join(surface.casefold().split())
                owner = contextual_surface_owners.setdefault(normalized_surface, entity.name)
                if owner != entity.name:
                    raise ValueError(
                        f"ontology contextual surface {surface!r} is shared by document-context "
                        f"entity types {owner} and {entity.name}"
                    )
        for relation in self.relation_types:
            unknown_types = (
                set(relation.source_types) | set(relation.target_types)
            ) - known_entities
            if unknown_types:
                raise ValueError(
                    f"ontology relation {relation.name} refers to unknown entity types: "
                    f"{', '.join(sorted(unknown_types))}"
                )
            if relation.inverse and relation.inverse not in known_relations:
                raise ValueError(
                    f"ontology relation {relation.name} has unknown inverse {relation.inverse}"
                )
            for cue in relation.evidence_cues:
                try:
                    re.compile(cue, flags=re.IGNORECASE)
                except re.error as exc:
                    raise ValueError(
                        f"ontology relation {relation.name} has invalid evidence cue: {exc}"
                    ) from exc
        return self

    def entity_type_names(self) -> list[str]:
        """Return canonical entity labels in their original configured prompt and schema order."""

        return [definition.name for definition in self.entity_types]

    def relation_type_names(self) -> list[str]:
        """Return canonical relation labels in their original configured prompt and schema order."""

        return [definition.name for definition in self.relation_types]

    def relation(self, name: str) -> RelationTypeDefinition | None:
        """Resolve a relation by normalized canonical name or configured alias.

        Returning the canonical definition centralizes direction, type, loop, and
        evidence policy for extraction and quality checks.
        """

        normalized = normalize_ontology_label(name)
        for definition in self.relation_types:
            labels = [definition.name, *definition.aliases]
            if normalized in {normalize_ontology_label(label) for label in labels}:
                return definition
        return None


def normalize_ontology_label(value: str) -> str:
    """Normalize ontology labels across case, hyphen, and whitespace variations.

    This is the single relation-label identity rule used by ontology lookup,
    canonical edge assembly, and graph quality checks. Keeping it public avoids
    subtle differences where one stage treats ``works-at`` and ``works at`` as
    equal while another persists them as separate predicates.
    """

    return "_".join(value.casefold().replace("-", " ").split())
