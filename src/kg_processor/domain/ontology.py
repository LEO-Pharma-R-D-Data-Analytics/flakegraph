# SPDX-License-Identifier: Apache-2.0
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
    # An entity of this type found by the document-context pass is what the
    # document itself is or describes - a paper's own title, the material a
    # certificate or data sheet covers - so a statement in that document may
    # have it as its implicit source. Requires document_context.
    document_subject: bool = False
    # Names of this type may be codes or numbers, such as a batch or lot number.
    # Where values have a home on relations (see RelationTypeDefinition), only
    # such types may have a name that is just a number or a value.
    identifier: bool = False
    contextual_surfaces: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def subject_requires_document_context(self) -> EntityTypeDefinition:
        """Only a type the document-context pass looks for can be a document's subject."""

        if self.document_subject and not self.document_context:
            raise ValueError(
                f"ontology entity {self.name} is a document subject without "
                "enabling document_context"
            )
        return self


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
    # The relation may carry the values it states - limits, ranges, results -
    # as quantities, so those values never become entities of their own.
    quantities: bool = False

    def admits(self, source_type: str, target_type: str) -> bool:
        """Return whether endpoint types satisfy this relation's signature.

        An empty source or target type list leaves that side unrestricted.
        """

        source_allowed = not self.source_types or source_type in self.source_types
        target_allowed = not self.target_types or target_type in self.target_types
        return source_allowed and target_allowed


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
    # A relation type that keeps a stated connection whose own type rules the
    # two entities break: "Granulator uses wet granulation" where the ontology
    # lets only products use a process. The edge takes this type, and the
    # label the model gave it is kept on its observation. Unset, such a
    # statement is rejected.
    fallback_relation: str | None = None

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

    @model_validator(mode="after")
    def fallback_relation_must_be_defined(self) -> OntologyProfile:
        """A fallback relation names one of the profile's own relation types."""

        if self.fallback_relation is not None and self.fallback_relation not in {
            relation.name for relation in self.relation_types
        }:
            raise ValueError(f"ontology fallback relation {self.fallback_relation} is not defined")
        return self

    def fallback_definition(self) -> RelationTypeDefinition | None:
        """The relation type a statement falls back to when its own rules reject it."""

        return next(
            (item for item in self.relation_types if item.name == self.fallback_relation), None
        )

    def carries_quantities(self) -> bool:
        """Whether any relation type gives stated values a home as quantities."""

        return any(definition.quantities for definition in self.relation_types)

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
