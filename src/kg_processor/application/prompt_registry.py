"""Versioned prompt loading and prompt fingerprint helpers.

Prompt text is treated like source code: every trace records the prompt revision
and checksum, and extraction cache keys include prompt fingerprints so prompt
edits cannot reuse stale graph output.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources

from kg_processor.application.extraction_schema import schema_hint
from kg_processor.domain.graph import Chunk, ExtractionResult
from kg_processor.ports.llm import CommunitySummaryRequest, DescriptionMergeRequest, LlmOptions

PROMPT_REVISIONS = {
    "graph_extraction": "graph_extraction@2026-07-03",
    "graph_gleaning": "graph_gleaning@2026-07-03",
    "graph_repair": "graph_repair@2026-07-03",
    "entity_description_merge": "entity_description_merge@2026-07-03",
    "community_report": "community_report@2026-07-03.2",
}
EXTRACTION_PROMPT_NAMES = ("graph_extraction", "graph_gleaning", "graph_repair")


@dataclass(frozen=True)
class PromptTemplate:
    """Loaded prompt text plus revision and checksum metadata."""

    name: str
    revision: str
    text: str
    checksum: str


@dataclass(frozen=True)
class GraphPrompt:
    """Provider-ready system/user prompt pair with template provenance."""

    system: str
    user: str
    template: PromptTemplate


def get_prompt_template(name: str) -> PromptTemplate:
    """Load a named prompt template and compute its checksum."""

    if name not in PROMPT_REVISIONS:
        raise ValueError(f"Unknown prompt template: {name}")
    text = resources.files("kg_processor.prompts").joinpath(f"{name}.md").read_text()
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return PromptTemplate(
        name=name,
        revision=PROMPT_REVISIONS[name],
        text=text.strip(),
        checksum=checksum,
    )


def prompt_fingerprints(names: tuple[str, ...]) -> dict[str, dict[str, str]]:
    """Return revision/checksum fingerprints for cache-key construction."""

    return {
        name: {
            "revision": template.revision,
            "checksum": template.checksum,
        }
        for name in names
        for template in [get_prompt_template(name)]
    }


def graph_extraction_prompt(
    chunks: list[Chunk],
    options: LlmOptions,
    pass_index: int = 0,
    previous_result: ExtractionResult | None = None,
) -> GraphPrompt:
    """Build the first-pass or gleaning graph extraction prompt."""

    template = get_prompt_template("graph_gleaning" if pass_index else "graph_extraction")
    valid_chunk_ids = [chunk.id for chunk in chunks]
    chunk_block = "\n\n".join(f"--- CHUNK id={chunk.id} ---\n{chunk.content}" for chunk in chunks)
    relation_type_text = (
        f"Use only these relation types: {', '.join(options.relation_types)}."
        if options.relation_types
        else "Use concise, normalized relation type labels."
    )
    previous_block = ""
    if previous_result is not None:
        previous_block = "\n\nPreviously accepted observations:\n" + json.dumps(
            {
                "entities": [
                    {
                        "name": entity.name,
                        "type": entity.type,
                        "source_chunk_id": entity.source_chunk_id,
                    }
                    for entity in previous_result.entities
                ],
                "relations": [
                    {
                        "source_name": relation.source_name,
                        "target_name": relation.target_name,
                        "relation_type": relation.relation_type,
                        "source_chunk_id": relation.source_chunk_id,
                    }
                    for relation in previous_result.relations
                ],
            },
            sort_keys=True,
        )
    user = (
        f"Extraction pass: {pass_index}\n"
        f"Use only these entity types: {', '.join(options.entity_types)}.\n"
        f"{relation_type_text}\n"
        f"Return at most {options.max_entities_per_batch} entities and "
        f"{options.max_relations_per_batch} relations for this batch.\n"
        f"Do not return entities below confidence {options.min_entity_confidence:.2f}; "
        f"do not return relations below confidence {options.min_relation_confidence:.2f}.\n"
        f"Valid source_chunk_id values are: {', '.join(valid_chunk_ids)}.\n"
        "JSON shape example:\n"
        f"{json.dumps(schema_hint(chunks, options), sort_keys=True)}"
        f"{previous_block}\n\n"
        f"{chunk_block}"
    )
    return GraphPrompt(system=template.text, user=user, template=template)


def graph_repair_prompt(
    chunks: list[Chunk],
    options: LlmOptions,
    invalid_response: str,
    validation_error: str,
) -> GraphPrompt:
    """Build a repair prompt for invalid or semantically ungrounded extraction."""

    template = get_prompt_template("graph_repair")
    payload = {
        "extraction_pass": options.extraction_pass,
        "entity_types": options.entity_types,
        "relation_types": options.relation_types,
        "max_entities_per_batch": options.max_entities_per_batch,
        "max_relations_per_batch": options.max_relations_per_batch,
        "valid_source_chunk_ids": [chunk.id for chunk in chunks],
        "json_shape_example": schema_hint(chunks, options),
        "validation_error": validation_error,
        "invalid_response": invalid_response,
        "chunks": [
            {
                "id": chunk.id,
                "content": chunk.content,
            }
            for chunk in chunks
        ],
    }
    return GraphPrompt(
        system=template.text,
        user=json.dumps(payload, sort_keys=True),
        template=template,
    )


def community_report_prompt(request: CommunitySummaryRequest) -> GraphPrompt:
    """Build the prompt that turns a detected community into report rows."""

    template = get_prompt_template("community_report")
    user = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    return GraphPrompt(system=template.text, user=user, template=template)


def entity_description_merge_prompt(request: DescriptionMergeRequest) -> GraphPrompt:
    """Build the prompt that synthesizes repeated entity descriptions."""

    template = get_prompt_template("entity_description_merge")
    user = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    return GraphPrompt(system=template.text, user=user, template=template)


def prompt_metadata(prompt: GraphPrompt) -> dict[str, str]:
    """Return prompt metadata persisted in provider traces."""

    return {
        "prompt_name": prompt.template.name,
        "prompt_revision": prompt.template.revision,
        "prompt_checksum": prompt.template.checksum,
    }
