export interface OntologyTerm {
  name: string;
  description: string;
}

/**
 * The entity and relation types a fleet's ontology profile names, as the
 * form shows them: a fleet's workers extract exactly these, so a request
 * cannot choose its own.
 */
export function fleetOntologyTerms(profile: { config: Record<string, unknown>; ontology: Record<string, unknown> | null }): { entityTypes: OntologyTerm[]; relationTypes: OntologyTerm[] } {
  const ontology = profile.ontology ?? {};
  const terms = (value: unknown): OntologyTerm[] =>
    Array.isArray(value)
      ? value
          .map((item) => {
            if (typeof item === "string") {
              return { name: item, description: "" };
            }
            if (item && typeof item === "object") {
              const record = item as Record<string, unknown>;
              return { name: String(record.name ?? ""), description: String(record.description ?? "") };
            }
            return { name: "", description: "" };
          })
          .filter((item) => item.name)
      : [];
  const graph = (profile.config.graph ?? {}) as Record<string, unknown>;
  const entityTypes = terms(ontology.entity_types);
  return {
    entityTypes: entityTypes.length ? entityTypes : terms(graph.entity_types),
    relationTypes: terms(ontology.relation_types),
  };
}
