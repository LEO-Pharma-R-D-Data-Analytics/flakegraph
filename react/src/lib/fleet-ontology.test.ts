import { describe, expect, it } from "vitest";
import { fleetOntologyTerms } from "./fleet-ontology";

describe("fleetOntologyTerms", () => {
  it("reads the profile's named types with their descriptions", () => {
    const terms = fleetOntologyTerms({
      config: { graph: { entity_types: ["PERSON"] } },
      ontology: {
        entity_types: [{ name: "PERSON", description: "A named human being." }, { name: "WORK" }],
        relation_types: [{ name: "CREATED_BY", description: "Made by." }],
      },
    });
    expect(terms.entityTypes).toEqual([
      { name: "PERSON", description: "A named human being." },
      { name: "WORK", description: "" },
    ]);
    expect(terms.relationTypes).toEqual([{ name: "CREATED_BY", description: "Made by." }]);
  });

  it("falls back to the graph section's plain type names without an ontology", () => {
    const terms = fleetOntologyTerms({ config: { graph: { entity_types: ["PERSON", "EVENT"] } }, ontology: null });
    expect(terms.entityTypes.map((term) => term.name)).toEqual(["PERSON", "EVENT"]);
    expect(terms.relationTypes).toEqual([]);
  });
});
