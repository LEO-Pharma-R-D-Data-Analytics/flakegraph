import path from "node:path";
import { describe, expect, it } from "vitest";
import { edgeStatedTypes, loadLocalGraph } from "./graph";

const FIXTURE = path.join(import.meta.dirname, "__fixtures__", "graph");

describe("loadLocalGraph", () => {
  it("reads parquet artifacts without their vector columns", async () => {
    const dataset = await loadLocalGraph(FIXTURE);
    expect(dataset.graphId).toBe("fixture-graph");
    expect(dataset.nodes.map((node) => node.id)).toEqual(["n1", "n2", "n3", "n4"]);
    expect(dataset.nodes.every((node) => !("embedding" in node))).toBe(true);
    expect(dataset.nodes[0]).toMatchObject({ name: "Judo", primary_type: "ART" });
    expect(dataset.edges).toHaveLength(3);
    expect(dataset.edges.every((edge) => !("description_embedding" in edge))).toBe(true);
    expect(dataset.edges[0]).toMatchObject({ source_node_id: "n1", target_node_id: "n2", confidence: 0.95 });
    expect(dataset.runReport.app_full_counts).toMatchObject({ node: 4, edge: 3, community: 0, evidence: 0 });
  });
});

describe("edgeStatedTypes", () => {
  it("keeps each type the model stated for an edge a fallback relation kept, once", () => {
    const stated = edgeStatedTypes([
      { edge_id: "e1", proposed_relation_type: "USES_EQUIPMENT" },
      { edge_id: "e1", proposed_relation_type: "USES_EQUIPMENT" },
      { edge_id: "e1", proposed_relation_type: "USES_PROCESS" },
      { edge_id: "e2", proposed_relation_type: null },
    ]);
    expect(stated.get("e1")).toEqual(["USES_EQUIPMENT", "USES_PROCESS"]);
    expect(stated.has("e2")).toBe(false);
  });
});
