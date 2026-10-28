import { describe, expect, it } from "vitest";
import { scopeDataset } from "./scope";
import { createAskTools } from "./tools";
import type { GraphDataset } from "../protocol/schema";

const dataset: GraphDataset = {
  graphId: "g", nodes: [
    { id: "kept", name: "Albatross", source_chunk_ids: ["chunk-kept"] },
    { id: "excluded", name: "Zirconquasar", source_chunk_ids: ["chunk-excluded"] },
  ], edges: [], communities: [],
  documents: [{ id: "doc-kept" }, { id: "doc-excluded" }],
  chunks: [
    { id: "chunk-kept", file_id: "doc-kept", content: "Albatross is in this document." },
    { id: "chunk-excluded", file_id: "doc-excluded", content: "Zirconquasar is outside the perspective." },
  ],
  evidence: [
    { id: "ev-kept", subject_kind: "node", subject_id: "kept", chunk_id: "chunk-kept", file_id: "doc-kept", quote: "Albatross" },
    { id: "ev-excluded", subject_kind: "node", subject_id: "excluded", chunk_id: "chunk-excluded", file_id: "doc-excluded", quote: "Zirconquasar" },
  ], runReport: {}, graphMetrics: {}, discardedWindows: [], rejectedRecords: [], failedDocuments: [],
};

describe("Ask retrieval boundaries", () => {
  it("follows canonical chunk provenance when applying a perspective", () => {
    const scoped = scopeDataset(dataset, { search: "Albatross" });
    expect(scoped.nodes.map((n) => n.id)).toEqual(["kept"]);
    expect(scoped.chunks.map((c) => c.id)).toEqual(["chunk-kept"]);
    expect(scoped.evidence.map((e) => e.id)).toEqual(["ev-kept"]);
    expect(scoped.documents.map((d) => d.id)).toEqual(["doc-kept"]);
  });

  it.each([{ documentIds: ["doc-kept"] }, { search: "Albatross" }])("limits every direct tool to the selected scope %j", async (scope) => {
    const tools = createAskTools({ dataset, scope });
    const options = { toolCallId: "fixture", messages: [], context: {} };
    const documents = await tools.listDocuments.execute!({}, options);
    const excluded = await tools.readDocumentEvidence.execute!({ documentId: "doc-excluded", startIndex: 0, count: 10 }, options);
    const entities = await tools.searchEntities.execute!({ query: "Zirconquasar" }, options);
    const kept = await tools.readDocumentEvidence.execute!({ documentId: "doc-kept", startIndex: 0, count: 10 }, options);
    expect(JSON.stringify(documents)).not.toContain("doc-excluded");
    expect(JSON.stringify(excluded)).not.toContain("Zirconquasar");
    expect(JSON.stringify(entities)).not.toContain('"id":"excluded"');
    expect(JSON.stringify(kept)).toContain("Albatross");
  });
});
