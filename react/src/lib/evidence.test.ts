import { describe, expect, it } from "vitest";
import { documentNameIndex, evidenceDocumentId, evidenceEntityId, evidenceRelationId } from "./evidence";

describe("evidence rows", () => {
  it("reads the pipeline's subject columns and the gold format's explicit ones alike", () => {
    const node = { subject_id: "node_1", subject_kind: "node", file_id: "s3_file_1", quote: "q" };
    const edge = { subject_id: "edge_1", subject_kind: "edge", file_id: "s3_file_1" };
    const gold = { entity_id: "kano", relation_id: "r1", document_id: "d1" };
    expect([evidenceEntityId(node), evidenceRelationId(node), evidenceDocumentId(node)]).toEqual(["node_1", null, "s3_file_1"]);
    expect([evidenceEntityId(edge), evidenceRelationId(edge)]).toEqual([null, "edge_1"]);
    expect([evidenceEntityId(gold), evidenceRelationId(gold), evidenceDocumentId(gold)]).toEqual(["kano", "r1", "d1"]);
  });

  it("names documents by filename, then by the tail of their source", () => {
    const names = documentNameIndex([
      { id: "s3_file_1", file_id: "s3_file_1", source_uri: "s3://corpus/papers/Hinton-1981.pdf" },
      { id: "d2", filename: "judo.md", source_uri: "/tmp/x/judo.md" },
      { id: "d3" },
    ]);
    expect(names.get("s3_file_1")).toBe("Hinton-1981.pdf");
    expect(names.get("d2")).toBe("judo.md");
    expect(names.get("d3")).toBe("d3");
  });
});
