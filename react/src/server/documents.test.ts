import { describe, expect, it } from "vitest";
import { documentStatusesFromTasks, withSkippedFiles } from "./documents";

const source = (filename: string) => ({ input_file: { filename, source_uri: `s3://corpus/${filename}` } });

function task(stage: string, scopeId: string, status: string, extra: Partial<{ attempts: number; lastError: unknown; sourceMetadata: unknown }> = {}) {
  return { stage, scopeId, status, attempts: 1, lastError: null, sourceMetadata: null, ...extra };
}

describe("documentStatusesFromTasks", () => {
  it("names each document and reads its phase from the durable stages", () => {
    const statuses = documentStatusesFromTasks([
      task("prepare_document", "f1", "succeeded", { sourceMetadata: source("b.pdf") }),
      task("extract_document_context", "f1", "succeeded"),
      task("compact_document", "f1", "succeeded"),
      task("prepare_document", "f2", "succeeded", { sourceMetadata: source("a.pdf") }),
      task("extract_document_context", "f2", "running"),
      task("prepare_document", "f3", "failed", {
        attempts: 3,
        lastError: { error_type: "RuntimeError", error_message: "mineru_api returned HTTP 400" },
      }),
      task("prepare_document", "f4", "succeeded", { sourceMetadata: source("d.pdf") }),
      task("extract_document_context", "f4", "queued"),
      task("prepare_document", "f5", "queued"),
      task("prepare_document", "f6", "succeeded", { sourceMetadata: source("c.pdf") }),
      task("compact_document", "f6", "failed", { attempts: 2, lastError: { message: "no windows" } }),
      task("prepare_document", "f7", "cancelled"),
    ]);
    expect(statuses.map((item) => [item.name ?? item.fileId, item.phase])).toEqual([
      ["a.pdf", "running"],
      ["b.pdf", "indexed"],
      ["c.pdf", "poison"],
      ["d.pdf", "scanned"],
      ["f3", "ocr-failed"],
      ["f5", "queued"],
      ["f7", "cancelled"],
    ]);
    expect(statuses.find((item) => item.fileId === "f3")?.detail).toBe("RuntimeError: mineru_api returned HTTP 400");
    expect(statuses.find((item) => item.fileId === "f6")?.detail).toBe("no windows (attempt 2)");
    expect(statuses.find((item) => item.fileId === "f2")?.detail).toBe("Extract document context");
  });

  it("marks quarantined files whatever the runtime reported", () => {
    const statuses = withSkippedFiles(
      [
        { fileId: "f1", phase: "poison", detail: "bad" },
        { fileId: "f2", phase: "indexed", detail: "Indexed" },
      ],
      ["f1"],
    );
    expect(statuses.map((item) => item.phase)).toEqual(["skipped", "indexed"]);
  });
});
