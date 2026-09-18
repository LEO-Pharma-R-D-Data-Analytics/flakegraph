import { mkdtemp, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { decodeSchema, IngestionRequest } from "./protocol/schema";
import { redactedConfig, sensitiveConfigKey } from "./config";
import { accumulateSummary, makeProgressRecord, progressEvent, readLocalProgress } from "./progress";
import { filterGraph, graphFacets } from "./graph-filter";
import { goldToDataset, compareToGold } from "./gold";
import { administrableGraphPredicate, mayReadStagePath, viewerStagePrefix, visibleGraphPredicate } from "./identity";
import { writeRunRecord, renameGraph, listRunRecords, hideRun } from "./catalog";
import { capabilitiesFor, validateGraphName } from "./protocol/schema";

describe("protocol schemas", () => {
  it("decodes an ingestion request", () => {
    const request = decodeSchema(IngestionRequest, {
      runtime: "local",
      jobId: "job_1",
      graphId: "graph_1",
      graphName: "Demo",
      sourceKind: "local_path",
      source: { path: "data/martial_arts/files" },
      ocr: { provider: "fallback", model: null, endpoint: null, apiKeyEnvironmentVariable: null, dimension: null },
      llm: { provider: "vllm_local", model: "qwen", endpoint: "http://localhost:8000/v1", apiKeyEnvironmentVariable: null, dimension: null },
      embedding: { provider: "sentence_transformers", model: "mini", endpoint: null, apiKeyEnvironmentVariable: null, dimension: 384 },
      output: { kind: "local_files", workspacePath: "/tmp/out", snowflake: null },
      baseConfigPath: null,
    });
    expect(request.providerParallelism).toBe(12);
    expect(request.includeGlobs).toEqual(["**/*"]);
  });

  it("rejects an empty graph name", () => {
    expect(() => validateGraphName("   ")).toThrow(/empty/);
  });

  it("advertises runtime capabilities", () => {
    expect(capabilitiesFor("local")).toContain("upload");
    expect(capabilitiesFor("local")).toContain("cancel");
    expect(capabilitiesFor("kubernetes")).toContain("cluster");
    expect(capabilitiesFor("snowflake")).toContain("share");
    expect(capabilitiesFor("local")).not.toContain("share");
  });
});

describe("configuration redaction", () => {
  it("redacts secrets and credential URLs", () => {
    const redacted = redactedConfig({
      llm: { api_key: "sk-live", endpoint: "https://user:pass@example.com/v1?api_key=secret" },
      token_environment_variable: "LLM_API_KEY",
    });
    const llm = redacted.llm as Record<string, string>;
    expect(llm.api_key).toBe("${REDACTED}");
    expect(llm.endpoint).toContain("***");
    expect(redacted.token_environment_variable).toBe("LLM_API_KEY");
    expect(sensitiveConfigKey("api_key")).toBe(true);
    expect(sensitiveConfigKey("api_key_environment_variable")).toBe(false);
  });
});

describe("progress aggregation", () => {
  it("counts distinct completed files and ignores retries", async () => {
    const directory = await mkdtemp(path.join(tmpdir(), "fg-progress-"));
    const file = path.join(directory, "events.jsonl");
    const lines = [
      makeProgressRecord({ timestamp: "t1", stage: "ocr", status: "completed", fileId: "a", message: null, elapsedMs: 10, counts: { files_total: 2 } }),
      makeProgressRecord({ timestamp: "t2", stage: "ocr", status: "completed", fileId: "a", message: null, elapsedMs: 12, counts: { files_total: 2 } }),
      makeProgressRecord({ timestamp: "t3", stage: "ocr", status: "completed", fileId: "b", message: null, elapsedMs: 8, counts: { files_total: 2 } }),
    ];
    await writeFile(file, lines.map((line) => JSON.stringify(line)).join("\n") + "\n");
    const progress = await readLocalProgress(file);
    const ocr = progress.stages.find((stage) => stage.stage === "ocr");
    expect(ocr?.completed).toBe(2);
    expect(progress.documentsCompleted).toBe(2);
  });

  it("parses core progress records", () => {
    const event = progressEvent({ event: "kg_processor.progress", stage: "ocr", status: "running", file_id: "doc" });
    expect(event.fileId).toBe("doc");
    const summary: Record<string, unknown> = {};
    accumulateSummary(summary, { event: "kg_processor.progress", stage: "file_source", status: "completed", counts: { files_seen: 4 } });
    expect(summary.documents_total).toBe(4);
  });
});

describe("graph filters", () => {
  const dataset = goldToDataset(
    {
      entities: [
        { id: "judo", name: "Judo", type: "ART", aliases: ["Kodokan judo"] },
        { id: "kano", name: "Jigoro Kano", type: "PERSON" },
        { id: "tokyo", name: "Tokyo", type: "PLACE" },
      ],
      relations: [
        { id: "r1", source: "judo", target: "kano", relation_type: "DEVELOPED_BY", observations: [{ document: "d1", sentence: "Kano developed judo" }] },
        { id: "r2", source: "judo", target: "tokyo", relation_type: "PRACTICED_IN", required: false },
      ],
    },
    "g1",
  );

  it("filters by search, type, relation, and confidence", () => {
    expect(graphFacets(dataset).nodeTypes).toContain("PERSON");
    const people = filterGraph(dataset, { nodeTypes: ["PERSON"] });
    expect(people.nodes.map((node) => node.id)).toEqual(["kano"]);
    const developed = filterGraph(dataset, { relationTypes: ["DEVELOPED_BY"], includeIsolates: false });
    expect(developed.edges).toHaveLength(1);
    const strict = filterGraph(dataset, { minimumConfidence: 0.9 });
    expect(strict.edges.every((edge) => Number(edge.confidence) >= 0.9)).toBe(true);
    const search = filterGraph(dataset, { search: "kodokan" });
    expect(search.nodes.map((node) => node.id)).toEqual(["judo"]);
    const connected = filterGraph(dataset, { includeIsolates: false });
    expect(connected.nodes.every((node) => ["judo", "kano", "tokyo"].includes(String(node.id)))).toBe(true);
  });
});

describe("identity", () => {
  it("hides owned graphs from unidentified viewers", () => {
    const unidentified = visibleGraphPredicate({ userName: "", email: "", roles: [] });
    expect(unidentified.sql).toContain("OWNER IS NULL");
    const owner = administrableGraphPredicate({ userName: "ALICE", email: "", roles: [] });
    expect(owner.parameters).toEqual(["ALICE"]);
    expect(viewerStagePrefix({ userName: "Alice.Smith", email: "", roles: [] }, "job")).toBe("app/Alice.Smith/job");
    expect(mayReadStagePath({ userName: "ALICE", email: "", roles: [] }, "app/ALICE/job/file.pdf")).toBe(true);
    expect(mayReadStagePath({ userName: "ALICE", email: "", roles: [] }, "app/BOB/job/file.pdf")).toBe(false);
  });
});

describe("catalog", () => {
  it("persists run records and graph names", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-catalog-"));
    const directory = path.join(stateRoot, "runs", "run_1");
    await mkdir(directory, { recursive: true });
    await writeRunRecord(directory, {
      runId: "run_1",
      graphId: "graph_1",
      status: "succeeded",
      runtime: "local",
    });
    const name = await renameGraph(stateRoot, "graph_1", "  Martial arts  ");
    expect(name).toBe("Martial arts");
    const listed = await listRunRecords(stateRoot, 10);
    expect(listed[0]?.record.runId).toBe("run_1");
    await hideRun(stateRoot, "run_1");
    expect(await listRunRecords(stateRoot, 10)).toHaveLength(0);
  });
});

describe("gold comparison", () => {
  it("treats gold-derived datasets as fully matched", () => {
    const gold = {
      name: "demo",
      entities: [
        { id: "e1", name: "Judo", type: "MARTIAL_ART" },
        { id: "e2", name: "Kano", type: "PERSON" },
      ],
      relations: [{ id: "r1", source: "e1", target: "e2", relation_type: "DEVELOPED_BY", required: true }],
    };
    const dataset = goldToDataset(gold, "graph_demo");
    const comparison = compareToGold(dataset, gold);
    expect(comparison.missingRequired).toEqual([]);
    expect(comparison.matchedRequired).toBe(1);
  });
});
