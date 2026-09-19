import { randomBytes } from "node:crypto";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { goldToDataset, writeDataset, type GoldGraph } from "../src/server/gold";
import { writeRunRecord } from "../src/server/catalog";
import { makeProgressRecord } from "../src/server/progress";
import { atomicWriteJson } from "../src/server/catalog";
import { loadEnv } from "../src/server/env";
import { hashApiSecret } from "../src/server/api-keys";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

async function main() {
  const env = loadEnv({
    ...process.env,
    FLAKEGRAPH_REPOSITORY_ROOT: repoRoot,
    FLAKEGRAPH_APP_STATE_ROOT: process.env.FLAKEGRAPH_APP_STATE_ROOT,
  });
  const stateRoot = env.stateRoot;
  await rm(stateRoot, { recursive: true, force: true });
  await mkdir(stateRoot, { recursive: true });

  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/martial_arts/gold.json"),
    runId: "run_martial_arts",
    graphId: "graph_martial_arts",
    graphName: "Martial arts history",
    status: "succeeded",
  });
  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/deep_learning_papers/gold.json"),
    runId: "run_deep_learning",
    graphId: "graph_deep_learning",
    graphName: "Deep learning papers",
    status: "succeeded",
  });
  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/martial_arts/gold.json"),
    runId: "run_rename_target",
    graphId: "graph_rename_target",
    graphName: "Rename target",
    status: "succeeded",
  });
  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/martial_arts/gold.json"),
    runId: "run_forget_me",
    graphId: "graph_forget_me",
    graphName: "Forgettable graph",
    status: "succeeded",
  });
  await seedGoldDataset({
    stateRoot,
    goldPath: path.join(repoRoot, "data/martial_arts/gold.json"),
    runId: "run_bulk_forget",
    graphId: "graph_bulk_forget",
    graphName: "Bulk forget me",
    status: "succeeded",
  });
  await seedThinGoldDataset(stateRoot);
  await seedActiveRun(stateRoot);
  await seedUnavailableRun(stateRoot);
  await seedFailedRun(stateRoot);
  await seedKubernetes(stateRoot);
  await seedSnowflake(stateRoot);
  await seedWorkspace(stateRoot);
  await seedCancellingRun(stateRoot);
  await seedLargeCorpusRun(stateRoot);
  await seedPiiPack(stateRoot);
  console.log(`Seeded control-plane fixtures under ${stateRoot}`);
}

async function seedGoldDataset(options: {
  stateRoot: string;
  goldPath: string;
  runId: string;
  graphId: string;
  graphName: string;
  status: string;
  owner?: string | null;
}) {
  const gold = JSON.parse(await readFile(options.goldPath, "utf8")) as GoldGraph;
  const outputPath = path.join(options.stateRoot, "graphs", options.graphId);
  await writeDataset(outputPath, goldToDataset(gold, options.graphId));
  const directory = path.join(options.stateRoot, "runs", options.runId);
  await writeRunRecord(directory, {
    runId: options.runId,
    graphId: options.graphId,
    graphName: options.graphName,
    status: options.status,
    runtime: "local",
    startedAt: "2026-09-01T10:00:00.000Z",
    updatedAt: "2026-09-01T12:00:00.000Z",
    outputPath,
    storageKind: "local_files",
    storageLocation: outputPath,
    documentsTotal: gold.documents?.length ?? 0,
    documentsCompleted: gold.documents?.length ?? 0,
    documentsFailed: 0,
    nodeCount: gold.entities?.length ?? 0,
    owner: options.owner ?? "ALICE",
    sourceKind: "local_path",
    sourcePath: options.goldPath.includes("deep_learning")
      ? "data/deep_learning_papers/files"
      : "data/martial_arts/files",
    ocrProvider: "builtin_text",
    llmProvider: "ollama",
    embeddingProvider: "sentence_transformers",
  });
  await persistName(options.stateRoot, options.graphId, options.graphName);
}

/**
 * A martial-arts graph that found three entities and little else. Quality
 * compares it to the full gold file, so nearly every required relation is
 * missing: the list the Quality tab has to page.
 */
async function seedThinGoldDataset(stateRoot: string) {
  const goldPath = path.join(repoRoot, "data/martial_arts/gold.json");
  const gold = JSON.parse(await readFile(goldPath, "utf8")) as GoldGraph;
  const kept = new Set((gold.entities ?? []).slice(0, 3).map((entity) => entity.id));
  const thin: GoldGraph = {
    ...gold,
    entities: (gold.entities ?? []).filter((entity) => kept.has(entity.id)),
    relations: (gold.relations ?? []).filter((relation) => kept.has(relation.source) && kept.has(relation.target)),
  };
  const graphId = "graph_martial_thin";
  const outputPath = path.join(stateRoot, "graphs", graphId);
  await writeDataset(outputPath, goldToDataset(thin, graphId));
  await writeRunRecord(path.join(stateRoot, "runs", "run_martial_thin"), {
    runId: "run_martial_thin",
    graphId,
    graphName: "Thin martial arts extract",
    status: "succeeded",
    runtime: "local",
    startedAt: "2026-08-20T10:00:00.000Z",
    updatedAt: "2026-08-20T10:30:00.000Z",
    outputPath,
    storageKind: "local_files",
    storageLocation: outputPath,
    documentsTotal: gold.documents?.length ?? 0,
    documentsCompleted: gold.documents?.length ?? 0,
    documentsFailed: 0,
    nodeCount: thin.entities?.length ?? 0,
    owner: "ALICE",
    sourceKind: "local_path",
    sourcePath: "data/martial_arts/files",
    ocrProvider: "builtin_text",
    llmProvider: "ollama",
    embeddingProvider: "sentence_transformers",
  });
  await persistName(stateRoot, graphId, "Thin martial arts extract");
  // Enough progress events that the run's details show a tail of the log,
  // not all of it.
  const events = Array.from({ length: THIN_RUN_PAGES }, (_, index) => `page-${String(index + 1).padStart(2, "0")}.md`).flatMap(
    (fileId, index) =>
      ["ocr", "extract"].map((stage, offset) =>
        makeProgressRecord({
          timestamp: `2026-08-20T10:${String(index).padStart(2, "0")}:${offset ? "30" : "00"}.000Z`,
          stage,
          status: "completed",
          fileId,
          message: null,
          elapsedMs: 100,
          counts: {},
        }),
      ),
  );
  await writeFile(
    path.join(stateRoot, "runs", "run_martial_thin", "events.jsonl"),
    events.map((event) => JSON.stringify(event)).join("\n") + "\n",
    "utf8",
  );
}

/** Documents the thin run logged: two events each, more than one page of the events card. */
const THIN_RUN_PAGES = 20;

async function seedActiveRun(stateRoot: string) {
  const runId = "run_active_ocr";
  const directory = path.join(stateRoot, "runs", runId);
  const outputPath = path.join(stateRoot, "graphs", "graph_active");
  await mkdir(outputPath, { recursive: true });
  await writeRunRecord(directory, {
    runId,
    graphId: "graph_active",
    graphName: "Active OCR run",
    status: "running",
    runtime: "local",
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    outputPath,
    storageKind: "local_files",
    storageLocation: outputPath,
    documentsTotal: 10,
    documentsCompleted: 3,
    documentsFailed: 0,
    owner: "ALICE",
  });
  const events = [
    makeProgressRecord({
      timestamp: new Date().toISOString(),
      stage: "file_source",
      status: "completed",
      fileId: null,
      message: "Discovered 10 files",
      elapsedMs: 120,
      counts: { files_seen: 10 },
    }),
    makeProgressRecord({
      timestamp: new Date().toISOString(),
      stage: "ocr",
      status: "failed",
      fileId: "doc_poison_scan",
      message: "Poison PDF quarantined",
      elapsedMs: 900,
      counts: { files_total: 10 },
    }),
  ];
  await writeFile(
    path.join(directory, "events.jsonl"),
    events.map((event) => JSON.stringify(event)).join("\n") + "\n",
    "utf8",
  );
}

async function seedUnavailableRun(stateRoot: string) {
  await writeRunRecord(path.join(stateRoot, "runs", "run_missing_artifacts"), {
    runId: "run_missing_artifacts",
    graphId: "graph_copied_history",
    graphName: "Copied host graph",
    status: "succeeded",
    runtime: "local",
    startedAt: "2026-08-01T00:00:00.000Z",
    updatedAt: "2026-08-01T01:00:00.000Z",
    outputPath: "/not/on/this/machine/graph",
    storageKind: "local_files",
    storageLocation: "/not/on/this/machine/graph",
    owner: null,
  });
}

async function seedFailedRun(stateRoot: string) {
  const directory = path.join(stateRoot, "runs", "run_failed_llm");
  await writeRunRecord(directory, {
    runId: "run_failed_llm",
    graphId: "graph_failed_llm",
    graphName: "LLM timeout",
    status: "failed",
    runtime: "local",
    startedAt: "2026-09-02T00:00:00.000Z",
    updatedAt: "2026-09-02T00:05:00.000Z",
    outputPath: path.join(stateRoot, "graphs", "graph_failed_llm"),
    storageKind: "local_files",
    error: "LLM endpoint timed out",
    sourceKind: "local_path",
    sourcePath: "data/martial_arts/files",
    ocrProvider: "builtin_text",
    llmProvider: "ollama",
    embeddingProvider: "sentence_transformers",
    owner: "ALICE",
  });
}

async function seedKubernetes(stateRoot: string) {
  const directory = path.join(stateRoot, "kubernetes");
  await mkdir(directory, { recursive: true });
  await atomicWriteJson(path.join(directory, "cluster.json"), {
    context: "lab",
    namespace: "flakegraph",
    nodesReady: 2,
    nodesTotal: 2,
    nodes: [
      {
        name: "gpu-a",
        ready: true,
        nodeClass: "gb10",
        gpuCount: 1,
        gpuModel: "NVIDIA GB10",
        cpuCapacity: "20",
        memoryCapacity: "128Gi",
        gpuPercent: "41%",
        cpuUsage: "4",
        cpuPercent: "20%",
        memoryUsage: "32Gi",
        memoryPercent: "25%",
        workloadCount: 2,
        workerCount: 1,
        modelServerReady: true,
        model: "unsloth/Qwen3.8-27B-NVFP4",
        modelImage: "vllm/vllm-openai",
      },
      {
        name: "gpu-b",
        ready: true,
        nodeClass: "gb10",
        gpuCount: 1,
        gpuModel: "NVIDIA GB10",
        cpuCapacity: "20",
        memoryCapacity: "128Gi",
        gpuPercent: null,
        cpuUsage: null,
        cpuPercent: null,
        memoryUsage: null,
        memoryPercent: null,
        workloadCount: 0,
        workerCount: 0,
        modelServerReady: false,
        model: null,
        modelImage: null,
      },
    ],
    workloads: [
      {
        name: "worker-a",
        component: "worker-extraction",
        phase: "Running",
        node: "gpu-a",
        ready: true,
        restarts: 0,
        cpu: "2",
        memory: "8Gi",
        image: "flakegraph:mineru-oss",
        model: null,
      },
      {
        name: "vllm-a",
        component: "model-server",
        phase: "Running",
        node: "gpu-a",
        ready: true,
        restarts: 0,
        cpu: "4",
        memory: "24Gi",
        image: "vllm/vllm-openai",
        model: "unsloth/Qwen3.8-27B-NVFP4",
      },
    ],
    warnings: [],
    observedAt: new Date().toISOString(),
  });
  await atomicWriteJson(path.join(directory, "assignments.json"), {
    assignments: [
      {
        workerId: "worker-a",
        runId: "run_k8s_martial",
        graphId: "graph_k8s_martial",
        taskId: "task-1",
        stage: "extract_entity_window",
        scopeId: "gpu-a",
        updatedAt: new Date().toISOString(),
      },
    ],
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_k8s_martial"), {
    runId: "run_k8s_martial",
    graphId: "graph_k8s_martial",
    graphName: "Fleet martial arts",
    status: "running",
    runtime: "kubernetes",
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
    cancellationRequestedAt: null,
    error: null,
    owner: "ALICE",
  });
  const done = path.join(stateRoot, "runs", "run_k8s_done");
  // The configuration the run was submitted with, carrying the vocabulary it
  // extracted: a revision of this run is typed like it.
  await mkdir(done, { recursive: true });
  await writeFile(
    path.join(done, "config.yaml"),
    [
      "ontology:",
      "  profile:",
      "    name: console",
      "    mode: hybrid",
      "    entity_types:",
      "      - name: PERSON",
      "        description: A named human being.",
      "      - name: SCHOOL",
      "        description: A martial-arts school or lineage.",
      "    relation_types:",
      "      - name: FOUNDED_BY",
      "        description: The source was founded by the target.",
      "",
    ].join("\n"),
    "utf8",
  );
  await writeRunRecord(done, {
    runId: "run_k8s_done",
    graphId: "graph_k8s_done",
    graphName: "Fleet judo",
    status: "succeeded",
    runtime: "kubernetes",
    configPath: path.join(done, "config.yaml"),
    startedAt: "2026-09-05T00:00:00.000Z",
    updatedAt: "2026-09-05T01:30:00.000Z",
    outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
    documentsTotal: 2,
    documentsCompleted: 2,
    documentsFailed: 0,
    sourceKind: "upload",
    sourcePath: path.join(stateRoot, "uploads", "judo"),
    ocrProvider: "fallback",
    llmProvider: "vllm_local",
    embeddingProvider: "sentence_transformers",
    owner: "ALICE",
  });
  await writeFile(
    path.join(done, "events.jsonl"),
    ["judo-history.md", "karate-history.md"]
      .flatMap((fileId) => [
        makeProgressRecord({
          timestamp: "2026-09-05T00:10:00.000Z",
          stage: "ocr",
          status: "completed",
          fileId,
          message: null,
          elapsedMs: 100,
          counts: {},
        }),
        makeProgressRecord({
          timestamp: "2026-09-05T01:00:00.000Z",
          stage: "extract",
          status: "completed",
          fileId,
          message: null,
          elapsedMs: 100,
          counts: {},
        }),
      ])
      .map((event) => JSON.stringify(event))
      .join("\n") + "\n",
    "utf8",
  );
  // A later run that published the same graph: the stub fleet lists both as
  // versions, this one as head, so editing the older one can be exercised.
  const doneV2 = path.join(stateRoot, "runs", "run_k8s_done_v2");
  await writeRunRecord(doneV2, {
    runId: "run_k8s_done_v2",
    graphId: "graph_k8s_done",
    graphName: "Fleet judo",
    status: "succeeded",
    runtime: "kubernetes",
    startedAt: "2026-09-06T00:00:00.000Z",
    updatedAt: "2026-09-06T01:00:00.000Z",
    outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
    documentsTotal: 1,
    documentsCompleted: 1,
    documentsFailed: 0,
    sourceKind: "upload",
    sourcePath: path.join(stateRoot, "uploads", "judo"),
    ocrProvider: "fallback",
    llmProvider: "vllm_local",
    embeddingProvider: "sentence_transformers",
    owner: "ALICE",
    baseRunId: "run_k8s_done",
  });
  await writeFile(
    path.join(doneV2, "events.jsonl"),
    [
      makeProgressRecord({
        timestamp: "2026-09-06T00:10:00.000Z",
        stage: "ocr",
        status: "completed",
        fileId: "judo-history.md",
        message: null,
        elapsedMs: 100,
        counts: {},
      }),
      makeProgressRecord({
        timestamp: "2026-09-06T00:20:00.000Z",
        stage: "extract",
        status: "completed",
        fileId: "judo-history.md",
        message: null,
        elapsedMs: 100,
        counts: {},
      }),
    ]
      .map((event) => JSON.stringify(event))
      .join("\n") + "\n",
    "utf8",
  );
  await writeRunRecord(path.join(stateRoot, "runs", "run_k8s_failed"), {
    runId: "run_k8s_failed",
    graphId: "graph_k8s_failed",
    graphName: "Fleet timeout",
    status: "failed",
    runtime: "kubernetes",
    startedAt: "2026-09-04T00:00:00.000Z",
    updatedAt: "2026-09-04T00:10:00.000Z",
    outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
    error: "Worker lease expired",
    sourceKind: "local_path",
    sourcePath: "data/martial_arts/files",
    ocrProvider: "fallback",
    llmProvider: "vllm_local",
    embeddingProvider: "sentence_transformers",
    owner: "ALICE",
  });
  // A graph rebuilt every few days: more versions than one page of the
  // versions list holds. Dated before the other fixtures so the newest
  // catalog rows stay what the other journeys expect.
  for (let index = 1; index <= MANY_VERSIONS; index += 1) {
    const day = String(index).padStart(2, "0");
    await writeRunRecord(path.join(stateRoot, "runs", `run_k8s_kata_${day}`), {
      runId: `run_k8s_kata_${day}`,
      graphId: "graph_k8s_kata",
      graphName: "Revised kata",
      status: "succeeded",
      runtime: "kubernetes",
      startedAt: `2026-08-${day}T00:00:00.000Z`,
      updatedAt: `2026-08-${day}T01:00:00.000Z`,
      outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
      storageKind: "local_files",
      storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
      documentsTotal: 2,
      documentsCompleted: 2,
      documentsFailed: 0,
      sourceKind: "upload",
      sourcePath: path.join(stateRoot, "uploads", "kata"),
      ocrProvider: "fallback",
      llmProvider: "vllm_local",
      embeddingProvider: "sentence_transformers",
      owner: "ALICE",
      baseRunId: index > 1 ? `run_k8s_kata_${String(index - 1).padStart(2, "0")}` : undefined,
    });
  }
}

/** Versions of the "Revised kata" fleet graph: a page of ten and a few more. */
const MANY_VERSIONS = 14;

async function seedSnowflake(stateRoot: string) {
  const outputPath = path.join(stateRoot, "graphs", "graph_martial_arts");
  const thinPath = path.join(stateRoot, "graphs", "graph_snow_thin");
  await writeDataset(thinPath, {
    graphId: "graph_snow_thin",
    nodes: [],
    edges: [],
    communities: [],
    evidence: [],
    documents: [],
    chunks: [],
    runReport: {},
    graphMetrics: {},
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_snow_martial"), {
    runId: "run_snow_martial",
    graphId: "graph_snow_martial",
    graphName: "Shared martial arts",
    status: "succeeded",
    runtime: "snowflake",
    startedAt: "2026-09-03T00:00:00.000Z",
    updatedAt: "2026-09-03T02:00:00.000Z",
    outputPath,
    storageKind: "snowflake",
    storageLocation: "FLAKEGRAPH.PUBLIC",
    documentsTotal: 10,
    documentsCompleted: 10,
    documentsFailed: 0,
    nodeCount: 74,
    owner: null,
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_snow_private"), {
    runId: "run_snow_private",
    graphId: "graph_snow_private",
    graphName: "Private graph",
    status: "succeeded",
    runtime: "snowflake",
    startedAt: "2026-09-03T00:00:00.000Z",
    updatedAt: "2026-09-03T01:00:00.000Z",
    outputPath,
    storageKind: "snowflake",
    storageLocation: "FLAKEGRAPH.PUBLIC",
    documentsTotal: 10,
    documentsCompleted: 10,
    documentsFailed: 0,
    owner: "BOB",
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_snow_delete"), {
    runId: "run_snow_delete",
    graphId: "graph_snow_delete",
    graphName: "Disposable snowflake graph",
    status: "succeeded",
    runtime: "snowflake",
    startedAt: "2026-09-03T00:00:00.000Z",
    updatedAt: "2026-09-03T01:30:00.000Z",
    outputPath,
    storageKind: "snowflake",
    storageLocation: "FLAKEGRAPH.PUBLIC",
    documentsTotal: 10,
    documentsCompleted: 10,
    documentsFailed: 0,
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_snow_thin"), {
    runId: "run_snow_thin",
    graphId: "graph_snow_thin",
    graphName: "Empty share candidate",
    status: "succeeded",
    runtime: "snowflake",
    startedAt: "2026-09-03T00:00:00.000Z",
    updatedAt: "2026-09-03T01:45:00.000Z",
    outputPath: thinPath,
    storageKind: "snowflake",
    storageLocation: "FLAKEGRAPH.PUBLIC",
    documentsTotal: 10,
    documentsCompleted: 10,
    documentsFailed: 0,
    nodeCount: 0,
  });
  await writeRunRecord(path.join(stateRoot, "runs", "run_snow_queued"), {
    runId: "run_snow_queued",
    graphId: "graph_snow_queued",
    graphName: "Queued Snowflake job",
    status: "queued",
    runtime: "snowflake",
    startedAt: "2026-09-03T03:00:00.000Z",
    updatedAt: "2026-09-03T03:00:00.000Z",
    outputPath: thinPath,
    storageKind: "snowflake",
    storageLocation: "FLAKEGRAPH.PUBLIC",
    documentsTotal: 10,
    documentsCompleted: 0,
    documentsFailed: 0,
  });
  await atomicWriteJson(path.join(stateRoot, "snowflake", "store.json"), {
    graphs: [
      {
        graphId: "graph_snow_martial",
        owner: null,
        location: outputPath,
        name: "Shared martial arts",
        shares: [{ granteeType: "ROLE", grantee: "ANALYST" }],
      },
      {
        graphId: "graph_snow_private",
        owner: "BOB",
        location: outputPath,
        name: "Private graph",
        shares: [],
      },
      {
        graphId: "graph_snow_delete",
        owner: null,
        location: outputPath,
        name: "Disposable snowflake graph",
        shares: [],
      },
      {
        graphId: "graph_snow_thin",
        owner: null,
        location: thinPath,
        name: "Empty share candidate",
        shares: [],
      },
    ],
    jobs: [],
  });
}

async function persistName(stateRoot: string, graphId: string, name: string) {
  const file = path.join(stateRoot, "graph-names.json");
  await atomicWriteJson(file, { ...(JSON.parse(await readFile(file, "utf8").catch(() => "{}")) as Record<string, string>), [graphId]: name });
}

async function seedCancellingRun(stateRoot: string) {
  await writeRunRecord(path.join(stateRoot, "runs", "run_cancelling"), {
    runId: "run_cancelling",
    graphId: "graph_cancelling",
    graphName: "Cancelling extract",
    status: "cancelling",
    runtime: "kubernetes",
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    outputPath: path.join(stateRoot, "graphs", "graph_martial_arts"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_martial_arts"),
    documentsTotal: 10,
    documentsCompleted: 4,
    owner: "ALICE",
  });
}

/**
 * A corpus the size FlakeGraph is built for: 300 documents in flight, a
 * few of them failed. Every document list in the console has to stay
 * readable here, not only on the ten-file packs.
 */
async function seedLargeCorpusRun(stateRoot: string) {
  const directory = path.join(stateRoot, "runs", "run_large_corpus");
  const total = 300;
  await writeRunRecord(directory, {
    runId: "run_large_corpus",
    graphId: "graph_large_corpus",
    graphName: "Large corpus",
    status: "running",
    runtime: "local",
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    outputPath: path.join(stateRoot, "graphs", "graph_large_corpus"),
    storageKind: "local_files",
    storageLocation: path.join(stateRoot, "graphs", "graph_large_corpus"),
    documentsTotal: total,
    documentsCompleted: 120,
    documentsFailed: 6,
    owner: "ALICE",
  });
  const events = [];
  for (let index = 1; index <= total; index += 1) {
    const fileId = `paper-${String(index).padStart(3, "0")}.pdf`;
    if (index % 43 === 0) {
      events.push(
        makeProgressRecord({
          timestamp: new Date().toISOString(),
          stage: "ocr",
          status: "failed",
          fileId,
          message: "Poison PDF quarantined",
          elapsedMs: 900,
          counts: {},
        }),
      );
      continue;
    }
    events.push(
      makeProgressRecord({
        timestamp: new Date().toISOString(),
        stage: "ocr",
        status: "completed",
        fileId,
        message: null,
        elapsedMs: 100,
        counts: {},
      }),
    );
    if (index <= 120) {
      events.push(
        makeProgressRecord({
          timestamp: new Date().toISOString(),
          stage: "extract",
          status: "completed",
          fileId,
          message: null,
          elapsedMs: 100,
          counts: {},
        }),
      );
    }
  }
  await writeFile(path.join(directory, "events.jsonl"), events.map((event) => JSON.stringify(event)).join("\n") + "\n", "utf8");
}

async function seedPiiPack(stateRoot: string) {
  const directory = path.join(stateRoot, "pii-pack");
  await mkdir(directory, { recursive: true });
  await writeFile(path.join(directory, "contact.md"), "Contact jane@example.com about the corpus.\n", "utf8");
}

async function seedWorkspace(stateRoot: string) {
  await atomicWriteJson(path.join(stateRoot, "workspace.json"), {
    identity: null,
    role: "operator",
    suggestionMode: "on-request",
    lastSuccessRuntime: "local",
    lastConfigDigest: { local: "local_path:data/martial_arts/files" },
    piiAcknowledged: [],
    skippedFiles: {},
    estimates: {
      run_martial_arts: {
        objectCount: 10,
        runtime: "local",
        preset: "fast",
        usdLow: 1.2,
        usdHigh: 2.4,
        minutesLow: 4,
        minutesHigh: 8,
        note: "Local work is priced against the hosted reference card, not a bill.",
      },
    },
    perspectives: [
      {
        id: "perspective_judo",
        graphId: "graph_martial_arts",
        name: "Judo",
        lifecycle: "production",
        search: "judo",
        communityIds: ["community_martial_art"],
        suggestedQuestions: ["Who developed judo?", "Which schools taught judo?"],
      },
      {
        id: "perspective_judo_draft",
        graphId: "graph_martial_arts",
        name: "Judo draft",
        lifecycle: "draft",
        search: "kano",
        communityIds: ["community_person"],
        suggestedQuestions: ["Who is Jigoro Kano?"],
      },
    ],
    // A graph the workspace has recorded more versions of than one page of
    // the Versions card holds; the last one is what is in production.
    versions: Array.from({ length: WORKSPACE_VERSIONS }, (_, index) => {
      const number = index + 1;
      const last = number === WORKSPACE_VERSIONS;
      return {
        id: `v_martial_${number}`,
        graphId: "graph_martial_arts",
        label: `v${number}`,
        lifecycle: last ? "production" : "candidate",
        createdAt: `2026-08-${String(number).padStart(2, "0")}T00:00:00.000Z`,
        note: last ? "Gold-passed martial arts extract" : `Rebuilt after watch ingest ${number}`,
      };
    }),
    pins: [],
    reviews: [
      {
        id: "review_1",
        graphId: "graph_martial_arts",
        edgeId: "rel_judo_developed_by",
        quote: "Judo was developed by Jigoro Kano.",
        confidence: 0.42,
        decision: null,
      },
    ],
    watches: [
      {
        id: "watch_martial",
        graphId: "graph_martial_arts",
        prefix: "data/martial_arts/files",
        paused: false,
        lastWindow: "2026-09-10T00:00:00.000Z",
        pendingFiles: 3,
        goldDrift: 2,
      },
    ],
    // A deployment that has minted a key per pipeline for a while: more than
    // one page of the keys list. Their secrets are random and never shown,
    // so the fixtures cannot be used to authenticate.
    apiKeys: Array.from({ length: SEEDED_KEYS }, (_, index) => {
      const number = index + 1;
      const secret = `fg_${randomBytes(18).toString("hex")}`;
      return {
        id: `key_seed_${number}`,
        name: `${["nightly-eval", "deploy-bot", "notebook"][index % 3]}-${String(number).padStart(2, "0")}`,
        preview: `${secret.slice(0, 7)}…${secret.slice(-4)}`,
        createdAt: `2026-07-${String(number).padStart(2, "0")}T00:00:00.000Z`,
        note: "This path does not use your SSO cookie. A 401 here is not a browser login bug.",
        secretHash: hashApiSecret(secret),
      };
    }),
    grants: [
      { object: "WAREHOUSE FLAKEGRAPH_WH", ok: true, sql: "GRANT USAGE ON WAREHOUSE FLAKEGRAPH_WH TO ROLE APP_OPERATOR;" },
      { object: "DATABASE FLAKEGRAPH", ok: true, sql: "GRANT USAGE ON DATABASE FLAKEGRAPH TO ROLE APP_OPERATOR;" },
      { object: "STAGE APP_STAGE", ok: true, sql: "GRANT READ, WRITE ON STAGE FLAKEGRAPH.PUBLIC.APP_STAGE TO ROLE APP_OPERATOR;" },
      { object: "CORTEX", ok: true, sql: "GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE APP_OPERATOR;" },
    ],
  });
}

/** Workspace versions of the martial arts graph: a page of ten and a couple more. */
const WORKSPACE_VERSIONS = 12;

/** Machine keys on the seeded control plane: a page of 25 and a few more. */
const SEEDED_KEYS = 30;

await main();
