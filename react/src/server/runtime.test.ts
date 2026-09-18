import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { Effect } from "effect";
import { resetAppEnv } from "./env";
import { LocalRuntime } from "./runtime/local";
import { SnowflakeRuntime } from "./runtime/snowflake";
import { KubernetesRuntime } from "./runtime/kubernetes";
import type { IngestionRequest } from "./protocol/schema";

function request(stateRoot: string, overrides: Partial<IngestionRequest> = {}): IngestionRequest {
  return {
    runtime: "local",
    jobId: "job_demo",
    graphId: "graph_demo",
    graphName: "Demo graph",
    sourceKind: "local_path",
    source: { path: path.join(stateRoot, "corpus") },
    ocr: { provider: "fallback", model: null, endpoint: null, apiKeyEnvironmentVariable: null, dimension: null, options: {} },
    llm: { provider: "vllm_local", model: "qwen", endpoint: "http://localhost:8000/v1", apiKeyEnvironmentVariable: null, dimension: null, options: {} },
    embedding: {
      provider: "sentence_transformers",
      model: "mini",
      endpoint: null,
      apiKeyEnvironmentVariable: null,
      dimension: 384,
      options: {},
    },
    output: { kind: "local_files", workspacePath: path.join(stateRoot, "graphs", "graph_demo"), snowflake: null },
    baseConfigPath: null,
    includeGlobs: ["**/*"],
    cacheProvider: "local",
    providerParallelism: 4,
    runtimeOptions: {},
    revision: null,
    ...overrides,
  };
}

describe("local runtime", () => {
  afterEach(() => {
    resetAppEnv();
    delete process.env.FLAKEGRAPH_CLI;
    delete process.env.FLAKEGRAPH_APP_STATE_ROOT;
  });

  it("lists local files, preflights, submits, and loads a completed graph", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-local-"));
    const corpus = path.join(stateRoot, "corpus");
    await mkdir(corpus, { recursive: true });
    await writeFile(path.join(corpus, "note.md"), "Judo was developed by Jigoro Kano.\n");
    process.env.FLAKEGRAPH_APP_STATE_ROOT = stateRoot;
    process.env.FLAKEGRAPH_CLI = `bun ${path.resolve(import.meta.dirname, "../../scripts/fake-flakegraph.ts")}`;
    resetAppEnv();
    const runtime = new LocalRuntime(path.resolve(import.meta.dirname, "../../.."), stateRoot);
    const objects = await Effect.runPromise(runtime.listSourceObjects({ kind: "local", path: corpus }));
    expect(objects.some((object) => object.name === "note.md")).toBe(true);
    const ingestion = request(stateRoot);
    const preflight = await Effect.runPromise(runtime.preflight(ingestion));
    expect(preflight.ok).toBe(true);
    const submitted = await Effect.runPromise(runtime.submit(ingestion));
    expect(submitted.status).toBe("running");
    const terminal = await waitFor(async () => {
      const snapshot = await Effect.runPromise(runtime.getRun(ingestion.jobId));
      if (snapshot.status === "running") {
        throw new Error("still running");
      }
      return snapshot;
    });
    expect(terminal.status).toBe("succeeded");
    const graph = await Effect.runPromise(runtime.loadRunGraph(terminal));
    expect(graph.nodes.length).toBeGreaterThan(0);
    const renamed = await Effect.runPromise(runtime.renameGraph(ingestion.graphId, "Renamed demo"));
    expect(renamed).toBe("Renamed demo");
  }, 20_000);

  it("fails preflight for a missing corpus marker", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-local-fail-"));
    process.env.FLAKEGRAPH_APP_STATE_ROOT = stateRoot;
    process.env.FLAKEGRAPH_CLI = `bun ${path.resolve(import.meta.dirname, "../../scripts/fake-flakegraph.ts")}`;
    resetAppEnv();
    const runtime = new LocalRuntime(path.resolve(import.meta.dirname, "../../.."), stateRoot);
    const result = await Effect.runPromise(
      runtime.preflight(request(stateRoot, { source: { path: "/tmp/missing-corpus" }, jobId: "job_fail" })),
    );
    expect(result.ok).toBe(false);
  });
});

describe("snowflake runtime", () => {
  it("enforces share and owner predicates", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-snow-"));
    const alice = new SnowflakeRuntime(process.cwd(), stateRoot, true, {
      userName: "ALICE",
      email: "",
      roles: [],
    });
    const ingestion = request(stateRoot, {
      runtime: "snowflake",
      jobId: "job_snow",
      graphId: "graph_snow",
      output: {
        kind: "snowflake",
        workspacePath: path.join(stateRoot, "graphs", "graph_snow"),
        snowflake: {
          account: "xy123",
          user: "ALICE",
          database: "FG",
          schema: "PUBLIC",
          warehouse: "COMPUTE",
          bulkStage: "@stage",
          role: null,
          host: null,
          authenticator: null,
          credentialEnvironmentVariable: null,
          credentialField: null,
        },
      },
    });
    await mkdir(path.join(stateRoot, "graphs", "graph_snow"), { recursive: true });
    await Effect.runPromise(alice.submit(ingestion));
    await Effect.runPromise(alice.shareGraph("graph_snow", "USER", "CAROL"));
    const bob = new SnowflakeRuntime(process.cwd(), stateRoot, true, {
      userName: "BOB",
      email: "",
      roles: [],
    });
    const visibleToBob = await Effect.runPromise(bob.listRuns());
    expect(visibleToBob).toHaveLength(0);
    const carol = new SnowflakeRuntime(process.cwd(), stateRoot, true, {
      userName: "CAROL",
      email: "",
      roles: [],
    });
    const visibleToCarol = await Effect.runPromise(carol.listRuns());
    expect(visibleToCarol[0]?.graphId).toBe("graph_snow");
    await expect(Effect.runPromise(carol.renameGraph("graph_snow", "Hijack"))).rejects.toThrow(/owner/i);
    const unidentified = new SnowflakeRuntime(process.cwd(), stateRoot, true, {
      userName: "",
      email: "",
      roles: [],
    });
    expect(await Effect.runPromise(unidentified.listRuns())).toHaveLength(0);
  });
});

describe("kubernetes runtime", () => {
  it("registers clusters and reads a stub fleet snapshot", async () => {
    const stateRoot = await mkdtemp(path.join(tmpdir(), "fg-k8s-"));
    const runtime = new KubernetesRuntime(process.cwd(), stateRoot, true);
    const saved = await Effect.runPromise(
      runtime.upsertCluster({
        name: "lab1",
        namespace: "flakegraph",
        context: "lab",
        kubeconfig: "",
        description: "test",
      }),
    );
    expect(saved.name).toBe("lab1");
    await Effect.runPromise(runtime.selectCluster("lab1"));
    const selected = await Effect.runPromise(runtime.selectedCluster());
    expect(selected?.name).toBe("lab1");
    const cluster = await Effect.runPromise(runtime.cluster("flakegraph"));
    expect(cluster === null || Array.isArray(cluster.nodes)).toBe(true);
    const submitted = await Effect.runPromise(
      runtime.submit(request(stateRoot, { runtime: "kubernetes", jobId: "job_k8s", graphId: "graph_k8s", graphName: "Fleet job" })),
    );
    expect(submitted.status).toBe("queued");
    const listed = await Effect.runPromise(runtime.listRuns());
    expect(listed.some((run) => run.runId === "job_k8s")).toBe(true);
    const cancelled = await Effect.runPromise(runtime.cancel("job_k8s"));
    expect(cancelled.status).toBe("cancelled");
    const retried = await Effect.runPromise(runtime.retry("job_k8s"));
    expect(retried.status).toBe("queued");
  });
});

async function waitFor<T>(fn: () => Promise<T>, attempts = 40): Promise<T> {
  let last: unknown;
  for (let index = 0; index < attempts; index += 1) {
    try {
      return await fn();
    } catch (error) {
      last = error;
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }
  throw last;
}
