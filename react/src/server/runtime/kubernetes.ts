import { mkdir } from "node:fs/promises";
import path from "node:path";
import { Effect } from "effect";
import { stringify as stringifyYaml } from "yaml";
import {
  atomicWriteJson,
  readClusterCatalog,
  readJsonFile,
  writeClusterCatalog,
  writeRunRecord,
  listRunRecords,
  readRunRecord,
  runDirectory,
  runRecordExists,
  graphName,
  renameGraph as persistGraphName,
  snapshotFromRecord,
  removeRunDirectory,
  cloneFieldsFromRequest,
} from "../catalog";
import { lastJsonObject, runFlakegraph } from "../cli";
import { buildRunConfig, environmentForRequest, redactedConfig, writeRunConfig } from "../config";
import { appEnv } from "../env";
import { listPostgresRuns } from "../db/client";
import {
  KUBERNETES_CAPABILITIES,
  type Capability,
  type ClusterProfile,
  type ClusterSnapshot,
  type GraphDataset,
  type GraphShare,
  type IngestionRequest,
  type NodeStatus,
  type NodeWorkAssignment,
  type PreflightResult,
  type RunSnapshot,
  type SourceObject,
  type Viewer,
  type WorkloadStatus,
  isActiveStatus,
  validateGraphName,
} from "../protocol/schema";
import { ControlPlaneError, fromCause, invalid, notFound, notSupported } from "../protocol/errors";
import type { ControlPlane } from "../protocol/runtime";
import { LocalRuntime } from "./local";
import { loadLocalGraph } from "../graph";
import { catalogWriterPrincipal } from "../workspace";

const NAME_PATTERN = /^[a-z0-9][a-z0-9-]{0,38}[a-z0-9]$/;
const NAMESPACE_PATTERN = /^[a-z0-9][a-z0-9-]{0,61}[a-z0-9]$/;

export class KubernetesRuntime implements ControlPlane {
  readonly runtime = "kubernetes" as const;
  readonly capabilities = new Set<Capability>(KUBERNETES_CAPABILITIES);
  private readonly local: LocalRuntime;

  constructor(
    readonly repositoryRoot: string,
    readonly stateRoot: string,
    readonly stubbed: boolean,
  ) {
    this.local = new LocalRuntime(repositoryRoot, stateRoot);
  }

  viewer(): Effect.Effect<Viewer, ControlPlaneError> {
    return this.local.viewer();
  }

  listSourceObjects(
    source: Record<string, unknown>,
    limit = 1_000,
  ): Effect.Effect<readonly SourceObject[], ControlPlaneError> {
    return this.local.listSourceObjects(source, limit);
  }

  preflight(request: IngestionRequest): Effect.Effect<PreflightResult, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (this.stubbed) {
          const configPath = path.join(runDirectory(this.stateRoot, request.jobId), "config.yaml");
          await writeRunConfig(request, configPath);
          return { ok: true, errors: [], warnings: ["Stub fleet accepted the run."], checks: [] };
        }
        const configPath = path.join(runDirectory(this.stateRoot, request.jobId), "config.yaml");
        await writeRunConfig(request, configPath);
        const result = await runFlakegraph(["preflight", "--config", configPath, "--orchestrator"], {
          cwd: this.repositoryRoot,
          env: environmentForRequest(request),
        });
        const payload = lastJsonObject(result.stdout) ?? lastJsonObject(result.stderr);
        if (!payload) {
          return {
            ok: false,
            errors: [result.stderr.trim() || "Fleet preflight returned no JSON result"],
            warnings: [],
            checks: [],
          };
        }
        return {
          ok: payload.ok !== false && result.exitCode === 0,
          errors: asStrings(payload.errors),
          warnings: asStrings(payload.warnings),
          checks: Array.isArray(payload.checks) ? (payload.checks as Record<string, unknown>[]) : [],
        };
      },
      catch: (cause) => fromCause(cause, "Fleet preflight failed"),
    });
  }

  submit(request: IngestionRequest): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (request.graphName) {
          await persistGraphName(this.stateRoot, request.graphId, validateGraphName(request.graphName));
        }
        const directory = runDirectory(this.stateRoot, request.jobId);
        const configPath = await writeRunConfig(request, path.join(directory, "config.yaml"));
        const startedAt = new Date().toISOString();
        if (this.stubbed) {
          await writeRunRecord(directory, {
            runId: request.jobId,
            graphId: request.graphId,
            graphName: request.graphName,
            status: "queued",
            runtime: "kubernetes",
            startedAt,
            updatedAt: startedAt,
            outputPath: request.output.workspacePath,
            configPath,
            storageKind: request.output.kind,
            storageLocation: request.output.workspacePath,
            owner: await catalogWriterPrincipal(this.stateRoot),
            ...cloneFieldsFromRequest(request),
          });
          await appendStubRun(this.stateRoot, {
            runId: request.jobId,
            graphId: request.graphId,
            status: "queued",
          });
        } else {
          const result = await runFlakegraph(
            ["distributed", "submit", "--config", configPath, "--run-id", request.jobId],
            { cwd: this.repositoryRoot, env: environmentForRequest(request) },
          );
          const payload = lastJsonObject(result.stdout) ?? {};
          await writeRunRecord(directory, {
            runId: request.jobId,
            graphId: request.graphId,
            graphName: request.graphName,
            status: String(payload.status ?? (result.exitCode === 0 ? "queued" : "failed")),
            runtime: "kubernetes",
            startedAt,
            updatedAt: startedAt,
            outputPath: request.output.workspacePath,
            configPath,
            storageKind: request.output.kind,
            storageLocation: request.output.workspacePath,
            error: result.exitCode === 0 ? null : result.stderr,
            owner: await catalogWriterPrincipal(this.stateRoot),
            ...cloneFieldsFromRequest(request),
          });
        }
        return Effect.runPromise(this.getRun(request.jobId));
      },
      catch: (cause) => fromCause(cause, "Unable to submit distributed run"),
    });
  }

  listRuns(limit = 100): Effect.Effect<readonly RunSnapshot[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const catalog = await listRunRecords(this.stateRoot, limit);
        const local = await Promise.all(
          catalog
            .filter(({ record }) => String(record.runtime || "") === "kubernetes")
            .map(async ({ record }) =>
              snapshotFromRecord(record, {
                graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
              }),
            ),
        );
        if (this.stubbed || local.length > 0) {
          return local;
        }
        try {
          const rows = await listPostgresRuns(limit);
          return rows.map((row) => ({
            runId: row.id,
            graphId: row.graphId,
            status: row.status,
            startedAt: row.createdAt?.toISOString() ?? null,
            updatedAt: row.updatedAt?.toISOString() ?? null,
            graphName: null,
            stages: [],
            events: [],
            documentsTotal: null,
            documentsCompleted: 0,
            documentsFailed: 0,
            outputPath: null,
            storageKind: "local_files" as const,
            storageLocation: null,
            warnings: [],
            error: row.errorJson ? JSON.stringify(row.errorJson) : null,
            listingWarning: null,
            raw: {},
          }));
        } catch (error) {
          return catalog.map(({ record }) =>
            snapshotFromRecord(record, {
              listingWarning: error instanceof Error ? error.message : "Postgres listing unavailable",
            }),
          );
        }
      },
      catch: (cause) => fromCause(cause, "Unable to list fleet runs"),
    });
  }

  getRun(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const directory = runDirectory(this.stateRoot, runId);
        if (!runRecordExists(directory)) {
          throw notFound(`Unknown Kubernetes run: ${runId}`);
        }
        const record = await readRunRecord(directory);
        if (!this.stubbed && record.configPath) {
          const result = await runFlakegraph(["distributed", "status", "--config", record.configPath, "--run-id", runId], {
            cwd: this.repositoryRoot,
          });
          const payload = lastJsonObject(result.stdout);
          if (payload?.status) {
            record.status = String(payload.status);
            await writeRunRecord(runDirectory(this.stateRoot, runId), { status: record.status });
          }
        }
        return snapshotFromRecord(record, {
          graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
        });
      },
      catch: (cause) => fromCause(cause, "Unable to load Kubernetes run"),
    });
  }

  cancel(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const directory = runDirectory(this.stateRoot, runId);
        const record = await readRunRecord(directory);
        if (!record.runId) {
          throw notFound(`Unknown Kubernetes run: ${runId}`);
        }
        if (!this.stubbed && record.configPath) {
          await runFlakegraph(["distributed", "cancel", "--config", record.configPath, "--run-id", runId], {
            cwd: this.repositoryRoot,
          });
        }
        await writeRunRecord(directory, {
          status: "cancelled",
          cancellationRequestedAt: new Date().toISOString(),
        });
        return Effect.runPromise(this.getRun(runId));
      },
      catch: (cause) => fromCause(cause, "Unable to cancel Kubernetes run"),
    });
  }

  retry(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const directory = runDirectory(this.stateRoot, runId);
        const record = await readRunRecord(directory);
        if (!record.runId) {
          throw notFound(`Unknown Kubernetes run: ${runId}`);
        }
        if (!this.stubbed && record.configPath) {
          await runFlakegraph(["distributed", "retry", "--config", record.configPath, "--run-id", runId], {
            cwd: this.repositoryRoot,
          });
        }
        await writeRunRecord(directory, { status: "queued", error: null, cancellationRequestedAt: null });
        return Effect.runPromise(this.getRun(runId));
      },
      catch: (cause) => fromCause(cause, "Unable to retry Kubernetes run"),
    });
  }

  recover(runId: string): Effect.Effect<string, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const directory = runDirectory(this.stateRoot, runId);
        if (!runRecordExists(directory)) {
          throw notFound(`Unknown Kubernetes run: ${runId}`);
        }
        await writeRunRecord(directory, { status: "queued", cancellationRequestedAt: null });
        return `Reconciled infrastructure for ${runId}`;
      },
      catch: (cause) => fromCause(cause, "Unable to recover Kubernetes run"),
    });
  }

  forget(runId: string): Effect.Effect<void, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const snapshot = await Effect.runPromise(this.getRun(runId));
        if (isActiveStatus(snapshot.status)) {
          throw invalid("An active run must be cancelled before removal");
        }
        await removeRunDirectory(this.stateRoot, runId);
      },
      catch: (cause) => fromCause(cause, "Unable to forget Kubernetes run"),
    });
  }

  renameGraph(graphId: string, displayName: string): Effect.Effect<string, ControlPlaneError> {
    return this.local.renameGraph(graphId, displayName);
  }

  loadGraph(location: string, graphId?: string | null): Effect.Effect<GraphDataset, ControlPlaneError> {
    return this.local.loadGraph(location, graphId);
  }

  loadRunGraph(snapshot: RunSnapshot): Effect.Effect<GraphDataset, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (!snapshot.outputPath) {
          throw invalid(`Run ${snapshot.runId} does not record an output directory`);
        }
        return loadLocalGraph(this.local.artifactDirectory(snapshot.outputPath));
      },
      catch: (cause) => fromCause(cause, "Unable to load fleet graph"),
    });
  }

  cluster(namespace: string): Effect.Effect<ClusterSnapshot | null, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (this.stubbed) {
          const snapshot = await readJsonFile<ClusterSnapshot>(path.join(this.stateRoot, "kubernetes", "cluster.json"));
          if (snapshot) {
            return { ...snapshot, namespace: snapshot.namespace || namespace };
          }
          return emptyClusterSnapshot(namespace, ["No stub cluster snapshot is present."]);
        }
        return await readLiveCluster(namespace);
      },
      catch: (cause) => fromCause(cause, "Unable to load cluster snapshot"),
    });
  }

  nodeAssignments(
    namespace: string,
    nodeName: string,
  ): Effect.Effect<readonly NodeWorkAssignment[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        try {
          const snapshot = await readJsonFile<{ assignments?: NodeWorkAssignment[] }>(
            path.join(this.stateRoot, "kubernetes", "assignments.json"),
          );
          void namespace;
          return (snapshot?.assignments ?? []).filter(
            (item) => item.scopeId === nodeName || item.workerId.includes(nodeName),
          );
        } catch {
          return [];
        }
      },
      catch: (cause) => fromCause(cause, "Unable to load node assignments"),
    });
  }

  listClusters(): Effect.Effect<readonly ClusterProfile[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => (await readClusterCatalog(this.stateRoot)).clusters,
      catch: (cause) => fromCause(cause, "Unable to list clusters"),
    });
  }

  upsertCluster(profile: ClusterProfile): Effect.Effect<ClusterProfile, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        validateCluster(profile);
        const catalog = await readClusterCatalog(this.stateRoot);
        const next = catalog.clusters.filter((item) => item.name !== profile.name);
        next.push(profile);
        await writeClusterCatalog(this.stateRoot, { ...catalog, clusters: next });
        return profile;
      },
      catch: (cause) => fromCause(cause, "Unable to save cluster"),
    });
  }

  deleteCluster(name: string): Effect.Effect<void, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const catalog = await readClusterCatalog(this.stateRoot);
        await writeClusterCatalog(this.stateRoot, {
          clusters: catalog.clusters.filter((item) => item.name !== name),
          selected: catalog.selected === name ? null : catalog.selected,
        });
      },
      catch: (cause) => fromCause(cause, "Unable to delete cluster"),
    });
  }

  selectCluster(name: string): Effect.Effect<ClusterProfile, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const catalog = await readClusterCatalog(this.stateRoot);
        const profile = catalog.clusters.find((item) => item.name === name);
        if (!profile) {
          throw notFound(`Unknown cluster: ${name}`);
        }
        await writeClusterCatalog(this.stateRoot, { ...catalog, selected: name });
        return profile;
      },
      catch: (cause) => fromCause(cause, "Unable to select cluster"),
    });
  }

  selectedCluster(): Effect.Effect<ClusterProfile | null, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const catalog = await readClusterCatalog(this.stateRoot);
        return catalog.clusters.find((item) => item.name === catalog.selected) ?? catalog.clusters[0] ?? null;
      },
      catch: (cause) => fromCause(cause, "Unable to read selected cluster"),
    });
  }

  graphOwner(): Effect.Effect<string | null, ControlPlaneError> {
    return Effect.succeed(null);
  }

  graphShares(): Effect.Effect<readonly GraphShare[], ControlPlaneError> {
    return Effect.succeed([]);
  }

  shareGraph(): Effect.Effect<void, ControlPlaneError> {
    return Effect.fail(notSupported("This runtime does not share graphs"));
  }

  unshareGraph(): Effect.Effect<void, ControlPlaneError> {
    return Effect.fail(notSupported("This runtime does not share graphs"));
  }

  deleteGraph(): Effect.Effect<Record<string, number>, ControlPlaneError> {
    return Effect.fail(notSupported("This runtime does not delete stored graphs"));
  }

  previewConfig(request: IngestionRequest): Effect.Effect<string, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => stringifyYaml(redactedConfig(await buildRunConfig(request)), { sortMapEntries: false }),
      catch: (cause) => fromCause(cause, "Unable to preview configuration"),
    });
  }
}

export function createKubernetesRuntime(): KubernetesRuntime {
  const env = appEnv();
  return new KubernetesRuntime(env.repositoryRoot, env.stateRoot, env.stubRuntimes);
}

async function readLiveCluster(namespace: string): Promise<ClusterSnapshot> {
  return Promise.race([
    readLiveClusterUnbound(namespace),
    new Promise<ClusterSnapshot>((resolve) => {
      setTimeout(() => {
        resolve(emptyClusterSnapshot(namespace, ["Timed out waiting for the Kubernetes API"]));
      }, 4_000);
    }),
  ]);
}

async function readLiveClusterUnbound(namespace: string): Promise<ClusterSnapshot> {
  try {
    const k8s = await import("@kubernetes/client-node");
    const config = new k8s.KubeConfig();
    config.loadFromDefault();
    const core = config.makeApiClient(k8s.CoreV1Api);
    const nodes = await core.listNode();
    const pods = await core.listNamespacedPod({ namespace });
    const nodeStatuses: NodeStatus[] = (nodes.items ?? []).map((node) => {
      const ready = (node.status?.conditions ?? []).some(
        (condition) => condition.type === "Ready" && condition.status === "True",
      );
      const gpu = Number(node.status?.capacity?.["nvidia.com/gpu"] ?? 0);
      const labels = node.metadata?.labels ?? {};
      return {
        name: node.metadata?.name ?? "unknown",
        ready,
        nodeClass: labels["node.kubernetes.io/instance-type"] ?? labels["kubernetes.io/arch"] ?? "unknown",
        gpuCount: Number.isFinite(gpu) ? gpu : 0,
        gpuModel: labels["nvidia.com/gpu.product"] ?? null,
        cpuCapacity: node.status?.capacity?.cpu ?? null,
        memoryCapacity: node.status?.capacity?.memory ?? null,
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
      };
    });
    const workloads: WorkloadStatus[] = (pods.items ?? []).map((pod) => ({
      name: pod.metadata?.name ?? "unknown",
      component: pod.metadata?.labels?.["app.kubernetes.io/component"] ?? "workload",
      phase: pod.status?.phase ?? "Unknown",
      node: pod.spec?.nodeName ?? null,
      ready: (pod.status?.containerStatuses ?? []).every((status) => status.ready),
      restarts: (pod.status?.containerStatuses ?? []).reduce((sum, status) => sum + (status.restartCount ?? 0), 0),
      cpu: null,
      memory: null,
      image: pod.spec?.containers?.[0]?.image ?? null,
      model: pod.metadata?.labels?.["flakegraph/model"] ?? null,
    }));
    const annotatedNodes = nodeStatuses.map((node) => {
      const onNode = workloads.filter((workload) => workload.node === node.name);
      const model = onNode.find((workload) => workload.component.includes("vllm") || workload.component.includes("model"));
      return {
        ...node,
        workloadCount: onNode.length,
        workerCount: onNode.filter((workload) => workload.component.startsWith("worker")).length,
        modelServerReady: Boolean(model?.ready),
        model: model?.model ?? null,
        modelImage: model?.image ?? null,
      };
    });
    return {
      context: config.currentContext ?? "default",
      namespace,
      nodesReady: annotatedNodes.filter((node) => node.ready).length,
      nodesTotal: annotatedNodes.length,
      nodes: annotatedNodes,
      workloads,
      warnings: [],
      observedAt: new Date().toISOString(),
    };
  } catch (error) {
    return emptyClusterSnapshot(namespace, [
      error instanceof Error ? error.message : "Kubernetes API is unavailable",
    ]);
  }
}

function emptyClusterSnapshot(namespace: string, warnings: string[]): ClusterSnapshot {
  return {
    context: "unavailable",
    namespace,
    nodesReady: 0,
    nodesTotal: 0,
    nodes: [],
    workloads: [],
    warnings,
    observedAt: new Date().toISOString(),
  };
}

async function appendStubRun(stateRoot: string, run: { runId: string; graphId: string; status: string }) {
  const file = path.join(stateRoot, "kubernetes", "runs.json");
  await mkdir(path.dirname(file), { recursive: true });
  const current = (await readJsonFile<typeof run[]>(file)) ?? [];
  current.unshift(run);
  await atomicWriteJson(file, current);
}

function validateCluster(profile: ClusterProfile): void {
  if (!NAME_PATTERN.test(profile.name) && profile.name !== "default") {
    throw invalid("Cluster name must be a short DNS label");
  }
  if (!NAMESPACE_PATTERN.test(profile.namespace) && profile.namespace.length > 1) {
    throw invalid("Namespace must be a valid Kubernetes namespace");
  }
}

function asStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)) : [];
}

