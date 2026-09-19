import path from "node:path";
import { mkdir, rm } from "node:fs/promises";
import { Effect } from "effect";
import { documentStatusesFromEvents, type DocumentStatus } from "../documents";
import { stringify as stringifyYaml } from "yaml";
import { listRunRecords, readJsonFile, readRunRecord, removeRunDirectory, renameGraph as persistGraphName, runDirectory, runRecordExists, snapshotFromRecord, writeRunRecord, graphName, atomicWriteJson, cloneFieldsFromRequest } from "../catalog";
import { buildRunConfig, redactedConfig, writeRunConfig } from "../config";
import { appEnv } from "../env";
import { loadLocalGraph } from "../graph";
import { unidentifiedViewer } from "../identity";
import {
  SNOWFLAKE_CAPABILITIES,
  type Capability,
  type ClusterSnapshot,
  type GraphDataset,
  type GraphShare,
  type IngestionRequest,
  type NodeWorkAssignment,
  type PreflightResult,
  type GraphVersion,
  type RunSnapshot,
  type SourceObject,
  type Viewer,
  isActiveStatus,
  validateGraphName,
} from "../protocol/schema";
import { ControlPlaneError, forbidden, fromCause, invalid, notFound, notSupported } from "../protocol/errors";
import type { ControlPlane } from "../protocol/runtime";
import { listLocalObjects } from "../sources";

interface SnowflakeGraphRecord {
  graphId: string;
  owner: string | null;
  location: string;
  name?: string | null;
  shares: GraphShare[];
  deleted?: boolean;
}

interface SnowflakeStore {
  graphs: SnowflakeGraphRecord[];
  jobs: Array<Record<string, unknown>>;
}

export class SnowflakeRuntime implements ControlPlane {
  readonly runtime = "snowflake" as const;
  readonly capabilities = new Set<Capability>(SNOWFLAKE_CAPABILITIES);
  private currentViewer: Viewer = unidentifiedViewer();

  constructor(
    readonly repositoryRoot: string,
    readonly stateRoot: string,
    readonly stubbed: boolean,
    viewer?: Viewer,
  ) {
    if (viewer) {
      this.currentViewer = viewer;
    }
  }

  withViewer(viewer: Viewer): SnowflakeRuntime {
    return new SnowflakeRuntime(this.repositoryRoot, this.stateRoot, this.stubbed, viewer);
  }

  viewer(): Effect.Effect<Viewer, ControlPlaneError> {
    return Effect.succeed(this.currentViewer);
  }

  listSourceObjects(
    source: Record<string, unknown>,
    limit = 1_000,
  ): Effect.Effect<readonly SourceObject[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const kind = String(source.kind ?? "snowflake_stage");
        if (kind === "local" || kind === "upload" || kind === "local_path") {
          return listLocalObjects(String(source.path ?? ""), limit);
        }
        const store = await this.store();
        const prefix = String(source.prefix ?? "");
        return store.jobs
          .filter((job) => String(job.stagePrefix ?? "").startsWith(prefix) || !prefix)
          .slice(0, limit)
          .map((job) => ({
            uri: String(job.uri ?? job.path ?? job.graphId ?? "stage://object"),
            name: String(job.name ?? job.uri ?? "object"),
            sizeBytes: Number(job.sizeBytes ?? 0),
            modifiedAt: String(job.modifiedAt ?? "") || null,
            checksum: null,
          }));
      },
      catch: (cause) => fromCause(cause, "Unable to list Snowflake stage objects"),
    });
  }

  preflight(request: IngestionRequest): Effect.Effect<PreflightResult, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        await writeRunConfig(request, path.join(runDirectory(this.stateRoot, request.jobId), "config.yaml"));
        if (request.embedding.dimension && request.embedding.dimension < 1) {
          return { ok: false, errors: ["Embedding dimension must be positive"], warnings: [], checks: [] };
        }
        return {
          ok: true,
          errors: [],
          warnings: this.stubbed ? ["Snowflake App Runtime stub accepted the job."] : [],
          checks: [{ name: "stage", ok: true }],
        };
      },
      catch: (cause) => fromCause(cause, "Snowflake preflight failed"),
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
        await writeRunRecord(directory, {
          runId: request.jobId,
          graphId: request.graphId,
          graphName: request.graphName,
          status: "queued",
          runtime: "snowflake",
          startedAt,
          updatedAt: startedAt,
          outputPath: request.output.workspacePath,
          configPath,
          storageKind: "snowflake",
          storageLocation: request.output.snowflake
            ? `${request.output.snowflake.database}.${request.output.snowflake.schema}`
            : request.output.workspacePath,
          ...cloneFieldsFromRequest(request),
          owner: this.currentViewer.userName || null,
        });
        const store = await this.store();
        store.jobs.unshift({
          jobId: request.jobId,
          graphId: request.graphId,
          status: "queued",
          owner: this.currentViewer.userName || null,
        });
        if (!store.graphs.some((graph) => graph.graphId === request.graphId)) {
          store.graphs.push({
            graphId: request.graphId,
            owner: this.currentViewer.userName || null,
            location: request.output.workspacePath,
            name: request.graphName,
            shares: [],
          });
        }
        await this.writeStore(store);
        return Effect.runPromise(this.getRun(request.jobId));
      },
      catch: (cause) => fromCause(cause, "Unable to submit Snowflake job"),
    });
  }

  listRuns(limit = 100): Effect.Effect<readonly RunSnapshot[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const rows = await listRunRecords(this.stateRoot, limit);
        const store = await this.store();
        const visible = [];
        for (const { record } of rows) {
          if (String(record.runtime || "") !== "snowflake") {
            continue;
          }
          if (!(await this.canSee(record.graphId))) {
            continue;
          }
          visible.push(
            snapshotFromRecord(record, {
              graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
              raw: {
                owner: record.owner ?? store.graphs.find((graph) => graph.graphId === record.graphId)?.owner ?? null,
              },
            }),
          );
        }
        return visible;
      },
      catch: (cause) => fromCause(cause, "Unable to list Snowflake jobs"),
    });
  }

  getRun(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const directory = runDirectory(this.stateRoot, runId);
        if (!runRecordExists(directory)) {
          throw notFound(`Unknown Snowflake job: ${runId}`);
        }
        const record = await readRunRecord(directory);
        if (!(await this.canSee(record.graphId))) {
          throw forbidden("This graph is not visible to the current viewer");
        }
        const store = await this.store();
        return snapshotFromRecord(record, {
          graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
          raw: {
            owner: record.owner ?? store.graphs.find((graph) => graph.graphId === record.graphId)?.owner ?? null,
          },
        });
      },
      catch: (cause) => fromCause(cause, "Unable to load Snowflake job"),
    });
  }

  documents(runId: string): Effect.Effect<readonly DocumentStatus[], ControlPlaneError> {
    return Effect.map(this.getRun(runId), (snapshot) => documentStatusesFromEvents(snapshot.events));
  }

  versions(_graphId: string): Effect.Effect<readonly GraphVersion[], ControlPlaneError> {
    return Effect.succeed([]);
  }

  cancel(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return this.mutateJob(runId, "cancelled");
  }

  retry(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return this.mutateJob(runId, "queued");
  }

  recover(_runId: string): Effect.Effect<string, ControlPlaneError> {
    return Effect.fail(notSupported("Infrastructure recovery is available only for Kubernetes runs"));
  }

  forget(runId: string): Effect.Effect<void, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const snapshot = await Effect.runPromise(this.getRun(runId));
        if (isActiveStatus(snapshot.status)) {
          throw invalid("An active run must be cancelled before removal");
        }
        await this.requireAdmin(snapshot.graphId);
        await removeRunDirectory(this.stateRoot, runId);
      },
      catch: (cause) => fromCause(cause, "Unable to forget Snowflake job"),
    });
  }

  renameGraph(graphId: string, displayName: string): Effect.Effect<string, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        await this.requireAdmin(graphId);
        const name = await persistGraphName(this.stateRoot, graphId, displayName);
        const store = await this.store();
        const graph = store.graphs.find((item) => item.graphId === graphId);
        if (graph) {
          graph.name = name;
          await this.writeStore(store);
        }
        return name;
      },
      catch: (cause) => fromCause(cause, "Unable to rename Snowflake graph"),
    });
  }

  loadGraph(location: string, graphId?: string | null): Effect.Effect<GraphDataset, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (graphId && !(await this.canSee(graphId))) {
          throw forbidden("This graph is not visible to the current viewer");
        }
        return loadLocalGraph(path.isAbsolute(location) ? location : path.resolve(this.repositoryRoot, location));
      },
      catch: (cause) => fromCause(cause, "Unable to load Snowflake graph"),
    });
  }

  loadRunGraph(snapshot: RunSnapshot): Effect.Effect<GraphDataset, ControlPlaneError> {
    if (!snapshot.outputPath) {
      return Effect.fail(invalid(`Job ${snapshot.runId} does not record an output location`));
    }
    return this.loadGraph(snapshot.outputPath, snapshot.graphId);
  }

  cluster(_namespace: string): Effect.Effect<ClusterSnapshot | null, ControlPlaneError> {
    return Effect.succeed(null);
  }

  nodeAssignments(): Effect.Effect<readonly NodeWorkAssignment[], ControlPlaneError> {
    return Effect.succeed([]);
  }


  graphOwner(graphId: string): Effect.Effect<string | null, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => (await this.store()).graphs.find((graph) => graph.graphId === graphId)?.owner ?? null,
      catch: (cause) => fromCause(cause, "Unable to read graph owner"),
    });
  }

  graphShares(graphId: string): Effect.Effect<readonly GraphShare[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => (await this.store()).graphs.find((graph) => graph.graphId === graphId)?.shares ?? [],
      catch: (cause) => fromCause(cause, "Unable to list graph shares"),
    });
  }

  shareGraph(graphId: string, granteeType: string, grantee: string): Effect.Effect<void, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        await this.requireAdmin(graphId);
        const store = await this.store();
        const graph = store.graphs.find((item) => item.graphId === graphId);
        if (!graph) {
          throw notFound(`Unknown graph: ${graphId}`);
        }
        const type = granteeType.trim().toUpperCase();
        const name = grantee.trim().toUpperCase();
        if (!name || (type !== "USER" && type !== "ROLE")) {
          throw invalid("Share requires a USER or ROLE grantee");
        }
        if (!graph.shares.some((share) => share.granteeType === type && share.grantee === name)) {
          graph.shares.push({ granteeType: type, grantee: name });
        }
        await this.writeStore(store);
      },
      catch: (cause) => fromCause(cause, "Unable to share graph"),
    });
  }

  unshareGraph(graphId: string, granteeType: string, grantee: string): Effect.Effect<void, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        await this.requireAdmin(graphId);
        const store = await this.store();
        const graph = store.graphs.find((item) => item.graphId === graphId);
        if (!graph) {
          throw notFound(`Unknown graph: ${graphId}`);
        }
        graph.shares = graph.shares.filter(
          (share) =>
            !(share.granteeType === granteeType.trim().toUpperCase() && share.grantee === grantee.trim().toUpperCase()),
        );
        await this.writeStore(store);
      },
      catch: (cause) => fromCause(cause, "Unable to unshare graph"),
    });
  }

  deleteGraph(graphId: string): Effect.Effect<Record<string, number>, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        await this.requireAdmin(graphId);
        const store = await this.store();
        const graph = store.graphs.find((item) => item.graphId === graphId);
        if (!graph) {
          throw notFound(`Unknown graph: ${graphId}`);
        }
        graph.deleted = true;
        await this.writeStore(store);
        const runs = await listRunRecords(this.stateRoot, 500);
        let removed = 0;
        for (const { record } of runs) {
          if (record.graphId === graphId) {
            await rm(runDirectory(this.stateRoot, record.runId), { recursive: true, force: true });
            removed += 1;
          }
        }
        return { KG_GRAPH: 1, runs: removed };
      },
      catch: (cause) => fromCause(cause, "Unable to delete graph"),
    });
  }

  previewConfig(request: IngestionRequest): Effect.Effect<string, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => stringifyYaml(redactedConfig(await buildRunConfig(request)), { sortMapEntries: false }),
      catch: (cause) => fromCause(cause, "Unable to preview configuration"),
    });
  }

  private mutateJob(runId: string, status: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const snapshot = await Effect.runPromise(this.getRun(runId));
        await this.requireAdmin(snapshot.graphId);
        await writeRunRecord(runDirectory(this.stateRoot, runId), {
          status,
          error: status === "queued" ? null : snapshot.error,
          cancellationRequestedAt: status === "cancelled" ? new Date().toISOString() : null,
        });
        return Effect.runPromise(this.getRun(runId));
      },
      catch: (cause) => fromCause(cause, "Unable to update Snowflake job"),
    });
  }

  private async store(): Promise<SnowflakeStore> {
    const file = this.storeFile();
    const current = await readJsonFile<SnowflakeStore>(file);
    return current ?? { graphs: [], jobs: [] };
  }

  private async writeStore(store: SnowflakeStore): Promise<void> {
    await mkdir(path.dirname(this.storeFile()), { recursive: true });
    await atomicWriteJson(this.storeFile(), store);
  }

  private storeFile(): string {
    return path.join(this.stateRoot, "snowflake", "store.json");
  }

  private async canSee(graphId: string): Promise<boolean> {
    const store = await this.store();
    const graph = store.graphs.find((item) => item.graphId === graphId && !item.deleted);
    if (!graph) {
      return true;
    }
    if (!graph.owner) {
      return true;
    }
    const viewer = this.currentViewer;
    if (!viewer.userName) {
      return false;
    }
    if (graph.owner.toUpperCase() === viewer.userName.toUpperCase()) {
      return true;
    }
    return graph.shares.some((share) => {
      const grantee = share.grantee.toUpperCase();
      if (share.granteeType.toUpperCase() === "USER") {
        return grantee === viewer.userName.toUpperCase();
      }
      return viewer.roles.some((role) => role.toUpperCase() === grantee);
    });
  }

  private async requireAdmin(graphId: string): Promise<void> {
    const owner = await Effect.runPromise(this.graphOwner(graphId));
    const viewer = this.currentViewer;
    if (!owner) {
      return;
    }
    if (!viewer.userName || owner.toUpperCase() !== viewer.userName.toUpperCase()) {
      throw forbidden("Only the graph owner can change this graph");
    }
  }
}

export function createSnowflakeRuntime(viewer?: Viewer): SnowflakeRuntime {
  const env = appEnv();
  return new SnowflakeRuntime(
    env.repositoryRoot,
    env.stateRoot,
    env.stubRuntimes || !env.snowflakeHosted,
    viewer,
  );
}

