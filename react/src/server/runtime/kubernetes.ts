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
  hiddenRunIds,
  hideRun,
  graphName,
  renameGraph as persistGraphName,
  snapshotFromRecord,
  removeRunDirectory,
  cloneFieldsFromRequest,
} from "../catalog";
import { lastJsonObject, runFlakegraph } from "../cli";
import { buildRunConfig, environmentForRequest, redactedConfig, writeRunConfig } from "../config";
import { appEnv } from "../env";
import {
  documentCountsByRun,
  documentTasksByRun,
  finalizerPayload,
  graphVersionsByGraph,
  leasedTasks,
  listPostgresRuns,
} from "../db/client";
import { documentStatusesFromEvents, documentStatusesFromTasks, type DocumentStatus } from "../documents";
import { readLocalProgress } from "../progress";
import {
  KUBERNETES_CAPABILITIES,
  type Capability,
  type ClusterProfile,
  type ClusterSnapshot,
  type GraphDataset,
  type GraphShare,
  type GraphVersion,
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
import { graphArtifactsExist, loadLocalGraph } from "../graph";
import { readFleetProfile, type FleetProfile } from "../fleet";
import { snapshotFromStatus } from "../fleet-status";
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
        await writeRunConfig(request, configPath, await this.fleetProfile());
        const result = await runFlakegraph(
          ["fleet", "preflight", "--config", configPath, ...(await this.fleetArguments())],
          { cwd: this.repositoryRoot, env: environmentForRequest(request) },
        );
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
        const revision = request.revision;
        // A revision that adds nothing has no source to discover; the config
        // still needs one to load, so it names the run's own directory, which
        // the planner is told not to read.
        const configured =
          revision && !revision.addDocuments
            ? { ...request, sourceKind: "local_path" as const, source: { kind: "local", path: directory } }
            : request;
        const configPath = await writeRunConfig(
          configured,
          path.join(directory, "config.yaml"),
          this.stubbed ? null : await this.fleetProfile(),
        );
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
            baseRunId: revision?.baseRunId ?? null,
            ...cloneFieldsFromRequest(request),
          });
          await appendStubRun(this.stateRoot, {
            runId: request.jobId,
            graphId: request.graphId,
            status: "queued",
          });
        } else {
          const result = await runFlakegraph(
            revision
              ? [
                  "distributed",
                  "revise",
                  "--config",
                  configPath,
                  "--run-id",
                  request.jobId,
                  "--base-run",
                  revision.baseRunId,
                  ...revision.dropFileIds.flatMap((fileId) => ["--drop-file", fileId]),
                  ...(revision.addDocuments ? [] : ["--no-add-documents"]),
                ]
              : ["distributed", "submit", "--config", configPath, "--run-id", request.jobId],
            { cwd: this.repositoryRoot, env: environmentForRequest(configured) },
          );
          if (revision && result.exitCode === 2) {
            // The planner refused before creating anything: nothing to record.
            throw invalid(result.stderr.trim() || "The revision was refused");
          }
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
            baseRunId: revision?.baseRunId ?? null,
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
        const records = new Map(
          catalog
            .filter(({ record }) => String(record.runtime || "") === "kubernetes")
            .map(({ record }) => [record.runId, record] as const),
        );
        const local = await Promise.all(
          [...records.values()].map(async (record) =>
            snapshotFromRecord(record, {
              graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
            }),
          ),
        );
        if (this.stubbed) {
          return local;
        }
        // The coordination store lists every run the fleet holds, including
        // those submitted from elsewhere; the catalog only adds what this
        // console knows about them. Without the store, the catalog is shown
        // with a warning rather than nothing.
        let rows;
        try {
          rows = await listPostgresRuns(limit);
        } catch (error) {
          const reason = error instanceof Error ? error.message : "coordination store unavailable";
          return local.map((snapshot) => ({
            ...snapshot,
            listingWarning: `Showing this console's own record of fleet runs, not the fleet's: ${reason}.`,
          }));
        }
        const hidden = await hiddenRunIds(this.stateRoot);
        const counts = await documentCountsByRun(rows.map((row) => row.id).filter((id) => !hidden.has(id)));
        const versions = await graphVersionsByGraph([...new Set(rows.map((row) => row.graphId))]);
        const seen = new Set<string>();
        const merged: RunSnapshot[] = [];
        for (const row of rows) {
          if (hidden.has(row.id)) {
            continue;
          }
          seen.add(row.id);
          const record = records.get(row.id);
          const name = await graphName(this.stateRoot, row.graphId);
          const documents = counts.get(row.id) ?? { total: null, completed: 0, failed: 0 };
          const snapshot = record
            ? snapshotFromRecord(
                { ...record, status: row.status, updatedAt: row.updatedAt?.toISOString() ?? record.updatedAt },
                { graphName: name ?? record.graphName ?? null },
              )
            : this.snapshotFromRow(row, name);
          merged.push({
            ...snapshot,
            documentsTotal: documents.total,
            documentsCompleted: documents.completed,
            documentsFailed: documents.failed,
            raw: { ...snapshot.raw, version: versionOf(versions.get(row.graphId) ?? [], row.id) },
          });
        }
        for (const snapshot of local) {
          if (!seen.has(snapshot.runId)) {
            merged.push(snapshot);
          }
        }
        return merged;
      },
      catch: (cause) => fromCause(cause, "Unable to list fleet runs"),
    });
  }

  private snapshotFromRow(
    row: { id: string; graphId: string; status: string; createdAt: Date | null; updatedAt: Date | null; errorJson: unknown },
    graphName: string | null,
  ): RunSnapshot {
    const outputPath = this.defaultOutputPath(row.id);
    return {
      runId: row.id,
      graphId: row.graphId,
      status: row.status,
      startedAt: row.createdAt?.toISOString() ?? null,
      updatedAt: row.updatedAt?.toISOString() ?? null,
      graphName,
      stages: [],
      events: [],
      documentsTotal: null,
      documentsCompleted: 0,
      documentsFailed: 0,
      outputPath,
      storageKind: "local_files",
      storageLocation: outputPath,
      warnings: [],
      error: row.errorJson ? JSON.stringify(row.errorJson) : null,
      listingWarning: null,
      raw: {},
    };
  }

  /** Where a fleet run's graph is materialised on first view: under the console's own state. */
  private defaultOutputPath(runId: string): string {
    return path.join(this.stateRoot, "artifacts", runId);
  }

  /** The record the console keeps for a run, or null for one it only knows through the fleet. */
  private async record(runId: string) {
    const directory = runDirectory(this.stateRoot, runId);
    return runRecordExists(directory) ? readRunRecord(directory) : null;
  }

  /** `distributed status` for one run, read with the run's own configuration when the console has it. */
  private async status(runId: string, configPath: string | null | undefined): Promise<Record<string, unknown> | null> {
    const args = ["distributed", "status", "--run-id", runId];
    if (configPath) {
      args.push("--config", configPath);
    }
    const result = await runFlakegraph(args, { cwd: this.repositoryRoot });
    return lastJsonObject(result.stdout);
  }

  getRun(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const record = await this.record(runId);
        if (this.stubbed) {
          if (!record) {
            throw notFound(`Unknown Kubernetes run: ${runId}`);
          }
          return snapshotFromRecord(record, {
            graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
          });
        }
        const payload = await this.status(runId, record?.configPath);
        if (!payload?.run) {
          if (!record) {
            throw notFound(`Unknown Kubernetes run: ${runId}`);
          }
          return snapshotFromRecord(record, {
            graphName: (await graphName(this.stateRoot, record.graphId)) ?? record.graphName ?? null,
            listingWarning: "The fleet's coordination store could not be read; this is the console's own record.",
          });
        }
        const run = payload.run as Record<string, unknown>;
        const graphId = String(run.graph_id ?? record?.graphId ?? runId);
        const name = (await graphName(this.stateRoot, graphId)) ?? record?.graphName ?? null;
        const base = record
          ? snapshotFromRecord(record, { graphName: name })
          : this.snapshotFromRow(
              { id: runId, graphId, status: String(run.status ?? "unknown"), createdAt: null, updatedAt: null, errorJson: null },
              name,
            );
        const snapshot = snapshotFromStatus(payload, base);
        if (record && record.status !== snapshot.status) {
          await writeRunRecord(runDirectory(this.stateRoot, runId), { status: snapshot.status });
        }
        const versions = (await graphVersionsByGraph([graphId])).get(graphId) ?? [];
        return { ...snapshot, raw: { ...snapshot.raw, version: versionOf(versions, runId) } };
      },
      catch: (cause) => fromCause(cause, "Unable to load Kubernetes run"),
    });
  }

  documents(runId: string): Effect.Effect<readonly DocumentStatus[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (this.stubbed) {
          // The stub keeps a run's progress beside its record, as a local run does.
          const progress = await readLocalProgress(path.join(runDirectory(this.stateRoot, runId), "events.jsonl"));
          return documentStatusesFromEvents(progress.events);
        }
        const own = documentStatusesFromTasks(await documentTasksByRun(runId));
        // A revision's kept documents live in the runs that processed them;
        // the finalizer's payload says which, and they are shown as kept.
        const inherited: DocumentStatus[] = [];
        const entries = inheritedDocuments(await finalizerPayload(runId));
        if (entries.length) {
          const graphId = (await this.record(runId))?.graphId ?? null;
          const versions = graphId ? ((await graphVersionsByGraph([graphId])).get(graphId) ?? []) : [];
          for (const entry of entries) {
            const kept = new Set(entry.fileIds);
            const number = versions.findIndex((version) => version.runId === entry.runId) + 1;
            const origin = number ? `version ${number}` : entry.runId;
            for (const status of documentStatusesFromTasks(await documentTasksByRun(entry.runId))) {
              if (kept.has(status.fileId)) {
                inherited.push({ ...status, phase: "inherited", detail: `Kept from ${origin}` });
              }
            }
          }
        }
        return [...own, ...inherited].sort((left, right) =>
          (left.name ?? left.fileId).localeCompare(right.name ?? right.fileId),
        );
      },
      catch: (cause) => fromCause(cause, "Unable to list the run's documents"),
    });
  }

  cancel(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const record = await this.record(runId);
        if (!record && this.stubbed) {
          throw notFound(`Unknown Kubernetes run: ${runId}`);
        }
        if (!this.stubbed) {
          await this.control("cancel", runId, record?.configPath);
        }
        if (record) {
          await writeRunRecord(runDirectory(this.stateRoot, runId), {
            status: "cancelled",
            cancellationRequestedAt: new Date().toISOString(),
          });
        }
        return Effect.runPromise(this.getRun(runId));
      },
      catch: (cause) => fromCause(cause, "Unable to cancel Kubernetes run"),
    });
  }

  retry(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const record = await this.record(runId);
        if (!record && this.stubbed) {
          throw notFound(`Unknown Kubernetes run: ${runId}`);
        }
        if (!this.stubbed) {
          await this.control("retry", runId, record?.configPath);
        }
        if (record) {
          await writeRunRecord(runDirectory(this.stateRoot, runId), {
            status: "queued",
            error: null,
            cancellationRequestedAt: null,
          });
        }
        return Effect.runPromise(this.getRun(runId));
      },
      catch: (cause) => fromCause(cause, "Unable to retry Kubernetes run"),
    });
  }

  versions(graphId: string): Effect.Effect<readonly GraphVersion[], ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        if (this.stubbed) {
          return [];
        }
        const rows = (await graphVersionsByGraph([graphId])).get(graphId) ?? [];
        return rows.map((row, index) => ({
          graphId,
          runId: row.runId,
          number: index + 1,
          createdAt: row.createdAt.toISOString(),
          head: row.head,
        }));
      },
      catch: (cause) => fromCause(cause, "Unable to list the graph's versions"),
    });
  }

  recover(runId: string): Effect.Effect<string, ControlPlaneError> {
    return Effect.tryPromise({
      try: async () => {
        const record = await this.record(runId);
        if (this.stubbed) {
          if (!record) {
            throw notFound(`Unknown Kubernetes run: ${runId}`);
          }
          await writeRunRecord(runDirectory(this.stateRoot, runId), { status: "queued", cancellationRequestedAt: null });
          return `Reconciled infrastructure for ${runId}`;
        }
        // The CLI restarts only a pool with no available worker, after checking
        // for the failures a restart cannot fix; task state is never touched.
        const args = ["fleet", "recover", "--run-id", runId, ...(await this.fleetArguments())];
        if (record?.configPath) {
          args.push("--config", record.configPath);
        }
        const result = await runFlakegraph(args, { cwd: this.repositoryRoot });
        const payload = lastJsonObject(result.stdout);
        if (result.exitCode !== 0 || !payload) {
          throw new Error(lastLine(result.stderr) || `Unable to recover the workers of ${runId}`);
        }
        return String(payload.message ?? `Reconciled infrastructure for ${runId}`);
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
        // The fleet keeps the run and its graph; the console stops listing it.
        await hideRun(this.stateRoot, runId);
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
        const directory = this.local.artifactDirectory(snapshot.outputPath);
        // The fleet publishes its graph to the artifact store; the local copy
        // is materialised once, by the CLI, then read like any other graph.
        if (!this.stubbed && !graphArtifactsExist(directory)) {
          const record = await this.record(snapshot.runId);
          const args = ["distributed", "export", "--run-id", snapshot.runId, "--output", directory];
          if (record?.configPath) {
            args.push("--config", record.configPath);
          }
          const result = await runFlakegraph(args, { cwd: this.repositoryRoot });
          if (result.exitCode !== 0) {
            throw new Error(result.stderr.trim() || `Exporting the graph of ${snapshot.runId} failed`);
          }
        }
        return loadLocalGraph(directory);
      },
      catch: (cause) => fromCause(cause, "Unable to load fleet graph"),
    });
  }

  private async control(action: "cancel" | "retry", runId: string, configPath: string | null | undefined): Promise<void> {
    const args = ["distributed", action, "--run-id", runId];
    if (configPath) {
      args.push("--config", configPath);
    }
    const result = await runFlakegraph(args, { cwd: this.repositoryRoot });
    if (result.exitCode !== 0) {
      throw new Error(result.stderr.trim() || `Unable to ${action} ${runId}`);
    }
  }

  /** What the fleet's workers run, or null when the fleet cannot be read. */
  async fleetProfile(): Promise<FleetProfile | null> {
    if (this.stubbed) {
      return null;
    }
    const cluster = await Effect.runPromise(this.selectedCluster());
    return readFleetProfile(cluster?.namespace || appEnv().kubernetesNamespace, {
      cwd: this.repositoryRoot,
      context: cluster?.context || null,
    });
  }

  private async fleetArguments(): Promise<string[]> {
    const cluster = await Effect.runPromise(this.selectedCluster());
    const args = ["--namespace", cluster?.namespace || appEnv().kubernetesNamespace];
    if (cluster?.context) {
      args.push("--context", cluster.context);
    }
    return args;
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
        if (this.stubbed) {
          const snapshot = await readJsonFile<{ assignments?: NodeWorkAssignment[] }>(
            path.join(this.stateRoot, "kubernetes", "assignments.json"),
          );
          return (snapshot?.assignments ?? []).filter(
            (item) => item.scopeId === nodeName || item.workerId.includes(nodeName),
          );
        }
        // A lease names the worker pod that holds it; the cluster says which
        // node that pod runs on.
        const cluster = await Effect.runPromise(this.cluster(namespace));
        const podsOnNode = new Set(
          (cluster?.workloads ?? []).filter((workload) => workload.node === nodeName).map((workload) => workload.name),
        );
        if (podsOnNode.size === 0) {
          return [];
        }
        return (await leasedTasks())
          .filter((row) => podsOnNode.has(row.leaseOwner))
          .map((row) => ({
            workerId: row.leaseOwner,
            runId: row.runId,
            graphId: row.graphId,
            taskId: row.taskId,
            stage: row.stage,
            scopeId: row.scopeId,
            updatedAt: row.updatedAt?.toISOString() ?? null,
          }));
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
      try: async () =>
        stringifyYaml(redactedConfig(await buildRunConfig(request, await this.fleetProfile())), {
          sortMapEntries: false,
        }),
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
      image: engineContainer(pod)?.image ?? pod.spec?.containers?.[0]?.image ?? null,
      model: servedModel(pod),
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

type PodLike = {
  metadata?: { labels?: Record<string, string> };
  spec?: { containers?: Array<{ name?: string; image?: string; args?: string[] }> };
};

/** The container that runs the model, on a pod that serves one. */
function engineContainer(pod: PodLike) {
  const containers = pod.spec?.containers ?? [];
  return containers.find((container) => container.name === "vllm") ?? null;
}

/**
 * The model a serving pod answers for: the `--served-model-name` it was
 * started with, else the checkpoint after `serve`. Nothing on the pod's
 * labels says it, so the arguments do.
 */
function servedModel(pod: PodLike): string | null {
  const labelled = pod.metadata?.labels?.["flakegraph/model"];
  if (labelled) {
    return labelled;
  }
  const args = engineContainer(pod)?.args ?? [];
  const named = args.indexOf("--served-model-name");
  if (named >= 0 && args[named + 1]) {
    return args[named + 1] ?? null;
  }
  const serve = args.indexOf("serve");
  if (serve >= 0 && args[serve + 1] && !args[serve + 1]!.startsWith("-")) {
    return args[serve + 1] ?? null;
  }
  return null;
}

/**
 * Where a run stands among its graph's published versions: its number, how
 * many there are, and whether the head points at it. Null for a run that has
 * not published one, whose graph may still have versions from other runs.
 */
function versionOf(
  versions: readonly { runId: string; head: boolean }[],
  runId: string,
): { number: number; count: number; head: boolean } | null {
  const index = versions.findIndex((version) => version.runId === runId);
  if (index < 0) {
    return versions.length ? { number: 0, count: versions.length, head: false } : null;
  }
  return { number: index + 1, count: versions.length, head: versions[index]!.head };
}

/** The documents a finalizer's payload says it kept, per run they came from. */
function inheritedDocuments(payload: Record<string, unknown> | null): Array<{ runId: string; fileIds: string[] }> {
  const entries = payload?.inherit;
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries.flatMap((entry) => {
    if (!entry || typeof entry !== "object") {
      return [];
    }
    const record = entry as Record<string, unknown>;
    const fileIds = Array.isArray(record.file_ids) ? record.file_ids.map(String) : [];
    return typeof record.run_id === "string" && fileIds.length ? [{ runId: record.run_id, fileIds }] : [];
  });
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

// A CLI failure ends with its cause; the traceback above it is not for the page.
function lastLine(text: string): string {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  return lines[lines.length - 1] ?? "";
}

function asStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item)) : [];
}

