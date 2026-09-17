import { mkdir, readdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { validateGraphName, type RunSnapshot, type StorageKind } from "./protocol/schema";

const HIDDEN_RUNS_FILE = "hidden-runs.json";
const GRAPH_NAMES_FILE = "graph-names.json";
const CLUSTERS_FILE = "clusters.json";
const SELECTED_CLUSTER_FILE = "selected-cluster.json";

export interface CatalogRecord {
  runId: string;
  graphId: string;
  graphName?: string | null;
  status: string;
  runtime: string;
  startedAt?: string | null;
  updatedAt?: string | null;
  outputPath?: string | null;
  configPath?: string | null;
  storageKind?: StorageKind | null;
  storageLocation?: string | null;
  error?: string | null;
  pid?: number | null;
  cancellationRequestedAt?: string | null;
  documentsTotal?: number | null;
  documentsCompleted?: number | null;
  documentsFailed?: number | null;
  [key: string]: unknown;
}

export function runDirectory(stateRoot: string, runId: string): string {
  return path.join(stateRoot, "runs", runId);
}

export async function writeRunRecord(
  directory: string,
  values: Partial<CatalogRecord> & Record<string, unknown>,
): Promise<CatalogRecord> {
  await mkdir(directory, { recursive: true });
  const current = await readRunRecord(directory);
  const payload: CatalogRecord = {
    runId: current.runId || path.basename(directory),
    graphId: current.graphId || path.basename(directory),
    status: current.status || "unknown",
    runtime: current.runtime || "local",
    ...current,
    ...Object.fromEntries(Object.entries(values).filter(([, value]) => value !== undefined)),
  };
  if (current.cancellationRequestedAt || payload.cancellationRequestedAt) {
    payload.status = "cancelled";
  }
  await atomicWriteJson(path.join(directory, "run.json"), payload);
  return payload;
}

export async function readRunRecord(directory: string): Promise<CatalogRecord> {
  const file = path.join(directory, "run.json");
  const value = await readJsonFile<Record<string, unknown>>(file);
  if (!value) {
    return emptyRecord(path.basename(directory));
  }
  return normalizeRecord(value, path.basename(directory));
}

export async function listRunRecords(
  stateRoot: string,
  limit = 100,
): Promise<Array<{ directory: string; record: CatalogRecord }>> {
  const runsRoot = path.join(stateRoot, "runs");
  if (!existsSync(runsRoot)) {
    return [];
  }
  const entries = await readdir(runsRoot, { withFileTypes: true });
  const directories = entries.filter((entry) => entry.isDirectory()).map((entry) => path.join(runsRoot, entry.name));
  const withTimes = await Promise.all(
    directories.map(async (directory) => {
      const file = path.join(directory, "run.json");
      try {
        const info = await stat(file);
        return { directory, mtime: info.mtimeMs };
      } catch {
        return null;
      }
    }),
  );
  return (
    await Promise.all(
      withTimes
        .filter((entry): entry is { directory: string; mtime: number } => entry !== null)
        .sort((left, right) => right.mtime - left.mtime)
        .slice(0, limit)
        .map(async ({ directory }) => ({ directory, record: await readRunRecord(directory) })),
    )
  ).filter((item) => Boolean(item.record.runId));
}

export async function hiddenRunIds(stateRoot: string): Promise<Set<string>> {
  const value = await readJsonFile<unknown>(path.join(stateRoot, HIDDEN_RUNS_FILE));
  if (!Array.isArray(value)) {
    return new Set();
  }
  return new Set(value.map((item) => String(item)).filter(Boolean));
}

export async function hideRun(stateRoot: string, runId: string): Promise<void> {
  const normalized = runId.trim();
  if (!normalized) {
    throw new Error("run ID must not be empty");
  }
  await mkdir(stateRoot, { recursive: true });
  const hidden = [...(await hiddenRunIds(stateRoot)), normalized].sort();
  await atomicWriteJson(path.join(stateRoot, HIDDEN_RUNS_FILE), hidden);
}

export async function graphName(stateRoot: string, graphId: string): Promise<string | null> {
  const names = await readGraphNames(stateRoot);
  return names[graphId] ?? null;
}

export async function renameGraph(
  stateRoot: string,
  graphId: string,
  displayName: string,
): Promise<string> {
  const normalizedId = graphId.trim();
  if (!normalizedId) {
    throw new Error("graph ID must not be empty");
  }
  const normalizedName = validateGraphName(displayName);
  await mkdir(stateRoot, { recursive: true });
  const names = await readGraphNames(stateRoot);
  names[normalizedId] = normalizedName;
  await atomicWriteJson(path.join(stateRoot, GRAPH_NAMES_FILE), names);
  return normalizedName;
}

export async function removeRunDirectory(stateRoot: string, runId: string): Promise<void> {
  const directory = path.resolve(runDirectory(stateRoot, runId));
  const parent = path.resolve(path.join(stateRoot, "runs"));
  if (path.dirname(directory) !== parent) {
    throw new Error("Refusing to remove a run outside the app state directory");
  }
  await rm(directory, { recursive: true, force: true });
}

export interface ClusterCatalog {
  clusters: Array<{
    name: string;
    namespace: string;
    context: string;
    kubeconfig: string;
    description: string;
  }>;
  selected?: string | null;
}

export async function readClusterCatalog(stateRoot: string): Promise<ClusterCatalog> {
  const value = await readJsonFile<ClusterCatalog>(path.join(stateRoot, CLUSTERS_FILE));
  if (!value || !Array.isArray(value.clusters)) {
    return { clusters: [] };
  }
  const selected = await readJsonFile<{ name?: string }>(path.join(stateRoot, SELECTED_CLUSTER_FILE));
  return {
    clusters: value.clusters.map((cluster) => ({
      name: String(cluster.name),
      namespace: String(cluster.namespace || "flakegraph"),
      context: String(cluster.context || ""),
      kubeconfig: String(cluster.kubeconfig || ""),
      description: String(cluster.description || ""),
    })),
    selected: selected?.name ?? value.selected ?? null,
  };
}

export async function writeClusterCatalog(stateRoot: string, catalog: ClusterCatalog): Promise<void> {
  await mkdir(stateRoot, { recursive: true });
  await atomicWriteJson(path.join(stateRoot, CLUSTERS_FILE), {
    clusters: catalog.clusters,
  });
  if (catalog.selected) {
    await atomicWriteJson(path.join(stateRoot, SELECTED_CLUSTER_FILE), { name: catalog.selected });
  }
}

export async function atomicWriteJson(file: string, value: unknown): Promise<void> {
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = path.join(path.dirname(file), `.${path.basename(file)}.${randomUUID()}.tmp`);
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(temporary, file);
}

export async function readJsonFile<T>(file: string): Promise<T | null> {
  try {
    const text = await readFile(file, "utf8");
    return JSON.parse(text) as T;
  } catch {
    return null;
  }
}

export function snapshotFromRecord(
  record: CatalogRecord,
  extras: Partial<RunSnapshot> = {},
): RunSnapshot {
  return {
    runId: record.runId,
    graphId: record.graphId,
    status: record.status,
    startedAt: record.startedAt ?? null,
    updatedAt: record.updatedAt ?? record.startedAt ?? null,
    graphName: (record.graphName as string | null | undefined) ?? extras.graphName ?? null,
    stages: extras.stages ?? [],
    events: extras.events ?? [],
    documentsTotal: record.documentsTotal ?? extras.documentsTotal ?? null,
    documentsCompleted: record.documentsCompleted ?? extras.documentsCompleted ?? 0,
    documentsFailed: record.documentsFailed ?? extras.documentsFailed ?? 0,
    outputPath: record.outputPath ?? null,
    storageKind: record.storageKind ?? "local_files",
    storageLocation: record.storageLocation ?? record.outputPath ?? null,
    warnings: extras.warnings ?? [],
    error: record.error ?? extras.error ?? null,
    listingWarning: extras.listingWarning ?? null,
    raw: {
      config_path: record.configPath ?? null,
      ...((extras.raw as Record<string, unknown> | undefined) ?? {}),
    },
  };
}

function emptyRecord(runId: string): CatalogRecord {
  return { runId, graphId: runId, status: "unknown", runtime: "local" };
}

function normalizeRecord(value: Record<string, unknown>, fallbackId: string): CatalogRecord {
  return {
    ...value,
    runId: String(value.runId ?? value.run_id ?? fallbackId),
    graphId: String(value.graphId ?? value.graph_id ?? fallbackId),
    graphName: (value.graphName ?? value.graph_name ?? null) as string | null,
    status: String(value.status ?? "unknown"),
    runtime: String(value.runtime ?? "local"),
    startedAt: (value.startedAt ?? value.started_at ?? null) as string | null,
    updatedAt: (value.updatedAt ?? value.updated_at ?? null) as string | null,
    outputPath: (value.outputPath ?? value.output_path ?? null) as string | null,
    configPath: (value.configPath ?? value.config_path ?? null) as string | null,
    storageKind: ((value.storageKind ?? value.storage_kind ?? "local_files") as StorageKind),
    storageLocation: (value.storageLocation ?? value.storage_location ?? null) as string | null,
    error: (value.error ?? null) as string | null,
    pid: (value.pid ?? null) as number | null,
    cancellationRequestedAt: (value.cancellationRequestedAt ??
      value.cancellation_requested_at ??
      null) as string | null,
    documentsTotal: (value.documentsTotal ?? value.documents_total ?? null) as number | null,
    documentsCompleted: (value.documentsCompleted ?? value.documents_completed ?? 0) as number,
    documentsFailed: (value.documentsFailed ?? value.documents_failed ?? 0) as number,
  };
}
