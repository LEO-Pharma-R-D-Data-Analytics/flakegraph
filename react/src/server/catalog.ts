import { mkdir, readdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { validateGraphName, type IngestionRequest, type RunSnapshot, type StorageKind } from "./protocol/schema";
import { assertSafeId } from "./confine";

/**
 * Runs the catalog does not list: revision attempts the planner refused,
 * which it may already have recorded as cancelled rows. They are not
 * versions of their graph. (Removing a graph is a delete, not a hiding.)
 */
const WITHDRAWN_RUNS_FILE = "withdrawn-runs.json";
const GRAPH_NAMES_FILE = "graph-names.json";

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
  sourceKind?: string | null;
  sourcePath?: string | null;
  source?: Record<string, unknown> | null;
  ocrProvider?: string | null;
  llmProvider?: string | null;
  embeddingProvider?: string | null;
  owner?: string | null;
  /** The run this one revised, when it is a new version of that run's graph. */
  baseRunId?: string | null;
  [key: string]: unknown;
}

export function cloneFieldsFromRequest(request: IngestionRequest) {
  const source = request.source as Record<string, unknown>;
  return {
    sourceKind: request.sourceKind,
    sourcePath: String(source.path ?? source.prefix ?? source.stage ?? ""),
    // The whole source, so a bucket or container can be run again as it
    // was named: kind, bucket, prefix, endpoint. Credentials never travel
    // in it - they come from the environment.
    source: { ...source },
    ocrProvider: request.ocr.provider,
    llmProvider: request.llm.provider,
    embeddingProvider: request.embedding.provider,
  };
}

/**
 * Where a run's record, configuration and events live. The id is checked
 * here as well as at the API edge: every runtime names the directory
 * through this one function, so nothing a caller sends can name a
 * directory anywhere else.
 */
export function runDirectory(stateRoot: string, runId: string): string {
  return path.join(stateRoot, "runs", assertSafeId(runId, "runId"));
}

export async function writeRunRecord(
  directory: string,
  values: Partial<CatalogRecord> & Record<string, unknown>,
): Promise<CatalogRecord> {
  await mkdir(directory, { recursive: true });
  const current = await readRunRecord(directory);
  const overrides = Object.fromEntries(Object.entries(values).filter(([, value]) => value !== undefined));
  const payload: CatalogRecord = {
    ...current,
    ...overrides,
    runId: String(overrides.runId ?? current.runId ?? path.basename(directory)),
    graphId: String(overrides.graphId ?? current.graphId ?? path.basename(directory)),
    status: String(overrides.status ?? current.status ?? "unknown"),
    runtime: String(overrides.runtime ?? current.runtime ?? "local"),
  };
  if (payload.cancellationRequestedAt) {
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
  options: { includeWithdrawn?: boolean } = {},
): Promise<Array<{ directory: string; record: CatalogRecord }>> {
  const runsRoot = path.join(stateRoot, "runs");
  if (!existsSync(runsRoot)) {
    return [];
  }
  const withdrawn = options.includeWithdrawn ? new Set<string>() : await withdrawnRunIds(stateRoot);
  const entries = await readdir(runsRoot, { withFileTypes: true });
  // A run's directory is named by its id, so withdrawn runs are left out before
  // the list is cut to size rather than taking places in it.
  const directories = entries
    .filter((entry) => entry.isDirectory() && !withdrawn.has(entry.name))
    .map((entry) => path.join(runsRoot, entry.name));
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
  ).filter((item) => Boolean(item.record.runId) && !withdrawn.has(item.record.runId));
}

export async function withdrawnRunIds(stateRoot: string): Promise<Set<string>> {
  const value = await readJsonFile<unknown>(path.join(stateRoot, WITHDRAWN_RUNS_FILE));
  if (!Array.isArray(value)) {
    return new Set();
  }
  return new Set(value.map((item) => String(item)).filter(Boolean));
}

/** Leave a refused revision attempt out of the catalog. */
export async function withdrawRun(stateRoot: string, runId: string): Promise<void> {
  const normalized = normalizeRunIds([runId]);
  await changeWithdrawn(stateRoot, (withdrawn) => normalized.forEach((id) => withdrawn.add(id)));
}

/** Forget withdrawn runs that no longer exist, once their graph is deleted. */
export async function unwithdrawRuns(stateRoot: string, runIds: readonly string[]): Promise<void> {
  if (runIds.length === 0) {
    return;
  }
  const normalized = normalizeRunIds(runIds);
  await changeWithdrawn(stateRoot, (withdrawn) => normalized.forEach((id) => withdrawn.delete(id)));
}

function normalizeRunIds(runIds: readonly string[]): string[] {
  const normalized = [...new Set(runIds.map((runId) => runId.trim()).filter(Boolean))];
  if (normalized.length === 0) {
    throw new Error("run ID must not be empty");
  }
  return normalized;
}

const withdrawnQueues = new Map<string, Promise<unknown>>();

/** One change to the withdrawn list at a time, so concurrent changes cannot lose each other. */
async function changeWithdrawn(stateRoot: string, change: (withdrawn: Set<string>) => void): Promise<void> {
  const file = path.join(stateRoot, WITHDRAWN_RUNS_FILE);
  const next = (withdrawnQueues.get(file) ?? Promise.resolve())
    .catch(() => undefined)
    .then(async () => {
      await mkdir(stateRoot, { recursive: true });
      const withdrawn = await withdrawnRunIds(stateRoot);
      change(withdrawn);
      await atomicWriteJson(file, [...withdrawn].sort());
    });
  withdrawnQueues.set(file, next);
  try {
    await next;
  } finally {
    if (withdrawnQueues.get(file) === next) {
      withdrawnQueues.delete(file);
    }
  }
}

export function runRecordExists(directory: string): boolean {
  return existsSync(path.join(directory, "run.json"));
}

export async function readGraphNames(stateRoot: string): Promise<Record<string, string>> {
  const value = await readJsonFile<Record<string, unknown>>(path.join(stateRoot, GRAPH_NAMES_FILE));
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(value).flatMap(([key, item]) => (typeof item === "string" ? [[key, item] as const] : [])),
  );
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
  const rows = await listRunRecords(stateRoot, 1_000, { includeWithdrawn: true });
  for (const { directory, record } of rows) {
    if (record.graphId === normalizedId) {
      await writeRunRecord(directory, { graphName: normalizedName });
    }
  }
  return normalizedName;
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
      ...((extras.raw as Record<string, unknown> | undefined) ?? {}),
      config_path: record.configPath ?? null,
      nodeCount: record.nodeCount ?? (extras.raw as Record<string, unknown> | undefined)?.nodeCount ?? null,
      sourceKind: record.sourceKind ?? null,
      sourcePath: record.sourcePath ?? null,
      source: record.source ?? (extras.raw as Record<string, unknown> | undefined)?.source ?? null,
      ocrProvider: record.ocrProvider ?? null,
      llmProvider: record.llmProvider ?? null,
      embeddingProvider: record.embeddingProvider ?? null,
      owner: record.owner ?? (extras.raw as Record<string, unknown> | undefined)?.owner ?? null,
      baseRunId: record.baseRunId ?? (extras.raw as Record<string, unknown> | undefined)?.baseRunId ?? null,
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
    sourceKind: (value.sourceKind ?? value.source_kind ?? null) as string | null,
    sourcePath: (value.sourcePath ?? value.source_path ?? null) as string | null,
    source: (value.source && typeof value.source === "object" ? value.source : null) as Record<string, unknown> | null,
    ocrProvider: (value.ocrProvider ?? value.ocr_provider ?? null) as string | null,
    llmProvider: (value.llmProvider ?? value.llm_provider ?? null) as string | null,
    embeddingProvider: (value.embeddingProvider ?? value.embedding_provider ?? null) as string | null,
    owner: (value.owner ?? null) as string | null,
  };
}
