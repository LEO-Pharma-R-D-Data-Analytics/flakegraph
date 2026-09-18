import { Either, Schema } from "effect";

export const DEFAULT_PROVIDER_PARALLELISM = 12;
export const MAX_GRAPH_NAME_LENGTH = 120;
export const GRAPH_REVIEW_ROW_LIMITS = {
  nodes: 2_500,
  edges: 5_000,
  communities: 1_000,
  evidence: 5_000,
} as const;

export const DISTRIBUTED_STAGE_ORDER = [
  "prepare_document",
  "extract_document_context",
  "extract_entity_window",
  "compact_entity_inventory",
  "extract_relation_window",
  "compact_document",
  "finalize_graph",
] as const;

export const PIPELINE_STAGE_ORDER = [
  "file_source",
  "ocr",
  "chunking",
  "chunk_embeddings",
  "graph_extraction",
  "filtering",
  "merge",
  "graph_embeddings",
  "community_detection",
  "quality",
  "write",
] as const;

export const RUN_STAGE_ORDER = [
  ...DISTRIBUTED_STAGE_ORDER,
  ...PIPELINE_STAGE_ORDER,
] as const;

export const ARTIFACTS_UNAVAILABLE_STATUS = "unavailable";
export const SUCCESS_STATUSES = ["succeeded", "completed", "done", "success"] as const;
export const ACTIVE_STATUSES = [
  "pending",
  "planning",
  "queued",
  "running",
  "processing",
  "claimed",
  "submitted",
  "cancelling",
] as const;

export const RuntimeMode = Schema.Literal("local", "kubernetes", "snowflake");
export type RuntimeMode = Schema.Schema.Type<typeof RuntimeMode>;

export const SourceKind = Schema.Literal(
  "upload",
  "local_path",
  "azure_blob",
  "s3",
  "snowflake_stage",
);
export type SourceKind = Schema.Schema.Type<typeof SourceKind>;

export const StorageKind = Schema.Literal("local_files", "snowflake");
export type StorageKind = Schema.Schema.Type<typeof StorageKind>;

export const Capability = Schema.Literal(
  "upload",
  "local",
  "azure_blob",
  "s3",
  "snowflake_stage",
  "progress",
  "graph",
  "forget",
  "rename",
  "distributed",
  "cancel",
  "retry",
  "recover",
  "cluster",
  "node_assignments",
  "share",
  "delete_graph",
  "spcs",
  "model_serving",
);
export type Capability = Schema.Schema.Type<typeof Capability>;

export const JsonRecord = Schema.Record({
  key: Schema.String,
  value: Schema.Unknown,
});

export const Viewer = Schema.Struct({
  userName: Schema.String,
  email: Schema.optionalWith(Schema.String, { default: () => "" }),
  roles: Schema.optionalWith(Schema.Array(Schema.String), {
    default: () => [] as string[],
  }),
});
export type Viewer = Schema.Schema.Type<typeof Viewer>;

export const SourceObject = Schema.Struct({
  uri: Schema.String,
  name: Schema.String,
  sizeBytes: Schema.Number,
  modifiedAt: Schema.NullOr(Schema.String),
  checksum: Schema.NullOr(Schema.String),
});
export type SourceObject = Schema.Schema.Type<typeof SourceObject>;

export const ProviderSelection = Schema.Struct({
  provider: Schema.String,
  model: Schema.NullOr(Schema.String),
  endpoint: Schema.NullOr(Schema.String),
  apiKeyEnvironmentVariable: Schema.NullOr(Schema.String),
  dimension: Schema.NullOr(Schema.Number.pipe(Schema.positive())),
  options: Schema.optionalWith(JsonRecord, { default: () => ({}) }),
});
export type ProviderSelection = Schema.Schema.Type<typeof ProviderSelection>;

export const SnowflakeOutput = Schema.Struct({
  account: Schema.String,
  user: Schema.String,
  database: Schema.String,
  schema: Schema.String,
  warehouse: Schema.String,
  bulkStage: Schema.String,
  role: Schema.NullOr(Schema.String),
  host: Schema.NullOr(Schema.String),
  authenticator: Schema.NullOr(Schema.String),
  credentialEnvironmentVariable: Schema.NullOr(Schema.String),
  credentialField: Schema.NullOr(Schema.String),
});
export type SnowflakeOutput = Schema.Schema.Type<typeof SnowflakeOutput>;

export const OutputDestination = Schema.Struct({
  kind: StorageKind,
  workspacePath: Schema.String,
  snowflake: Schema.NullOr(SnowflakeOutput),
});
export type OutputDestination = Schema.Schema.Type<typeof OutputDestination>;

export const IngestionRequest = Schema.Struct({
  runtime: RuntimeMode,
  jobId: Schema.String,
  graphId: Schema.String,
  graphName: Schema.NullOr(Schema.String),
  sourceKind: SourceKind,
  source: JsonRecord,
  ocr: ProviderSelection,
  llm: ProviderSelection,
  embedding: ProviderSelection,
  output: OutputDestination,
  baseConfigPath: Schema.NullOr(Schema.String),
  includeGlobs: Schema.optionalWith(Schema.Array(Schema.String), {
    default: () => ["**/*"],
  }),
  cacheProvider: Schema.optionalWith(Schema.String, { default: () => "local" }),
  providerParallelism: Schema.optionalWith(
    Schema.Number.pipe(Schema.int(), Schema.between(1, 64)),
    { default: () => DEFAULT_PROVIDER_PARALLELISM },
  ),
  runtimeOptions: Schema.optionalWith(JsonRecord, { default: () => ({}) }),
});
export type IngestionRequest = Schema.Schema.Type<typeof IngestionRequest>;

export const ProgressEvent = Schema.Struct({
  timestamp: Schema.String,
  stage: Schema.String,
  status: Schema.String,
  fileId: Schema.NullOr(Schema.String),
  message: Schema.NullOr(Schema.String),
  elapsedMs: Schema.NullOr(Schema.Number),
  counts: Schema.optionalWith(JsonRecord, { default: () => ({}) }),
  metadata: Schema.optionalWith(JsonRecord, { default: () => ({}) }),
});
export type ProgressEvent = Schema.Schema.Type<typeof ProgressEvent>;

export const StageProgress = Schema.Struct({
  stage: Schema.String,
  status: Schema.String,
  completed: Schema.optionalWith(Schema.Number, { default: () => 0 }),
  total: Schema.NullOr(Schema.Number),
  elapsedMs: Schema.optionalWith(Schema.Number, { default: () => 0 }),
  message: Schema.NullOr(Schema.String),
});
export type StageProgress = Schema.Schema.Type<typeof StageProgress>;

export const RunSnapshot = Schema.Struct({
  runId: Schema.String,
  graphId: Schema.String,
  status: Schema.String,
  startedAt: Schema.NullOr(Schema.String),
  updatedAt: Schema.NullOr(Schema.String),
  graphName: Schema.NullOr(Schema.String),
  stages: Schema.optionalWith(Schema.Array(StageProgress), { default: () => [] }),
  events: Schema.optionalWith(Schema.Array(ProgressEvent), { default: () => [] }),
  documentsTotal: Schema.NullOr(Schema.Number),
  documentsCompleted: Schema.optionalWith(Schema.Number, { default: () => 0 }),
  documentsFailed: Schema.optionalWith(Schema.Number, { default: () => 0 }),
  outputPath: Schema.NullOr(Schema.String),
  storageKind: StorageKind,
  storageLocation: Schema.NullOr(Schema.String),
  warnings: Schema.optionalWith(Schema.Array(Schema.String), { default: () => [] }),
  error: Schema.NullOr(Schema.String),
  listingWarning: Schema.NullOr(Schema.String),
  raw: Schema.optionalWith(JsonRecord, { default: () => ({}) }),
});
export type RunSnapshot = Schema.Schema.Type<typeof RunSnapshot>;

export const GraphDataset = Schema.Struct({
  graphId: Schema.String,
  nodes: Schema.Array(JsonRecord),
  edges: Schema.Array(JsonRecord),
  communities: Schema.optionalWith(Schema.Array(JsonRecord), { default: () => [] }),
  evidence: Schema.optionalWith(Schema.Array(JsonRecord), { default: () => [] }),
  documents: Schema.optionalWith(Schema.Array(JsonRecord), { default: () => [] }),
  chunks: Schema.optionalWith(Schema.Array(JsonRecord), { default: () => [] }),
  runReport: Schema.optionalWith(JsonRecord, { default: () => ({}) }),
  graphMetrics: Schema.optionalWith(JsonRecord, { default: () => ({}) }),
});
export type GraphDataset = Schema.Schema.Type<typeof GraphDataset>;

export const GraphCounts = Schema.Struct({
  documents: Schema.Number,
  nodes: Schema.Number,
  edges: Schema.Number,
  communities: Schema.Number,
  evidence: Schema.Number,
});
export type GraphCounts = Schema.Schema.Type<typeof GraphCounts>;

export const WorkloadStatus = Schema.Struct({
  name: Schema.String,
  component: Schema.String,
  phase: Schema.String,
  node: Schema.NullOr(Schema.String),
  ready: Schema.Boolean,
  restarts: Schema.Number,
  cpu: Schema.NullOr(Schema.String),
  memory: Schema.NullOr(Schema.String),
  image: Schema.NullOr(Schema.String),
  model: Schema.NullOr(Schema.String),
});
export type WorkloadStatus = Schema.Schema.Type<typeof WorkloadStatus>;

export const NodeStatus = Schema.Struct({
  name: Schema.String,
  ready: Schema.Boolean,
  nodeClass: Schema.String,
  gpuCount: Schema.Number,
  gpuModel: Schema.NullOr(Schema.String),
  cpuCapacity: Schema.NullOr(Schema.String),
  memoryCapacity: Schema.NullOr(Schema.String),
  gpuPercent: Schema.NullOr(Schema.String),
  cpuUsage: Schema.NullOr(Schema.String),
  cpuPercent: Schema.NullOr(Schema.String),
  memoryUsage: Schema.NullOr(Schema.String),
  memoryPercent: Schema.NullOr(Schema.String),
  workloadCount: Schema.optionalWith(Schema.Number, { default: () => 0 }),
  workerCount: Schema.optionalWith(Schema.Number, { default: () => 0 }),
  modelServerReady: Schema.optionalWith(Schema.Boolean, { default: () => false }),
  model: Schema.NullOr(Schema.String),
  modelImage: Schema.NullOr(Schema.String),
});
export type NodeStatus = Schema.Schema.Type<typeof NodeStatus>;

export const NodeWorkAssignment = Schema.Struct({
  workerId: Schema.String,
  runId: Schema.String,
  graphId: Schema.String,
  taskId: Schema.String,
  stage: Schema.String,
  scopeId: Schema.String,
  updatedAt: Schema.NullOr(Schema.String),
});
export type NodeWorkAssignment = Schema.Schema.Type<typeof NodeWorkAssignment>;

export const ClusterSnapshot = Schema.Struct({
  context: Schema.String,
  namespace: Schema.String,
  nodesReady: Schema.Number,
  nodesTotal: Schema.Number,
  nodes: Schema.Array(NodeStatus),
  workloads: Schema.Array(WorkloadStatus),
  warnings: Schema.optionalWith(Schema.Array(Schema.String), { default: () => [] }),
  observedAt: Schema.String,
});
export type ClusterSnapshot = Schema.Schema.Type<typeof ClusterSnapshot>;

export const ClusterProfile = Schema.Struct({
  name: Schema.String,
  namespace: Schema.optionalWith(Schema.String, { default: () => "flakegraph" }),
  context: Schema.optionalWith(Schema.String, { default: () => "" }),
  kubeconfig: Schema.optionalWith(Schema.String, { default: () => "" }),
  description: Schema.optionalWith(Schema.String, { default: () => "" }),
});
export type ClusterProfile = Schema.Schema.Type<typeof ClusterProfile>;

export const PreflightResult = Schema.Struct({
  ok: Schema.Boolean,
  errors: Schema.optionalWith(Schema.Array(Schema.String), { default: () => [] }),
  warnings: Schema.optionalWith(Schema.Array(Schema.String), { default: () => [] }),
  checks: Schema.optionalWith(Schema.Array(JsonRecord), { default: () => [] }),
});
export type PreflightResult = Schema.Schema.Type<typeof PreflightResult>;

export const RuntimeCapabilities = Schema.Struct({
  runtime: RuntimeMode,
  capabilities: Schema.Array(Capability),
});
export type RuntimeCapabilities = Schema.Schema.Type<typeof RuntimeCapabilities>;

export const GraphShare = Schema.Struct({
  granteeType: Schema.String,
  grantee: Schema.String,
});
export type GraphShare = Schema.Schema.Type<typeof GraphShare>;

export const Session = Schema.Struct({
  viewer: Viewer,
  runtime: RuntimeMode,
  capabilities: Schema.Array(Capability),
  identified: Schema.Boolean,
  snowflakeHosted: Schema.Boolean,
});
export type Session = Schema.Schema.Type<typeof Session>;

export function decodeSchema<A, I>(schema: Schema.Schema<A, I>, value: unknown): A {
  return Schema.decodeUnknownSync(schema)(value);
}

export function encodeSchema<A, I>(schema: Schema.Schema<A, I>, value: A): I {
  return Schema.encodeSync(schema)(value);
}

export function parseSchema<A, I>(
  schema: Schema.Schema<A, I>,
  value: unknown,
): Either.Either<A, unknown> {
  return Schema.decodeUnknownEither(schema)(value);
}

export function standardSchema<A, I>(schema: Schema.Schema<A, I>) {
  return {
    "~standard": {
      version: 1 as const,
      vendor: "effect",
      validate(value: unknown) {
        const result = Schema.decodeUnknownEither(schema)(value);
        if (Either.isRight(result)) {
          return { value: result.right };
        }
        return {
          issues: [{ message: formatParseError(result.left) }],
        };
      },
    },
  };
}

export function formatParseError(error: unknown): string {
  if (error && typeof error === "object" && "message" in error) {
    return String((error as { message: unknown }).message);
  }
  return String(error);
}

export function isActiveStatus(status: string): boolean {
  return (ACTIVE_STATUSES as readonly string[]).includes(status.toLowerCase());
}

export function isSuccessStatus(status: string): boolean {
  return (SUCCESS_STATUSES as readonly string[]).includes(status.toLowerCase());
}

export function displayName(snapshot: Pick<RunSnapshot, "graphName" | "graphId">): string {
  return snapshot.graphName || snapshot.graphId;
}

export function graphCounts(dataset: GraphDataset): GraphCounts {
  const full = dataset.runReport.app_full_counts;
  if (full && typeof full === "object") {
    const counts = full as Record<string, unknown>;
    return {
      documents: numberOr(counts.document, dataset.documents.length),
      nodes: numberOr(counts.node, dataset.nodes.length),
      edges: numberOr(counts.edge, dataset.edges.length),
      communities: numberOr(counts.community, dataset.communities.length),
      evidence: numberOr(counts.evidence, dataset.evidence.length),
    };
  }
  return {
    documents: dataset.documents.length,
    nodes: dataset.nodes.length,
    edges: dataset.edges.length,
    communities: dataset.communities.length,
    evidence: dataset.evidence.length,
  };
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function writerProvider(kind: StorageKind): string {
  return kind === "snowflake" ? "snowflake_bulk" : "local_artifacts";
}

export function storageLocation(output: OutputDestination): string {
  if (output.snowflake) {
    return `${output.snowflake.database}.${output.snowflake.schema}`;
  }
  return output.workspacePath;
}

export function validateGraphName(value: string): string {
  const normalized = value.split(/\s+/).join(" ").trim();
  if (!normalized) {
    throw new Error("Graph name must not be empty");
  }
  if (normalized.length > MAX_GRAPH_NAME_LENGTH) {
    throw new Error(`Graph name must be at most ${MAX_GRAPH_NAME_LENGTH} characters`);
  }
  if ([...normalized].some((character) => character.charCodeAt(0) < 32)) {
    throw new Error("Graph name must not contain control characters");
  }
  return normalized;
}

export const LOCAL_CAPABILITIES: Capability[] = [
  "upload",
  "local",
  "azure_blob",
  "s3",
  "progress",
  "graph",
  "forget",
  "rename",
  "cancel",
];

export const KUBERNETES_CAPABILITIES: Capability[] = [
  ...LOCAL_CAPABILITIES,
  "cluster",
  "distributed",
  "cancel",
  "model_serving",
  "node_assignments",
  "recover",
  "retry",
];

export const SNOWFLAKE_CAPABILITIES: Capability[] = [
  "upload",
  "snowflake_stage",
  "progress",
  "graph",
  "spcs",
  "cancel",
  "rename",
  "forget",
  "share",
  "delete_graph",
  "retry",
];

export function capabilitiesFor(runtime: RuntimeMode): Capability[] {
  if (runtime === "kubernetes") {
    return KUBERNETES_CAPABILITIES;
  }
  if (runtime === "snowflake") {
    return SNOWFLAKE_CAPABILITIES;
  }
  return LOCAL_CAPABILITIES;
}
