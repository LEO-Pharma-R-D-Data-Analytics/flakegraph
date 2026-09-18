import { randomBytes } from "node:crypto";
import path from "node:path";
import { hashApiSecret } from "./api-keys";
import { appEnv } from "./env";
import { atomicWriteJson, hideRuns, listRunRecords, readJsonFile } from "./catalog";
import type { Viewer } from "./protocol/schema";
import type { PiiHit } from "./pii";
import type { ConsumptionEstimate } from "./estimate";

export type ShellRole = "operator" | "analyst" | "staff";
export type SuggestionMode = "off" | "on-request" | "auto-fill";
export type PerspectiveLifecycle = "draft" | "candidate" | "production";
export type ReviewDecision = "keep" | "drop" | "merge" | "relabel";

export interface Perspective {
  id: string;
  graphId: string;
  name: string;
  lifecycle: PerspectiveLifecycle;
  search: string;
  communityIds: string[];
  suggestedQuestions: string[];
}

export interface GraphVersion {
  id: string;
  graphId: string;
  label: string;
  lifecycle: PerspectiveLifecycle;
  createdAt: string;
  note: string;
}

export interface Pin {
  id: string;
  graphId: string;
  targetId: string;
  kind: "node" | "edge";
  note: string;
  createdAt: string;
}

export interface HitlItem {
  id: string;
  graphId: string;
  edgeId: string;
  quote: string;
  confidence: number;
  decision: ReviewDecision | null;
}

export interface IncrementalWatch {
  id: string;
  graphId: string;
  prefix: string;
  paused: boolean;
  lastWindow: string;
  pendingFiles: number;
  goldDrift: number;
}

export interface ApiKey {
  id: string;
  name: string;
  preview: string;
  createdAt: string;
  note: string;
  secretHash?: string;
}

export interface OntologyProposal {
  intent: string;
  types: string[];
  relations: string[];
  warning: string;
}

export interface SnowflakeGrant {
  object: string;
  ok: boolean;
  sql: string;
}

export interface PromotionPlan {
  fromRuntime: string;
  toRuntime: string;
  keepGraphId: boolean;
  remaps: Array<{ from: string; to: string }>;
}

export interface PendingPromotion {
  fromRuntime: string;
  toRuntime: string;
  keepGraphId: boolean;
  graphId: string;
  graphName: string;
  sourceKind: string;
  sourcePath: string;
  remaps: Array<{ from: string; to: string }>;
}

export interface IdentityIncident {
  expectedPrincipal: string;
  actualPrincipal: string;
  graphOwner: string;
  grants: string;
  template: string;
}

export interface WorkspaceState {
  identity: Viewer | null;
  role: ShellRole;
  suggestionMode: SuggestionMode;
  lastSuccessRuntime: string | null;
  lastConfigDigest: Record<string, string>;
  piiAcknowledged: string[];
  skippedFiles: Record<string, string[]>;
  estimates: Record<string, ConsumptionEstimate>;
  perspectives: Perspective[];
  versions: GraphVersion[];
  pins: Pin[];
  reviews: HitlItem[];
  watches: IncrementalWatch[];
  apiKeys: ApiKey[];
  grants: SnowflakeGrant[];
  pendingPromotion: PendingPromotion | null;
  identityIncident: IdentityIncident | null;
}

const FILE = "workspace.json";

export function emptyWorkspace(): WorkspaceState {
  return {
    identity: null,
    role: "operator",
    suggestionMode: "on-request",
    lastSuccessRuntime: null,
    lastConfigDigest: {},
    piiAcknowledged: [],
    skippedFiles: {},
    estimates: {},
    perspectives: [],
    versions: [],
    pins: [],
    reviews: [],
    watches: [],
    apiKeys: [],
    grants: defaultGrants(),
    pendingPromotion: null,
    identityIncident: null,
  };
}

export async function loadWorkspace(stateRoot = appEnv().stateRoot): Promise<WorkspaceState> {
  const value = await readJsonFile<Partial<WorkspaceState>>(path.join(stateRoot, FILE));
  return { ...emptyWorkspace(), ...(value ?? {}) };
}

export async function catalogWriterPrincipal(stateRoot = appEnv().stateRoot): Promise<string | null> {
  const identity = (await loadWorkspace(stateRoot)).identity;
  return identity?.userName ? identity.userName.toUpperCase() : null;
}

export async function saveWorkspace(state: WorkspaceState, stateRoot = appEnv().stateRoot): Promise<WorkspaceState> {
  await atomicWriteJson(path.join(stateRoot, FILE), state);
  return state;
}

export async function patchWorkspace(
  patch: (current: WorkspaceState) => WorkspaceState | Promise<WorkspaceState>,
  stateRoot = appEnv().stateRoot,
): Promise<WorkspaceState> {
  const current = await loadWorkspace(stateRoot);
  const next = await patch(current);
  return saveWorkspace(next, stateRoot);
}

export async function assumeIdentity(input: {
  userName: string;
  email?: string;
  roles: string[];
  role: ShellRole;
}): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    identity: input.userName
      ? { userName: input.userName.toUpperCase(), email: input.email ?? "", roles: input.roles.map((role) => role.toUpperCase()) }
      : null,
    role: input.role,
  }));
}

export async function setSuggestionMode(mode: SuggestionMode): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({ ...current, suggestionMode: mode }));
}

export async function recordRuntimeSuccess(runtime: string, digest: string): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    lastSuccessRuntime: runtime,
    lastConfigDigest: { ...current.lastConfigDigest, [runtime]: digest },
  }));
}

export async function acknowledgePii(sourceKey: string): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    piiAcknowledged: [...new Set([...current.piiAcknowledged, sourceKey])],
  }));
}

export function piiBlocked(state: WorkspaceState, sourceKey: string, hits: readonly PiiHit[]): boolean {
  return hits.length > 0 && !state.piiAcknowledged.includes(sourceKey);
}

export async function skipFile(runId: string, fileId: string): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    skippedFiles: {
      ...current.skippedFiles,
      [runId]: [...new Set([...(current.skippedFiles[runId] ?? []), fileId])],
    },
  }));
}

export async function recordEstimate(runId: string, estimate: ConsumptionEstimate): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    estimates: { ...current.estimates, [runId]: estimate },
  }));
}

export async function upsertPerspective(perspective: Omit<Perspective, "id"> & { id?: string }): Promise<WorkspaceState> {
  return patchWorkspace((current) => {
    const id = perspective.id ?? `perspective_${randomId()}`;
    const next: Perspective = { ...perspective, id };
    const others = current.perspectives.filter((item) => item.id !== id);
    return { ...current, perspectives: [...others, next] };
  });
}

export async function promotePerspective(id: string, lifecycle: PerspectiveLifecycle): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    perspectives: current.perspectives.map((item) => (item.id === id ? { ...item, lifecycle } : item)),
  }));
}

export async function createWatch(input: { graphId: string; prefix: string }): Promise<WorkspaceState> {
  return patchWorkspace((current) => {
    const existing = current.watches.find((item) => item.graphId === input.graphId && item.prefix === input.prefix);
    if (existing) {
      return current;
    }
    return {
      ...current,
      watches: [
        ...current.watches,
        {
          id: `watch_${randomId()}`,
          graphId: input.graphId,
          prefix: input.prefix,
          paused: false,
          lastWindow: new Date().toISOString(),
          pendingFiles: 0,
          goldDrift: 0,
        },
      ],
    };
  });
}

export async function setPendingPromotion(promotion: PendingPromotion | null): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({ ...current, pendingPromotion: promotion }));
}

export async function recordIdentityIncident(input: {
  expectedPrincipal: string;
  actualPrincipal: string;
  graphOwner: string;
  grants: string;
}): Promise<WorkspaceState> {
  const expected = input.expectedPrincipal.trim().toUpperCase();
  const actual = input.actualPrincipal.trim().toUpperCase();
  const template = [
    "Identity mismatch incident",
    `Expected ACL principal: ${expected}`,
    `Actual ACL principal compared: ${actual}`,
    `Graph owner: ${input.graphOwner.trim().toUpperCase() || "(unowned)"}`,
    `Grants: ${input.grants.trim() || "(none)"}`,
    expected === actual
      ? "Principals match. Look at ROLE vs USER grantee type next."
      : "These strings are not equal. Sharing bugs are identity bugs.",
  ].join("\n");
  return patchWorkspace((current) => ({
    ...current,
    identityIncident: { ...input, expectedPrincipal: expected, actualPrincipal: actual, template },
  }));
}

export function sampleHighConfidenceReviews(
  graphId: string,
  edges: ReadonlyArray<Record<string, unknown>>,
): HitlItem[] {
  const high = edges.filter((edge) => {
    const confidence = Number(edge.confidence ?? 0);
    return Number.isFinite(confidence) && confidence >= 0.9;
  });
  const stride = Math.max(1, Math.round(high.length / Math.max(1, Math.round(high.length * 0.01))));
  const sampled = high.length <= 1 ? high : high.filter((_, index) => index % stride === 0).slice(0, Math.max(1, Math.ceil(high.length * 0.01)));
  return sampled.map((edge) => ({
    id: `review_${randomId()}`,
    graphId,
    edgeId: String(edge.id ?? "edge"),
    quote: String(edge.description ?? edge.quote ?? "High-confidence triple sampled for silent OCR drift."),
    confidence: Number(edge.confidence ?? 0.9),
    decision: null,
  }));
}

export async function enqueueReviews(items: HitlItem[]): Promise<WorkspaceState> {
  return patchWorkspace((current) => {
    const seen = new Set(current.reviews.map((item) => `${item.graphId}:${item.edgeId}`));
    const extra = items.filter((item) => !seen.has(`${item.graphId}:${item.edgeId}`));
    return { ...current, reviews: [...current.reviews, ...extra] };
  });
}

export async function pinTarget(pin: Omit<Pin, "id" | "createdAt">): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    pins: [
      ...current.pins,
      { ...pin, id: `pin_${randomId()}`, createdAt: new Date().toISOString() },
    ],
  }));
}

export async function decideReview(id: string, decision: ReviewDecision): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    reviews: current.reviews.map((item) => (item.id === id ? { ...item, decision } : item)),
  }));
}

export async function publishVersion(graphId: string, note: string): Promise<WorkspaceState> {
  return patchWorkspace((current) => {
    const versions = current.versions.map((item) =>
      item.graphId === graphId && item.lifecycle === "production"
        ? { ...item, lifecycle: "candidate" as const }
        : item,
    );
    versions.push({
      id: `v_${randomId()}`,
      graphId,
      label: `v${versions.filter((item) => item.graphId === graphId).length + 1}`,
      lifecycle: "production",
      createdAt: new Date().toISOString(),
      note,
    });
    return { ...current, versions };
  });
}

export async function toggleWatch(id: string): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    watches: current.watches.map((item) => (item.id === id ? { ...item, paused: !item.paused } : item)),
  }));
}

export async function applyIncremental(id: string): Promise<WorkspaceState> {
  return patchWorkspace((current) => {
    const watch = current.watches.find((item) => item.id === id);
    if (!watch || watch.paused) {
      return current;
    }
    const nextWatches = current.watches.map((item) =>
      item.id === id
        ? { ...item, pendingFiles: 0, lastWindow: new Date().toISOString(), goldDrift: item.goldDrift }
        : item,
    );
    const versions = [
      ...current.versions,
      {
        id: `v_${randomId()}`,
        graphId: watch.graphId,
        label: `v${current.versions.filter((item) => item.graphId === watch.graphId).length + 1} incremental`,
        lifecycle: "draft" as const,
        createdAt: new Date().toISOString(),
        note: `${watch.pendingFiles} new files from ${watch.prefix}. Gold drift ${watch.goldDrift} missing triples.`,
      },
    ];
    return { ...current, watches: nextWatches, versions };
  });
}

export async function createApiKey(name: string): Promise<{ state: WorkspaceState; secret: string }> {
  const secret = `fg_${randomBytes(18).toString("hex")}`;
  const state = await patchWorkspace((current) => ({
    ...current,
    apiKeys: [
      ...current.apiKeys,
      {
        id: `key_${randomId()}`,
        name,
        preview: `${secret.slice(0, 7)}…${secret.slice(-4)}`,
        createdAt: new Date().toISOString(),
        note: "This path does not use your SSO cookie. A 401 here is not a browser login bug.",
        secretHash: hashApiSecret(secret),
      },
    ],
  }));
  return { state, secret };
}

export async function revokeApiKey(id: string): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    apiKeys: current.apiKeys.filter((key) => key.id !== id),
  }));
}

export async function forgetMany(runIds: string[]): Promise<number> {
  return hideRuns(appEnv().stateRoot, runIds);
}

export function promotionPlan(fromRuntime: string, toRuntime: string, keepGraphId: boolean): PromotionPlan {
  const remaps: PromotionPlan["remaps"] = [];
  if (fromRuntime === "local" && toRuntime !== "local") {
    remaps.push({
      from: "Laptop path",
      to: toRuntime === "snowflake" ? "Stage app/<user>/<job> that workers can LIST" : "Object storage the fleet can mount",
    });
  }
  if (toRuntime === "snowflake") {
    remaps.push({ from: "Local embeddings / vLLM", to: "Cortex or a Snowflake-hosted model" });
  }
  if (fromRuntime !== toRuntime) {
    remaps.push({ from: `${fromRuntime} catalog`, to: `${toRuntime} catalog · same graph ID ${keepGraphId ? "kept" : "minted as a sibling"}` });
  }
  return { fromRuntime, toRuntime, keepGraphId, remaps };
}

export function proposeOntology(intent: string, goldTypes: readonly string[] = []): OntologyProposal {
  const tokens = intent
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 3 && !["these", "those", "want", "graph", "with", "from"].includes(token));
  const types = [...new Set(tokens.slice(0, 6).map((token) => token.replace(/s$/, "").toUpperCase()))];
  const proposed = types.length ? types : ["PERSON", "ORGANIZATION", "CONCEPT"];
  const missing = goldTypes.filter((type) => !proposed.includes(type.toUpperCase()));
  return {
    intent,
    types: proposed,
    relations: ["RELATED_TO", "PART_OF", "LOCATED_IN"],
    warning: missing.length
      ? `Proposed types are a coverage overlay. Gold still needs ${missing.join(", ")} before a full corpus run.`
      : "Proposed types cover the gold vocabulary. Confirm before a full corpus run.",
  };
}

export function defaultGrants(): SnowflakeGrant[] {
  return [
    { object: "WAREHOUSE FLAKEGRAPH_WH", ok: true, sql: "GRANT USAGE ON WAREHOUSE FLAKEGRAPH_WH TO ROLE APP_OPERATOR;" },
    { object: "DATABASE FLAKEGRAPH", ok: true, sql: "GRANT USAGE ON DATABASE FLAKEGRAPH TO ROLE APP_OPERATOR;" },
    { object: "STAGE APP_STAGE", ok: false, sql: "GRANT READ, WRITE ON STAGE FLAKEGRAPH.PUBLIC.APP_STAGE TO ROLE APP_OPERATOR;" },
    { object: "CORTEX", ok: false, sql: "GRANT DATABASE ROLE SNOWFLAKE.CORTEX_USER TO ROLE APP_OPERATOR;" },
  ];
}

export async function markGrant(object: string, ok: boolean): Promise<WorkspaceState> {
  return patchWorkspace((current) => ({
    ...current,
    grants: current.grants.map((item) => (item.object === object ? { ...item, ok } : item)),
  }));
}

export async function staffIncident(): Promise<{
  banner: string;
  queueDepth: number;
  modelReady: boolean;
  events: Array<{ runId: string; graphName: string; status: string; error: string | null; runtime: string }>;
  activeRunIds: string[];
}> {
  const rows = await listRunRecords(appEnv().stateRoot, 40, { includeHidden: true });
  const events = rows.slice(0, 20).map(({ record }) => ({
    runId: record.runId,
    graphName: record.graphName || record.graphId,
    status: record.status,
    error: record.error ?? null,
    runtime: String(record.runtime ?? ""),
  }));
  const active = events.filter((item) => ["queued", "running", "pending", "cancelling"].includes(item.status));
  return {
    banner: "Staff shell uses owner’s-rights. This is not tenant isolation.",
    queueDepth: active.length,
    modelReady: true,
    events,
    activeRunIds: active.map((item) => item.runId),
  };
}

function randomId(): string {
  return randomBytes(4).toString("hex");
}
