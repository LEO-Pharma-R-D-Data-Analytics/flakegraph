import { stat } from "node:fs/promises";
import path from "node:path";
import { Effect } from "effect";
import { isWithin } from "../confine";
import type { FleetProfile } from "../fleet";
import type { ControlPlane, GraphAccessPlane, SnowflakePublishing } from "../protocol/runtime";
import { forbidden, fromCause, invalid, type ControlPlaneError } from "../protocol/errors";
import {
  isActiveStatus,
  type Capability,
  type GraphAccessView,
  type GraphDeletionPreview,
  type GraphDataset,
  type GraphRole,
  type IngestionRequest,
  type RunSnapshot,
  type Viewer,
} from "../protocol/schema";
import {
  allows,
  principalKey,
  roleFor,
  validPrincipal,
  type AccessLevel,
  type AccessPolicy,
  type GraphAccessRecord,
} from "./graph-access";
import {
  claim,
  claimUpload,
  forgetGraph,
  grant,
  loadAccess,
  recordFor,
  revoke,
  sawPerson,
  updateAccess,
  type AccessState,
} from "./store";
import { graphIsKnown, runIsKnown } from "./known";
import { exclusiveUploads, graphFootprint, removeGraphRecords } from "../graph-cleanup";

/** What a refused viewer is told: the same whether the graph exists or not. */
export const NO_ACCESS = "You don't have access to this graph. Ask its owner to share it with you.";

const NEEDS: Record<GraphRole, string> = {
  read: NO_ACCESS,
  write: "You have read access to this graph. Changing it needs write access from its owner.",
  owner: "Only the graph's owner can do this.",
};

/**
 * How many of the fleet's most recent runs are read to answer one request:
 * to find who submitted a graph, and to fill a viewer's list. The list is
 * filtered to the viewer's graphs before it is cut to the size asked for, so
 * other people's runs do not push a viewer's own graphs off it.
 */
const RUN_SCAN_LIMIT = 5_000;

/** A list's size when the caller names none. */
const DEFAULT_LIST_LIMIT = 100;

/**
 * Wrap a runtime so everything it serves is limited to one viewer's graphs.
 *
 * Ownership and grants live in the console's access store. A graph the store
 * has no record for is owned by whoever submitted its earliest run, as the
 * run records say; sharing such a graph first writes that owner down.
 */
export function controlledPlane(inner: ControlPlane, viewer: Viewer, stateRoot: string, policy: AccessPolicy): GraphAccessPlane {
  return new ControlledPlane(inner, viewer, stateRoot, policy);
}

class ControlledPlane implements GraphAccessPlane {
  readonly runtime: ControlPlane["runtime"];
  readonly capabilities: ReadonlySet<Capability>;
  private runsMemo: Promise<readonly RunSnapshot[]> | null = null;

  constructor(
    private readonly inner: ControlPlane,
    private readonly who: Viewer,
    private readonly stateRoot: string,
    private readonly policy: AccessPolicy,
  ) {
    this.runtime = inner.runtime;
    this.capabilities = new Set<Capability>([...inner.capabilities, "share"]);
  }

  // ---------------------------------------------------------------- access

  requireGraph(graphId: string, needed: GraphRole): Effect.Effect<GraphRole, ControlPlaneError> {
    return this.attempt(async () => this.roleOrFail(graphId, needed), "Unable to check access");
  }

  requireRun(runId: string, needed: GraphRole): Effect.Effect<GraphRole, ControlPlaneError> {
    return this.attempt(async () => {
      const snapshot = await Effect.runPromise(this.inner.getRun(runId));
      return this.roleOrFail(snapshot.graphId, needed, ownerHint(snapshot));
    }, "Unable to check access");
  }

  visibleGraphs(): Effect.Effect<ReadonlyMap<string, GraphRole>, ControlPlaneError> {
    return this.attempt(async () => {
      const state = await loadAccess(this.stateRoot);
      const owners = firstOwners(await this.allRuns());
      const graphIds = new Set([...owners.keys(), ...Object.keys(state.graphs)]);
      const visible = new Map<string, GraphRole>();
      for (const graphId of graphIds) {
        const role = roleFor(effectiveRecord(state, graphId, owners.get(graphId) ?? null), this.who, this.policy);
        if (role) {
          visible.set(graphId, role);
        }
      }
      return visible;
    }, "Unable to list graphs");
  }

  graphAccess(graphId: string): Effect.Effect<GraphAccessView, ControlPlaneError> {
    return this.attempt(async () => this.view(graphId), "Unable to read who can use this graph");
  }

  shareGraph(graphId: string, principal: string, level: AccessLevel): Effect.Effect<GraphAccessView, ControlPlaneError> {
    return this.attempt(async () => {
      await this.roleOrFail(graphId, "owner");
      const grantee = validPrincipal(principal, this.policy);
      if (!grantee) {
        throw invalid(
          this.policy.strict ? "Enter the person's sign-in address, such as name@example.com." : "Enter a user name.",
        );
      }
      const owner = await this.ownerOf(graphId);
      if (owner && principalKey(owner) === grantee) {
        throw invalid("The owner already has full access.");
      }
      await updateAccess(this.stateRoot, (state) => {
        if (owner) {
          claim(state, graphId, owner);
        }
        grant(state, graphId, grantee, level, this.who.userName || "SYSTEM");
      });
      return this.view(graphId);
    }, "Unable to share graph");
  }

  unshareGraph(graphId: string, principal: string): Effect.Effect<GraphAccessView | null, ControlPlaneError> {
    return this.attempt(async () => {
      const target = principalKey(principal);
      const me = principalKey(this.who.userName);
      const leaving = Boolean(me) && target === me;
      const role = await this.roleOrFail(graphId, leaving ? "read" : "owner");
      if (leaving && role === "owner") {
        throw invalid("The owner cannot remove their own access.");
      }
      await updateAccess(this.stateRoot, (state) => {
        revoke(state, graphId, target);
      });
      return leaving ? null : this.view(graphId);
    }, "Unable to remove access");
  }

  people(): Effect.Effect<readonly string[], ControlPlaneError> {
    return this.attempt(async () => {
      const state = await loadAccess(this.stateRoot);
      const me = principalKey(this.who.userName);
      // People who have signed in, and the people on graphs the viewer can
      // already see; who else owns or holds what stays private.
      const known = new Set<string>(Object.keys(state.people));
      const visible = await Effect.runPromise(this.visibleGraphs());
      for (const graphId of visible.keys()) {
        const record = recordFor(state, graphId);
        if (record?.owner) {
          known.add(record.owner);
        }
        for (const item of record?.grants ?? []) {
          known.add(item.principal);
        }
      }
      known.delete(me);
      known.delete("SYSTEM");
      return [...known].filter(Boolean).sort();
    }, "Unable to list people");
  }

  // ---------------------------------------------------------------- reads

  viewer(): Effect.Effect<Viewer, ControlPlaneError> {
    return this.inner.viewer();
  }

  listSourceObjects(source: Record<string, unknown>, limit?: number) {
    return Effect.flatMap(this.checkSource(source), () => this.inner.listSourceObjects(source, limit));
  }

  summarizeSource(source: Record<string, unknown>) {
    return Effect.flatMap(this.checkSource(source), () => this.inner.summarizeSource(source));
  }

  preflight(request: IngestionRequest) {
    return Effect.flatMap(this.checkRequest(request), () => this.inner.preflight(request));
  }

  previewConfig(request: IngestionRequest) {
    return Effect.flatMap(this.checkRequest(request), () => this.inner.previewConfig(request));
  }

  cluster(namespace: string) {
    return this.inner.cluster(namespace);
  }

  /** A node's leased work, limited to the viewer's graphs: others' run and graph ids are theirs. */
  nodeAssignments(namespace: string, nodeName: string) {
    return this.attempt(async () => {
      const assignments = await Effect.runPromise(this.inner.nodeAssignments(namespace, nodeName));
      const visible = await Effect.runPromise(this.visibleGraphs());
      return assignments.filter((item) => visible.has(item.graphId));
    }, "Unable to read the node's work");
  }

  fleetProfile(): Promise<FleetProfile | null> {
    const inner = this.inner as { fleetProfile?: () => Promise<FleetProfile | null> };
    return inner.fleetProfile ? inner.fleetProfile() : Promise.resolve(null);
  }

  get snowflakePublishing(): SnowflakePublishing | null {
    const inner = this.inner as { publishToSnowflake?: SnowflakePublishing };
    const publish = inner.publishToSnowflake?.bind(this.inner);
    if (!publish) {
      return null;
    }
    // Publishing copies a graph out of the console, so it needs write access.
    return async (runId, target) => {
      await Effect.runPromise(this.requireRun(runId, "write"));
      return publish(runId, target);
    };
  }

  listRuns(limit?: number): Effect.Effect<readonly RunSnapshot[], ControlPlaneError> {
    return this.attempt(async () => {
      const runs = await this.allRuns();
      const state = await loadAccess(this.stateRoot);
      const owners = firstOwners(runs);
      const visible: RunSnapshot[] = [];
      for (const run of runs) {
        const record = effectiveRecord(state, run.graphId, owners.get(run.graphId) ?? ownerHint(run));
        const role = roleFor(record, this.who, this.policy);
        if (role) {
          visible.push(annotate(run, record, role));
          if (visible.length >= (limit ?? DEFAULT_LIST_LIMIT)) {
            break;
          }
        }
      }
      return visible;
    }, "Unable to list runs");
  }

  getRun(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return this.attempt(async () => {
      const run = await Effect.runPromise(this.inner.getRun(runId));
      const state = await loadAccess(this.stateRoot);
      const record = effectiveRecord(state, run.graphId, (await this.ownerOf(run.graphId, state)) ?? ownerHint(run));
      const role = roleFor(record, this.who, this.policy);
      if (!role) {
        throw forbidden(NO_ACCESS);
      }
      return annotate(run, record, role);
    }, "Unable to load run");
  }

  documents(runId: string) {
    return this.guardRun(runId, "read", () => this.inner.documents(runId));
  }

  versions(graphId: string) {
    return this.guardGraph(graphId, "read", () => this.inner.versions(graphId));
  }

  /**
   * A graph by location, only as the output of one of that graph's own runs:
   * a location names a directory, and the graph id the access check is made
   * on has to be the graph that directory holds.
   */
  loadGraph(location: string, graphId?: string | null): Effect.Effect<GraphDataset, ControlPlaneError> {
    if (!graphId) {
      return Effect.fail(forbidden("A stored graph is opened through its run, or with its graph id."));
    }
    return this.guardGraph(graphId, "read", () =>
      Effect.flatMap(
        this.attempt(async () => {
          const target = path.resolve(location);
          const outputs = (await this.allRuns())
            .filter((run) => run.graphId === graphId)
            .flatMap((run) => [run.outputPath, run.storageLocation])
            .filter((value): value is string => Boolean(value))
            .map((value) => path.resolve(value));
          if (!outputs.includes(target)) {
            throw forbidden("That location is not where this graph is stored.");
          }
        }, "Unable to open graph"),
        () => this.inner.loadGraph(location, graphId),
      ),
    );
  }

  /** A path only: the caller has already opened the run, which checked the viewer may read it. */
  graphDirectory(snapshot: RunSnapshot): string | null {
    return this.inner.graphDirectory(snapshot);
  }

  loadRunGraph(snapshot: RunSnapshot): Effect.Effect<GraphDataset, ControlPlaneError> {
    return this.guardGraph(snapshot.graphId, "read", () => this.inner.loadRunGraph(snapshot), ownerHint(snapshot));
  }

  // -------------------------------------------------------------- changes

  submit(request: IngestionRequest): Effect.Effect<RunSnapshot, ControlPlaneError> {
    return this.attempt(async () => {
      await Effect.runPromise(this.checkRequest(request));
      const existing = await this.exists(request.graphId);
      if (existing) {
        await this.roleOrFail(request.graphId, "write");
      }
      const base = request.revision?.baseRunId;
      if (base) {
        const baseRun = await Effect.runPromise(this.inner.getRun(base));
        if (baseRun.graphId !== request.graphId) {
          throw invalid("A revision builds on a run of the same graph.");
        }
      }
      const snapshot = await Effect.runPromise(this.inner.submit(request));
      const me = principalKey(this.who.userName);
      if (!existing && me) {
        await updateAccess(this.stateRoot, (state) => claim(state, request.graphId, me));
      }
      const state = await loadAccess(this.stateRoot);
      const record = effectiveRecord(state, snapshot.graphId, ownerHint(snapshot));
      return annotate(snapshot, record, roleFor(record, this.who, this.policy));
    }, "Unable to submit run");
  }

  cancel(runId: string) {
    return this.guardRun(runId, "write", () => this.inner.cancel(runId));
  }

  retry(runId: string) {
    return this.guardRun(runId, "write", () => this.inner.retry(runId));
  }

  recover(runId: string) {
    return this.guardRun(runId, "write", () => this.inner.recover(runId));
  }

  renameGraph(graphId: string, displayName: string) {
    return this.guardGraph(graphId, "write", () => this.inner.renameGraph(graphId, displayName));
  }

  /**
   * Delete a graph for good, owner only: the runtime removes what it stored
   * (on a fleet, every run's rows and objects), then the console removes its
   * own records - run records, local copies, name, gold, workspace items, the
   * upload folders only this graph used - and the access record last.
   */
  deleteGraph(graphId: string): Effect.Effect<Record<string, number>, ControlPlaneError> {
    return this.attempt(async () => {
      await this.roleOrFail(graphId, "owner");
      const runs = (await this.allRuns()).filter((run) => run.graphId === graphId);
      const active = runs.filter((run) => isActiveStatus(run.status));
      if (active.length > 0) {
        throw invalid(
          `This graph still has ${active.length === 1 ? "a run" : `${active.length} runs`} under way; cancel ${active.length === 1 ? "it" : "them"} before deleting the graph.`,
        );
      }
      const footprint = await graphFootprint(this.stateRoot, graphId, runs.map((run) => run.runId));
      const removed = await Effect.runPromise(this.inner.deleteGraph(graphId));
      const uploads = await removeGraphRecords(this.stateRoot, footprint);
      await updateAccess(this.stateRoot, (state) => {
        forgetGraph(state, graphId);
        for (const jobId of uploads) {
          delete state.uploads[jobId];
        }
      });
      this.runsMemo = null;
      return { ...removed, runs: footprint.runIds.length, uploads: uploads.length };
    }, "Unable to delete graph");
  }

  deletionPreview(graphId: string): Effect.Effect<GraphDeletionPreview, ControlPlaneError> {
    return this.attempt(async () => {
      await this.roleOrFail(graphId, "owner");
      const runs = (await this.allRuns()).filter((run) => run.graphId === graphId);
      const footprint = await graphFootprint(this.stateRoot, graphId, runs.map((run) => run.runId));
      const record = recordFor(await loadAccess(this.stateRoot), graphId);
      return {
        graphId,
        graphName: runs.find((run) => run.graphName)?.graphName ?? graphId,
        runs: footprint.runIds.length,
        active: runs.filter((run) => isActiveStatus(run.status)).length,
        uploads: (await exclusiveUploads(this.stateRoot, footprint)).length,
        sharedWith: (record?.grants ?? []).map((item) => item.principal).sort(),
      };
    }, "Unable to read what deleting this graph removes");
  }

  // ------------------------------------------------------------ internals

  private guardRun<A>(runId: string, needed: GraphRole, action: () => Effect.Effect<A, ControlPlaneError>) {
    return Effect.flatMap(this.requireRun(runId, needed), () => action());
  }

  private guardGraph<A>(
    graphId: string,
    needed: GraphRole,
    action: () => Effect.Effect<A, ControlPlaneError>,
    hint: string | null = null,
  ) {
    return Effect.flatMap(
      this.attempt(async () => this.roleOrFail(graphId, needed, hint), "Unable to check access"),
      () => action(),
    );
  }

  private async roleOrFail(graphId: string, needed: GraphRole, hint: string | null = null): Promise<GraphRole> {
    const state = await loadAccess(this.stateRoot);
    const owner = (await this.ownerOf(graphId, state)) ?? hint;
    const role = roleFor(effectiveRecord(state, graphId, owner), this.who, this.policy);
    if (!role) {
      throw forbidden(NO_ACCESS);
    }
    if (!allows(role, needed)) {
      throw forbidden(NEEDS[needed]);
    }
    return role;
  }

  private async view(graphId: string): Promise<GraphAccessView> {
    const role = await this.roleOrFail(graphId, "read");
    const state = await loadAccess(this.stateRoot);
    const owner = await this.ownerOf(graphId, state);
    const record = recordFor(state, graphId);
    return {
      graphId,
      owner,
      role,
      shares: [...(record?.grants ?? [])].sort((left, right) => left.principal.localeCompare(right.principal)),
    };
  }

  /** The graph's owner: the store's, else whoever submitted its earliest run. */
  private async ownerOf(graphId: string, loaded?: AccessState): Promise<string | null> {
    const state = loaded ?? (await loadAccess(this.stateRoot));
    const stored = recordFor(state, graphId)?.owner;
    if (stored) {
      return stored;
    }
    return firstOwners(await this.allRuns()).get(graphId) ?? null;
  }

  /** Anywhere a graph can be recorded, not only in the runtime this request names. */
  private async exists(graphId: string): Promise<boolean> {
    if (await graphIsKnown(this.stateRoot, graphId)) {
      return true;
    }
    return (await this.allRuns()).some((run) => run.graphId === graphId);
  }

  /**
   * What a run, a preflight or a preview may name: a run id nobody has used
   * (writing under a taken one would overwrite that run's record and
   * configuration), and a source the viewer may read.
   */
  private checkRequest(request: IngestionRequest): Effect.Effect<void, ControlPlaneError> {
    return Effect.flatMap(
      this.attempt(async () => {
        if (await runIsKnown(this.stateRoot, request.jobId)) {
          throw invalid("That run id is already taken; start the run again for a new one.");
        }
      }, "Unable to check the run"),
      () => (request.revision && !request.revision.addDocuments ? Effect.void : this.checkSource(request.source)),
    );
  }

  /**
   * A folder source may not reach into the console's own records. Inside the
   * state root, an upload folder is readable only by the person who uploaded
   * it, or by someone who can read a graph built from it; runs, graphs,
   * artifacts, the access store and the other records are never a source.
   * Any other folder a host lets a request name - the checkout, the
   * operator's source roots, a folder an operator placed in the state root -
   * is allowed as before.
   */
  private checkSource(source: Record<string, unknown> | undefined): Effect.Effect<void, ControlPlaneError> {
    return this.attempt(async () => {
      const kind = String(source?.kind ?? "local");
      const named = String(source?.path ?? "").trim();
      if ((kind !== "local" && kind !== "upload" && kind !== "local_path") || !named) {
        return;
      }
      const root = path.resolve(this.stateRoot);
      const target = path.resolve(named);
      if (!isWithin(root, target)) {
        return;
      }
      const [area, jobId] = path.relative(root, target).split(path.sep);
      if (area === "uploads" && jobId) {
        if (!(await this.mayReadUpload(jobId))) {
          throw forbidden("That upload folder belongs to someone else.");
        }
        return;
      }
      if (!area || CONSOLE_RECORDS.has(area) || (await isFile(path.join(root, area)))) {
        throw forbidden("Documents come from your uploads or a documents folder, not the console's own records.");
      }
    }, "Unable to check the source");
  }

  private async mayReadUpload(jobId: string): Promise<boolean> {
    const me = principalKey(this.who.userName);
    if (!me && this.policy.laptop && !this.policy.strict) {
      return true;
    }
    const owner = (await loadAccess(this.stateRoot)).uploads[jobId]?.owner;
    if (me && owner === me) {
      return true;
    }
    // The documents behind a graph the viewer can read, reused for a new one.
    const folder = path.resolve(this.stateRoot, "uploads", jobId);
    const visible = await Effect.runPromise(this.visibleGraphs());
    return (await this.allRuns()).some((run) => {
      const used = sourcePathOf(run);
      return Boolean(used) && visible.has(run.graphId) && isWithin(folder, path.resolve(used!));
    });
  }

  private allRuns(): Promise<readonly RunSnapshot[]> {
    this.runsMemo ??= Effect.runPromise(this.inner.listRuns(RUN_SCAN_LIMIT));
    return this.runsMemo;
  }

  private attempt<A>(run: () => Promise<A>, fallback: string): Effect.Effect<A, ControlPlaneError> {
    return Effect.tryPromise({ try: run, catch: (cause) => fromCause(cause, fallback) });
  }
}

/** Record that someone used the console, for the people a share can name. */
export async function notePerson(stateRoot: string, viewer: Viewer): Promise<void> {
  const me = principalKey(viewer.userName);
  if (!me) {
    return;
  }
  const state = await loadAccess(stateRoot);
  if (!sawPerson(state, me)) {
    return;
  }
  await updateAccess(stateRoot, (current) => {
    sawPerson(current, me);
  });
}

/** What the console keeps in its state root for itself: never a document source. */
const CONSOLE_RECORDS = new Set(["uploads", "runs", "graphs", "artifacts", "browse", "access", "gold", "kubernetes", "snowflake"]);

async function isFile(candidate: string): Promise<boolean> {
  try {
    return (await stat(candidate)).isFile();
  } catch {
    return false;
  }
}

/** Record who created an upload folder, so its documents are theirs to build from. */
export async function claimUploadFolder(stateRoot: string, jobId: string, viewer: Viewer): Promise<boolean> {
  const me = principalKey(viewer.userName);
  return updateAccess(stateRoot, (state) => {
    const owner = state.uploads[jobId]?.owner;
    if (owner) {
      return owner === me;
    }
    if (me) {
      claimUpload(state, jobId, me);
    }
    return true;
  });
}

/** The folder a run read its documents from, as its record names it. */
function sourcePathOf(run: RunSnapshot): string | null {
  const raw = run.raw as Record<string, unknown>;
  const source = raw.source as Record<string, unknown> | undefined;
  const value = raw.sourcePath ?? source?.path;
  return typeof value === "string" && value.trim() ? value : null;
}

function ownerHint(run: RunSnapshot): string | null {
  const value = run.owner ?? (run.raw as Record<string, unknown> | undefined)?.owner;
  return typeof value === "string" && value.trim() ? principalKey(value) : null;
}

/** Each graph's earliest recorded submitter; listings run newest first. */
function firstOwners(runs: readonly RunSnapshot[]): Map<string, string> {
  const owners = new Map<string, string>();
  for (const run of runs) {
    const owner = ownerHint(run);
    if (owner) {
      owners.set(run.graphId, owner);
    }
  }
  return owners;
}

function effectiveRecord(state: AccessState, graphId: string, fallbackOwner: string | null): GraphAccessRecord {
  const stored = recordFor(state, graphId);
  return {
    owner: stored?.owner ?? fallbackOwner,
    grants: stored?.grants ?? [],
  };
}

function annotate(run: RunSnapshot, record: GraphAccessRecord, role: GraphRole | null): RunSnapshot {
  return {
    ...run,
    owner: record.owner,
    access: role,
    sharedWith: record.grants.length,
    raw: { ...run.raw, owner: record.owner },
  };
}
