import { Effect } from "effect";
import {
  type ClusterProfile,
  type ClusterSnapshot,
  type GraphDataset,
  type GraphShare,
  type IngestionRequest,
  type NodeWorkAssignment,
  type PreflightResult,
  type RunSnapshot,
  type RuntimeMode,
  type SourceObject,
  type Viewer,
  type Capability,
} from "./schema";
import { ControlPlaneError } from "./errors";
import type { DocumentStatus } from "../documents";

export interface ControlPlane {
  readonly runtime: RuntimeMode;
  readonly capabilities: ReadonlySet<Capability>;
  viewer(): Effect.Effect<Viewer, ControlPlaneError>;
  listSourceObjects(
    source: Record<string, unknown>,
    limit?: number,
  ): Effect.Effect<readonly SourceObject[], ControlPlaneError>;
  preflight(request: IngestionRequest): Effect.Effect<PreflightResult, ControlPlaneError>;
  submit(request: IngestionRequest): Effect.Effect<RunSnapshot, ControlPlaneError>;
  listRuns(limit?: number): Effect.Effect<readonly RunSnapshot[], ControlPlaneError>;
  getRun(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError>;
  /** Where each of a run's documents stands, by whatever the runtime records. */
  documents(runId: string): Effect.Effect<readonly DocumentStatus[], ControlPlaneError>;
  cancel(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError>;
  retry(runId: string): Effect.Effect<RunSnapshot, ControlPlaneError>;
  recover(runId: string): Effect.Effect<string, ControlPlaneError>;
  forget(runId: string): Effect.Effect<void, ControlPlaneError>;
  renameGraph(graphId: string, displayName: string): Effect.Effect<string, ControlPlaneError>;
  loadGraph(location: string, graphId?: string | null): Effect.Effect<GraphDataset, ControlPlaneError>;
  loadRunGraph(snapshot: RunSnapshot): Effect.Effect<GraphDataset, ControlPlaneError>;
  cluster(namespace: string): Effect.Effect<ClusterSnapshot | null, ControlPlaneError>;
  nodeAssignments(
    namespace: string,
    nodeName: string,
  ): Effect.Effect<readonly NodeWorkAssignment[], ControlPlaneError>;
  listClusters(): Effect.Effect<readonly ClusterProfile[], ControlPlaneError>;
  upsertCluster(profile: ClusterProfile): Effect.Effect<ClusterProfile, ControlPlaneError>;
  deleteCluster(name: string): Effect.Effect<void, ControlPlaneError>;
  selectCluster(name: string): Effect.Effect<ClusterProfile, ControlPlaneError>;
  selectedCluster(): Effect.Effect<ClusterProfile | null, ControlPlaneError>;
  graphOwner(graphId: string): Effect.Effect<string | null, ControlPlaneError>;
  graphShares(graphId: string): Effect.Effect<readonly GraphShare[], ControlPlaneError>;
  shareGraph(
    graphId: string,
    granteeType: string,
    grantee: string,
  ): Effect.Effect<void, ControlPlaneError>;
  unshareGraph(
    graphId: string,
    granteeType: string,
    grantee: string,
  ): Effect.Effect<void, ControlPlaneError>;
  deleteGraph(graphId: string): Effect.Effect<Record<string, number>, ControlPlaneError>;
  previewConfig(request: IngestionRequest): Effect.Effect<string, ControlPlaneError>;
}

export function hasCapability(plane: ControlPlane, capability: Capability): boolean {
  return plane.capabilities.has(capability);
}
