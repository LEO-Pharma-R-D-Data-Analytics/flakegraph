"use client";

import { Fragment, useEffect, useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { StatusBadge } from "@/components/console/sidebar";
import { GraphExplorer } from "@/components/console/graph-explorer";
import { GuideCard } from "@/components/console/guide-card";
import { GraphEditor, GraphVersionsCard } from "@/components/console/graph-editor";
import { PageHeader } from "@/components/console/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { AskPanel, ReviewPanel, VersionsPanel, WatchPanel } from "@/components/console/workspace-panels";
import { PromotionCard } from "@/components/console/operator-tools";
import {
  ARTIFACTS_UNAVAILABLE_STATUS,
  graphCounts,
  isActiveStatus,
  isSuccessStatus,
  type Capability,
  type GraphDataset,
  type RunSnapshot,
  type StageProgress,
} from "@/server/protocol/schema";
import { readLastIngestion, writeLastIngestion } from "@/lib/last-ingestion";
import { formatDocumentPhase, documentPhaseNeedsSkip, formatDuration, formatInstant, formatRelativeTime, titleCaseStage } from "@/lib/utils";
import type { DocumentStatus } from "@/server/documents";
import { statusSentence } from "@/lib/status-sentence";

export function RunWorkspace({
  runId,
  capabilities,
  runtime = "local",
  role = "operator",
  suggestionMode: _suggestionMode = "on-request",
  jumpSearch = "",
  onJumpConsumed,
  onDeleted,
  onNewGraph,
  onPromote,
  onOpenRun,
}: {
  runId: string;
  capabilities: Set<Capability>;
  runtime?: "local" | "kubernetes" | "snowflake";
  role?: "operator" | "analyst" | "staff";
  suggestionMode?: "off" | "on-request" | "auto-fill";
  jumpSearch?: string;
  onJumpConsumed?: () => void;
  onDeleted?: () => Promise<void> | void;
  onNewGraph?: () => Promise<void> | void;
  onPromote?: (runtime: "local" | "kubernetes" | "snowflake") => Promise<void> | void;
  /** Open another run of this catalog, such as another version of the graph. */
  onOpenRun?: (runId: string) => Promise<void> | void;
}) {
  const utils = trpc.useUtils();
  const run = trpc.runs.get.useQuery(
    { runId },
    { refetchInterval: (query) => (isActiveStatus(query.state.data?.status ?? "") ? 3_000 : false) },
  );
  const graph = trpc.graphs.loadRun.useQuery(
    { runId },
    { enabled: isSuccessStatus(run.data?.status ?? "") },
  );
  const quality = trpc.graphs.quality.useQuery(
    { runId },
    { enabled: isSuccessStatus(run.data?.status ?? "") },
  );
  const shares = trpc.graphs.shares.useQuery(
    { graphId: run.data?.graphId ?? "" },
    { enabled: Boolean(run.data && capabilities.has("share")) },
  );
  const owner = trpc.graphs.owner.useQuery(
    { graphId: run.data?.graphId ?? "" },
    { enabled: Boolean(run.data && capabilities.has("share")) },
  );
  // Cancel and retry answer with the run as the control plane now sees it;
  // that answer goes into the page at once, and the refetch behind it only
  // confirms. Otherwise the page shows the old state for as long as one
  // more status read takes, which reads as the click having done nothing.
  const cancel = trpc.runs.cancel.useMutation({
    onSuccess: async (snapshot) => {
      toast.success("Cancellation requested");
      utils.runs.get.setData({ runId }, snapshot);
      await Promise.all([run.refetch(), utils.runs.list.invalidate()]);
    },
    onError: (error) => toast.error(error.message),
  });
  const retry = trpc.runs.retry.useMutation({
    onSuccess: async (snapshot) => {
      toast.success(run.data?.status.toLowerCase() === "cancelled" ? "Run resumed" : "Retry submitted");
      utils.runs.get.setData({ runId }, snapshot);
      await Promise.all([run.refetch(), utils.runs.list.invalidate()]);
    },
    onError: (error) => toast.error(error.message),
  });
  const recover = trpc.runs.recover.useMutation({
    onSuccess: async (message) => {
      toast.success(message);
      await Promise.all([run.refetch(), utils.runs.list.invalidate()]);
    },
    onError: (error) => toast.error(error.message),
  });
  const rename = trpc.graphs.rename.useMutation({
    onSuccess: async (name) => {
      toast.success(`Renamed to ${name}`);
      await Promise.all([run.refetch(), utils.runs.list.invalidate()]);
    },
    onError: (error) => toast.error(error.message),
  });
  const share = trpc.graphs.share.useMutation({
    onSuccess: () => {
      toast.success("Share granted");
      void shares.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const unshare = trpc.graphs.unshare.useMutation({
    onSuccess: () => {
      toast.success("Share withdrawn");
      void shares.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const remove = trpc.graphs.delete.useMutation({
    onSuccess: async () => {
      toast.success("Graph deleted");
      await onDeleted?.();
    },
    onError: (error) => toast.error(error.message),
  });
  const forget = trpc.runs.forget.useMutation({
    onSuccess: async () => {
      toast.success("Removed from this catalog. Stored files were not deleted.");
      await onDeleted?.();
    },
    onError: (error) => toast.error(error.message),
  });
  const documents = trpc.runs.documents.useQuery(
    { runId },
    { refetchInterval: isActiveStatus(run.data?.status ?? "") ? 5_000 : false },
  );
  const skipFile = trpc.runs.skipFile.useMutation({
    onSuccess: async () => {
      toast.success("File quarantined. Later retries skip it.");
      await documents.refetch();
    },
  });
  const workspace = trpc.workspace.get.useQuery();
  const pin = trpc.graphs.pin.useMutation({
    onSuccess: async () => {
      toast.success("Pinned as looks wrong. This is not a delete.");
      await workspace.refetch();
    },
  });
  const [renameOpen, setRenameOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [forgetGuide, setForgetGuide] = useState(false);
  const [nextName, setNextName] = useState("");
  const [granteeType, setGranteeType] = useState("USER");
  const [grantee, setGrantee] = useState("");
  const [exploreSeed, setExploreSeed] = useState("");
  const [workspaceTab, setWorkspaceTab] = useState<string | null>(null);

  // A jump lands in the explorer with its search; the shell is told once the
  // jump has been taken so it does not fire again on the next graph.
  const [jumpTaken, setJumpTaken] = useState("");
  if (jumpSearch && jumpTaken !== jumpSearch) {
    setJumpTaken(jumpSearch);
    setExploreSeed(jumpSearch);
    setWorkspaceTab("explore");
  } else if (!jumpSearch && jumpTaken) {
    setJumpTaken("");
  }
  useEffect(() => {
    if (jumpSearch) {
      onJumpConsumed?.();
    }
  }, [jumpSearch, onJumpConsumed]);

  if (run.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-24 w-full" />
        <p className="text-sm text-muted-foreground">Loading run…</p>
      </div>
    );
  }
  if (run.error || !run.data) {
    const hidden = run.error?.message?.toLowerCase().includes("not visible");
    return (
      <div className="space-y-3">
        <PageHeader
          kicker="Graph"
          title={hidden ? "This graph is hidden" : "Run not found"}
          description={
            hidden
              ? "You are not on the ACL for this graph. Sign in from the identity chip as the owner or a grantee."
              : "Pick a graph from the sidebar, or go back to New graph."
          }
        />
        <Alert variant="destructive">{run.error?.message ?? "Unknown run"}</Alert>
        {hidden || !onNewGraph ? null : <Button onClick={() => void onNewGraph()}>New graph</Button>}
      </div>
    );
  }

  const snapshot = run.data;
  const status = snapshot.status.toLowerCase();
  const operator = role !== "analyst";
  const canRevise = operator && capabilities.has("revise") && isSuccessStatus(status);
  const nodeCount = graph.data ? graphCounts(graph.data).nodes : null;
  const sentence = statusSentence(snapshot, { nodeCount });
  const shareBlocked = quality.data?.shareBlockedReason;
  const zeroEntities = quality.data?.zeroEntityFailure || nodeCount === 0;

  function cloneToCompose() {
    const last = readLastIngestion(runtime);
    const raw = snapshot.raw ?? {};
    writeLastIngestion({
      runtime,
      graphName: snapshot.graphName || last?.graphName || "",
      sourceKind: String(raw.sourceKind || last?.sourceKind || "local_path"),
      sourcePath: String(raw.sourcePath || watchPrefixFrom(snapshot) || last?.sourcePath || ""),
      ...(raw.source && typeof raw.source === "object"
        ? { source: raw.source as Record<string, unknown> }
        : last?.source
          ? { source: last.source }
          : {}),
      ocrProvider: String(raw.ocrProvider || last?.ocrProvider || "fallback"),
      llmProvider: String(raw.llmProvider || last?.llmProvider || "vllm_local"),
      embeddingProvider: String(raw.embeddingProvider || last?.embeddingProvider || "sentence_transformers"),
      savedAt: new Date().toISOString(),
    });
    if (typeof window !== "undefined") {
      sessionStorage.setItem("flakegraph.pending-clone", "1");
    }
    return onNewGraph?.();
  }

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Graph"
        title={snapshot.graphName || snapshot.graphId}
        description={
          <>
            {snapshot.graphId} · {snapshot.storageKind === "snowflake" ? snapshot.storageLocation : snapshot.outputPath} ·{" "}
            {formatRelativeTime(snapshot.updatedAt)}
            {versionLine(snapshot) ? <> · {versionLine(snapshot)}</> : null}
          </>
        }
        actions={
          <>
            {status === "failed" || zeroEntities ? null : <StatusBadge status={snapshot.status} />}
            {operator && capabilities.has("rename") ? (
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  setNextName(snapshot.graphName || snapshot.graphId);
                  setRenameOpen(true);
                }}
              >
                Rename
              </Button>
            ) : null}
            {operator && capabilities.has("recover") && isActiveStatus(status) ? (
              <Button size="sm" variant="outline" onClick={() => recover.mutate({ runId })}>
                Recover workers
              </Button>
            ) : null}
            {isSuccessStatus(status) && !capabilities.has("share") ? (
              <Button size="sm" variant="outline" onClick={() => exportBundle(snapshot, graph.data, quality.data)}>
                Export review bundle
              </Button>
            ) : null}
            {operator && capabilities.has("delete_graph") ? (
              <Button size="sm" variant="destructive" onClick={() => setDeleteOpen(true)}>
                Delete graph
              </Button>
            ) : null}
          </>
        }
      />
      <p className="text-sm" data-testid="status-sentence">
        {sentence}
      </p>

      {guideForRun({
        status,
        sentence,
        zeroEntities: Boolean(zeroEntities),
        shareBlocked,
        canShare: capabilities.has("share"),
        canRetry: capabilities.has("retry"),
        canCancel: capabilities.has("cancel") && isActiveStatus(status),
        onRetry: () => retry.mutate({ runId }),
        onCancel: () => cancel.mutate({ runId }),
        onExport: () => exportBundle(snapshot, graph.data, quality.data),
        onNewGraph,
        onCloneConfig: () => cloneToCompose(),
        onForget:
          operator && capabilities.has("forget")
            ? () => {
                if (!forgetGuide) {
                  setForgetGuide(true);
                  return;
                }
                forget.mutate({ runId });
              }
            : undefined,
        forgetLabel: forgetGuide ? "Confirm forget this graph" : "Forget this graph",
        onOpenQuality: () => setWorkspaceTab("quality"),
        staleQueue: isStaleQueue(snapshot),
      })}

      {snapshot.error && !sentence.includes(snapshot.error) ? (
        <Alert variant="destructive">{snapshot.error}</Alert>
      ) : null}

      {status === ARTIFACTS_UNAVAILABLE_STATUS ? (
        <Alert>
          This graph completed, but its files are not available on this host. The catalog keeps the record so the
          artifacts can still exist where they were written.
        </Alert>
      ) : null}

      {zeroEntities && isSuccessStatus(status) && !sentence.toLowerCase().includes("0 entities") ? (
        <Alert variant="destructive">
          Pipeline finished with 0 entities. This is not a succeeded graph. Check OCR pages or the schema, then retry.
        </Alert>
      ) : null}

      {isActiveStatus(status) || status === "interrupted" ? (
        <ProgressPanel
          sentence={sentence}
          stages={snapshot.stages}
          documentsCompleted={snapshot.documentsCompleted}
          documentsTotal={snapshot.documentsTotal}
          estimate={workspace.data?.estimates[runId] ?? null}
          documents={documents.data ?? []}
          onSkip={(fileId) => skipFile.mutate({ runId, fileId })}
        />
      ) : null}

      {isSuccessStatus(status) ? (
        <Tabs value={workspaceTab ?? (zeroEntities ? "quality" : "explore")} onValueChange={setWorkspaceTab}>
          <TabsList className="h-auto min-h-9 w-auto max-w-full flex-wrap justify-start">
            <TabsTrigger value="explore">Explore</TabsTrigger>
            <TabsTrigger value="ask">Ask</TabsTrigger>
            <TabsTrigger value="quality">Quality</TabsTrigger>
            <TabsTrigger value="review">Review</TabsTrigger>
            <TabsTrigger value="versions">Versions</TabsTrigger>
            {canRevise ? <TabsTrigger value="edit">Edit</TabsTrigger> : null}
            <TabsTrigger value="details">Run details</TabsTrigger>
            {operator && capabilities.has("share") ? <TabsTrigger value="sharing">Sharing</TabsTrigger> : null}
          </TabsList>
          <TabsContent
            value="explore"
            forceMount
            hidden={(workspaceTab ?? (zeroEntities ? "quality" : "explore")) !== "explore"}
            className={(workspaceTab ?? (zeroEntities ? "quality" : "explore")) === "explore" ? undefined : "hidden"}
          >
            {graph.isLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-9 w-80" />
                <Skeleton className="h-72 w-full" />
                <p className="text-sm text-muted-foreground">Loading graph…</p>
              </div>
            ) : null}
            {graph.error ? <Alert variant="destructive">{graph.error.message}</Alert> : null}
            {graph.data ? (
              <GraphExplorer
                dataset={graph.data}
                perspectives={(workspace.data?.perspectives ?? []).filter((item) => item.graphId === snapshot.graphId)}
                analyst={role === "analyst"}
                estimate={workspace.data?.estimates[runId] ?? null}
                missingGold={quality.data?.gold?.missingRequired ?? []}
                goldPending={quality.isLoading}
                goldPresent={Boolean(quality.data?.gold)}
                initialSearch={exploreSeed}
                onPin={(targetId, kind) => pin.mutate({ graphId: snapshot.graphId, targetId, kind, note: "looks wrong" })}
              />
            ) : null}
          </TabsContent>
          <TabsContent value="ask">
            <AskPanel
              runId={runId}
              perspectives={(workspace.data?.perspectives ?? []).filter(
                (item) => item.graphId === snapshot.graphId && (operator || item.lifecycle === "production"),
              )}
              onOpenEntity={(name) => {
                setExploreSeed(name);
                setWorkspaceTab("explore");
              }}
            />
          </TabsContent>
          <TabsContent value="quality">
            <QualityPanel
              report={quality.data}
              loading={quality.isLoading}
              error={quality.error?.message}
              inspectHref={`/api/inspect?run=${runId}`}
              canShare={capabilities.has("share")}
            />
          </TabsContent>
          <TabsContent value="review">
              <ReviewPanel graphId={snapshot.graphId} runId={runId} canExport={!capabilities.has("share")} />
          </TabsContent>
          {canRevise ? (
            <TabsContent value="edit">
              <GraphEditor
                snapshot={snapshot}
                documents={documents.data ?? []}
                runtime={runtime}
                capabilities={capabilities}
                onSubmitted={onOpenRun}
              />
            </TabsContent>
          ) : null}
          <TabsContent value="versions">
            <div className="space-y-4">
              {capabilities.has("revise") ? (
                // The fleet keeps the graph's versions itself, and a new one
                // is built from the Edit tab; the workspace's own version
                // labels and watches belong to runtimes without either.
                <GraphVersionsCard graphId={snapshot.graphId} runId={runId} onOpenRun={onOpenRun} />
              ) : (
                <>
                  <VersionsPanel graphId={snapshot.graphId} canPublish={operator} />
                  {operator ? (
                    <WatchPanel
                      graphId={snapshot.graphId}
                      sourcePath={watchPrefixFrom(snapshot)}
                      onProcessNow={() => void cloneToCompose()}
                    />
                  ) : null}
                </>
              )}
              {operator ? (
                <PromotionCard
                  fromRuntime={runtime}
                  graphId={snapshot.graphId}
                  graphName={snapshot.graphName || snapshot.graphId}
                  onApplied={onPromote}
                />
              ) : null}
            </div>
          </TabsContent>
          <TabsContent value="details">
            <RunDetails snapshot={snapshot} documents={documents.data ?? []} onSkip={(fileId) => skipFile.mutate({ runId, fileId })} />
          </TabsContent>
            {operator && capabilities.has("share") ? (
            <TabsContent value="sharing">
              <Card>
                {shareBlocked ? null : (
                  <CardHeader>
                    <CardTitle>Share this graph</CardTitle>
                    <CardDescription>
                      Grantees receive this graph’s scene and tables. They do not receive raw files, prompts, or the OCR dump.
                      Application roles, not ACCOUNTADMIN, should own day-to-day sharing.
                    </CardDescription>
                  </CardHeader>
                )}
                <CardContent className="space-y-3">
                  {shareBlocked ? (
                    <Alert variant="destructive">
                      {shareBlocked} Open Quality, then come back. Share stays disabled until that gate is green.
                    </Alert>
                  ) : (
                    <>
                      <p className="text-sm">
                        Owner: <span className="font-medium">{owner.data || "unowned"}</span>
                        {" · Quality allows Share"}
                      </p>
                      <div className="flex flex-wrap items-end gap-2">
                        <label className="grid gap-1.5 text-sm">
                          <span className="font-medium">Grantee type</span>
                          <Select value={granteeType} onValueChange={setGranteeType}>
                            <SelectTrigger className="w-28" aria-label="Grantee type">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="USER">USER</SelectItem>
                              <SelectItem value="ROLE">ROLE</SelectItem>
                            </SelectContent>
                          </Select>
                        </label>
                        <label className="grid gap-1.5 text-sm">
                          <span className="font-medium">Grantee</span>
                          <Input
                            className="w-64"
                            value={grantee}
                            onChange={(event) => setGrantee(event.target.value)}
                            aria-label="Grantee"
                          />
                        </label>
                        <Button
                          disabled={share.isPending || !grantee.trim()}
                          onClick={() => share.mutate({ graphId: snapshot.graphId, granteeType, grantee: grantee.trim() })}
                        >
                          Share
                        </Button>
                      </div>
                      <p className="text-sm" data-testid="share-preview">
                        {grantee.trim()
                          ? `${granteeType} ${grantee.trim().toUpperCase()} will see graph ${snapshot.graphName || snapshot.graphId}. ACL compared this exact string. They get scene and tables, not raw files.`
                          : "Type a ROLE or USER to preview the ACL change before you grant it."}
                      </p>
                    </>
                  )}
                  <details className="text-xs text-muted-foreground">
                    <summary className="cursor-pointer font-medium text-foreground">Limits</summary>
                    <p className="mt-2">
                      Preview uses the same owner’s-rights predicate as the API. Owner’s-rights SQL still sees every graph the
                      service role can; this dialog cannot close that hole.
                    </p>
                  </details>
                  <ul className="space-y-2">
                    {(shares.data ?? []).length === 0 ? (
                      <li className="text-sm text-muted-foreground">No grantees yet.</li>
                    ) : null}
                    {(shares.data ?? []).map((item) => (
                      <li key={`${item.granteeType}:${item.grantee}`} className="flex items-center justify-between gap-3 text-sm">
                        <span>
                          {item.granteeType} {item.grantee}
                        </span>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() =>
                            unshare.mutate({
                              graphId: snapshot.graphId,
                              granteeType: item.granteeType,
                              grantee: item.grantee,
                            })
                          }
                        >
                          Unshare
                        </Button>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </TabsContent>
          ) : null}
        </Tabs>
      ) : null}

      {!isActiveStatus(status) && !isSuccessStatus(status) && status !== ARTIFACTS_UNAVAILABLE_STATUS ? (
        <RunDetails snapshot={snapshot} />
      ) : null}

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename graph</DialogTitle>
            <DialogDescription>The stable graph ID does not change.</DialogDescription>
          </DialogHeader>
          <Input
            value={nextName}
            onChange={(event) => setNextName(event.target.value)}
            aria-label="Graph name"
          />
          <Button
            type="button"
            disabled={rename.isPending || !nextName.trim()}
            onClick={async () => {
              await rename.mutateAsync({ graphId: snapshot.graphId, displayName: nextName });
              setRenameOpen(false);
            }}
          >
            Save name
          </Button>
        </DialogContent>
      </Dialog>
      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete stored graph</DialogTitle>
            <DialogDescription>
              This removes the graph from Snowflake storage for this account. Forgetting history is catalog hygiene, not this delete.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="destructive"
              disabled={remove.isPending}
              onClick={async () => {
                await remove.mutateAsync({ graphId: snapshot.graphId });
                setDeleteOpen(false);
              }}
            >
              Confirm delete
            </Button>
            <Button variant="outline" onClick={() => setDeleteOpen(false)}>
              Cancel
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function guideForRun(options: {
  status: string;
  sentence: string;
  zeroEntities: boolean;
  shareBlocked: string | null | undefined;
  canShare: boolean;
  canRetry: boolean;
  canCancel: boolean;
  onRetry: () => void;
  onCancel: () => void;
  onExport: () => void;
  onNewGraph?: () => Promise<void> | void;
  onCloneConfig?: () => Promise<void> | void;
  onForget?: () => void;
  forgetLabel?: string;
  onOpenQuality?: () => void;
  staleQueue: boolean;
}) {
  if (options.zeroEntities) {
    return (
      <GuideCard
        tone="warning"
        title="Inspect OCR before you share"
        why="The pipeline is green, but entity count is 0. A hairball would hide this."
        actions={[{ label: "Open Quality", onClick: () => options.onOpenQuality?.(), variant: "secondary" }]}
      />
    );
  }
  if (options.status === "failed" || options.status === "interrupted") {
    return (
      <GuideCard
        tone="warning"
        title={options.canRetry ? "Retry failed documents" : "This run did not finish"}
        why={options.canRetry ? options.sentence : "Clone the config, or forget this graph from the catalog."}
        actions={[
          ...(options.canRetry ? [{ label: "Retry failed documents", onClick: options.onRetry }] : []),
          { label: "New graph from this config", onClick: () => void options.onCloneConfig?.(), variant: "secondary" as const },
          ...(options.onForget
            ? [
                {
                  label: options.forgetLabel ?? "Forget this graph",
                  onClick: options.onForget,
                  variant: "outline" as const,
                },
              ]
            : []),
        ]}
      />
    );
  }
  if (options.status === "cancelled") {
    return (
      <GuideCard
        title={options.canRetry ? "Resume this run where it stopped" : "This run was cancelled"}
        why={
          options.canRetry
            ? "Work that finished before the cancellation is kept; only what was stopped is queued again."
            : "Clone the config, or forget this graph from the catalog."
        }
        actions={[
          ...(options.canRetry ? [{ label: "Resume run", onClick: options.onRetry }] : []),
          { label: "New graph from this config", onClick: () => void options.onCloneConfig?.(), variant: "secondary" as const },
          ...(options.onForget
            ? [
                {
                  label: options.forgetLabel ?? "Forget this graph",
                  onClick: options.onForget,
                  variant: "outline" as const,
                },
              ]
            : []),
        ]}
      />
    );
  }
  if (options.status === "cancelling") {
    return (
      <GuideCard
        title="Cancellation is in progress"
        why={options.sentence}
        actions={[]}
      />
    );
  }
  if (isActiveStatus(options.status) && options.canCancel) {
    return (
      <GuideCard
        title={options.staleQueue ? "This job has not started" : "This job is still moving"}
        why={options.sentence}
        actions={[{ label: "Cancel job", onClick: options.onCancel, variant: "outline" }]}
      />
    );
  }
  if (isSuccessStatus(options.status) && options.canShare && options.shareBlocked) {
    return (
      <GuideCard
        title="Quality blocks Share"
        why={options.shareBlocked}
        actions={[{ label: "Open Quality", onClick: () => options.onOpenQuality?.(), variant: "secondary" }]}
      />
    );
  }
  return null;
}

function QualityPanel({
  report,
  loading,
  error,
  inspectHref,
  canShare = false,
}: {
  report: {
    nodeCount: number;
    edgeCount: number;
    documentCount: number;
    documentsFailed: number;
    zeroEntityFailure: boolean;
    shareBlockedReason: string | null;
    gold: {
      goldName: string;
      expectedEntities: number;
      foundEntities: number;
      expectedRelations: number;
      foundRelations: number;
      requiredTotal: number;
      matchedRequired: number;
      missingRequired: Array<{ id: string; source: string; target: string; relationType: string }>;
    } | null;
  } | undefined;
  loading: boolean;
  error?: string;
  inspectHref?: string;
  canShare?: boolean;
}) {
  if (loading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-32 w-full" />
        <p className="text-sm text-muted-foreground">Comparing to gold…</p>
      </div>
    );
  }
  if (error) {
    return <Alert variant="destructive">{error}</Alert>;
  }
  if (!report) {
    return <p className="text-sm text-muted-foreground">No quality report for this run.</p>;
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>Quality</CardTitle>
        <CardDescription>
          Gold is a QA contract. It is not stuffed into RAG context. Missing required relations block Share.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <p>
          {report.documentCount} documents · {report.nodeCount} entities · {report.edgeCount} relations
          {report.documentsFailed ? ` · ${report.documentsFailed} failed documents` : ""}
        </p>
        {report.zeroEntityFailure ? (
          <Alert variant="destructive">0 entities after a green pipeline. Treat as OCR or schema failure, not Succeeded.</Alert>
        ) : null}
        <p className="font-medium">
          {canShare ? (report.shareBlockedReason ? "Share blocked" : "Share allowed") : "Gold is the QA bar for this graph"}
        </p>
        {report.gold ? (
          <>
            <p>
              Gold “{report.gold.goldName}”: {report.gold.foundEntities}/{report.gold.expectedEntities} entities,{" "}
              {report.gold.matchedRequired}/{report.gold.requiredTotal} required relations.
            </p>
            {report.gold.missingRequired.length > 0 ? (
              <ul className="list-disc pl-5">
                {report.gold.missingRequired.slice(0, 12).map((item) => (
                  <li key={item.id}>
                    {item.source} —{item.relationType}→ {item.target}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-muted-foreground">Required gold relations are present.</p>
            )}
          </>
        ) : (
          <p className="text-muted-foreground">No gold.json matched this graph name. Counts above are still the QA bar.</p>
        )}
        {report.shareBlockedReason ? <Alert variant="destructive">{report.shareBlockedReason}</Alert> : null}
        {inspectHref ? (
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" asChild>
              <a href={inspectHref} target="_blank" rel="noreferrer">
                Open inspect HTML
              </a>
            </Button>
            <Button size="sm" variant="ghost" asChild>
              <a href={`${inspectHref}#gold-compare`} target="_blank" rel="noreferrer">
                Compare gold
              </a>
            </Button>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function ProgressPanel({
  sentence,
  stages,
  documentsCompleted,
  documentsTotal,
  estimate,
  documents,
  onSkip,
}: {
  sentence: string;
  stages: readonly StageProgress[];
  documentsCompleted: number;
  documentsTotal: number | null;
  estimate: { usdLow: number; usdHigh: number } | null;
  documents: DocumentStatus[];
  onSkip: (fileId: string) => void;
}) {
  const kept = documents.filter((item) => item.phase === "inherited").length;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Progress</CardTitle>
        <CardDescription data-testid="blocking-sentence">{sentence}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">
          Documents {documentsCompleted}
          {documentsTotal != null ? ` / ${documentsTotal}` : ""}
          {kept > 0 ? ` · ${kept} kept from earlier versions` : ""}
          {estimate ? ` · cost so far in band ${estimate.usdLow}–${estimate.usdHigh} usd · split deserved GPU hours from idle after the run` : ""}
        </p>
        {stages.map((stage) => {
          const percent = stage.total
            ? Math.min(100, Math.round((stage.completed / stage.total) * 100))
            : stage.status === "completed" || stage.status === "succeeded"
              ? 100
              : undefined;
          return (
            <div key={stage.stage} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span>{titleCaseStage(stage.stage)}</span>
                <span className="text-muted-foreground">
                  {stage.completed}
                  {stage.total != null ? ` / ${stage.total}` : ""} · {stage.status}
                </span>
              </div>
              <Progress value={percent} />
              {stage.message ? <p className="text-xs text-muted-foreground">{stage.message}</p> : null}
            </div>
          );
        })}
        {documents.length ? (
          <ul className="space-y-1 text-sm">
            {documents.map((item) => (
              <li key={item.fileId} className="flex items-center justify-between gap-2">
                <span title={item.fileId}>
                  {item.name ?? item.fileId} · {formatDocumentPhase(item.phase)} · {item.detail}
                </span>
                {documentPhaseNeedsSkip(item.phase) ? (
                  <Button size="sm" variant="outline" onClick={() => onSkip(item.fileId)}>
                    Skip file
                  </Button>
                ) : null}
              </li>
            ))}
          </ul>
        ) : null}
      </CardContent>
    </Card>
  );
}

function RunDetails({
  snapshot,
  documents = [],
  onSkip,
}: {
  snapshot: RunSnapshot;
  documents?: DocumentStatus[];
  onSkip?: (fileId: string) => void;
}) {
  const raw = snapshot.raw as Record<string, unknown>;
  const keptCount = documents.filter((item) => item.phase === "inherited").length;
  const startedAt = Date.parse(snapshot.startedAt ?? "");
  const updatedAt = Date.parse(snapshot.updatedAt ?? "");
  const duration = Number.isFinite(startedAt) && Number.isFinite(updatedAt) ? updatedAt - startedAt : null;
  const facts: Array<[string, string]> = [
    ["Run", snapshot.runId],
    ["Graph", snapshot.graphId],
    ["Status", snapshot.status],
    ["Started", formatInstant(snapshot.startedAt)],
    [isActiveStatus(snapshot.status) ? "Last update" : "Finished", formatInstant(snapshot.updatedAt)],
    [isActiveStatus(snapshot.status) ? "Running for" : "Took", formatDuration(duration)],
    [
      "Documents",
      `${
        snapshot.documentsTotal == null
          ? `${snapshot.documentsCompleted} indexed`
          : `${snapshot.documentsCompleted} of ${snapshot.documentsTotal} indexed${snapshot.documentsFailed ? `, ${snapshot.documentsFailed} failed` : ""}`
      }${keptCount ? ` · ${keptCount} kept from earlier versions` : ""}`,
    ],
    ["Storage", `${snapshot.storageKind}${snapshot.storageLocation ? ` · ${snapshot.storageLocation}` : ""}`],
  ];
  const source = [raw.sourceKind, raw.sourcePath].filter(Boolean).map(String).join(" · ");
  if (source) {
    facts.push(["Source", source]);
  }
  const providers = [
    ["OCR", raw.ocrProvider],
    ["LLM", raw.llmProvider],
    ["Embedding", raw.embeddingProvider],
  ]
    .filter(([, value]) => Boolean(value))
    .map(([label, value]) => `${label} ${String(value)}`)
    .join(" · ");
  if (providers) {
    facts.push(["Providers", providers]);
  }
  if (typeof raw.owner === "string" && raw.owner) {
    facts.push(["Owner", raw.owner]);
  }
  return (
    <div className="space-y-4" data-testid="run-details">
      <Card>
        <CardHeader>
          <CardTitle>Run</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[max-content_1fr]">
            {facts.map(([label, value]) => (
              <Fragment key={label}>
                <dt className="text-muted-foreground">{label}</dt>
                <dd className="break-all">{value}</dd>
              </Fragment>
            ))}
          </dl>
          {snapshot.warnings.length ? (
            <ul className="mt-4 space-y-1 text-sm text-amber-700 dark:text-amber-400">
              {snapshot.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          ) : null}
        </CardContent>
      </Card>

      {snapshot.stages.length ? (
        <Card>
          <CardHeader>
            <CardTitle>Stages</CardTitle>
            <CardDescription>Work per stage, and how long the workers spent on it.</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Stage</TableHead>
                  <TableHead>Progress</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Worker time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {snapshot.stages.map((stage) => {
                  const percent = stage.total
                    ? Math.min(100, Math.round((stage.completed / stage.total) * 100))
                    : stage.status === "completed" || stage.status === "succeeded"
                      ? 100
                      : 0;
                  return (
                    <TableRow key={stage.stage}>
                      <TableCell className="font-medium">{titleCaseStage(stage.stage)}</TableCell>
                      <TableCell className="min-w-48">
                        <div className="flex items-center gap-3">
                          <Progress value={percent} className="h-2 w-28" />
                          <span className="text-muted-foreground">
                            {stage.completed}
                            {stage.total != null ? ` / ${stage.total}` : ""}
                          </span>
                        </div>
                        {stage.message ? <p className="mt-1 text-xs text-muted-foreground">{stage.message}</p> : null}
                      </TableCell>
                      <TableCell>{stage.status}</TableCell>
                      <TableCell className="text-right tabular-nums">{formatDuration(stage.elapsedMs)}</TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}

      {documents.length ? (
        <Card>
          <CardHeader>
            <CardTitle>Documents</CardTitle>
            <CardDescription>
              {documents.length} document{documents.length === 1 ? "" : "s"} the run discovered, and where each stands.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Document</TableHead>
                  <TableHead>Phase</TableHead>
                  <TableHead>Detail</TableHead>
                  {onSkip ? <TableHead /> : null}
                </TableRow>
              </TableHeader>
              <TableBody>
                {documents.map((item) => (
                  <TableRow key={item.fileId}>
                    <TableCell className="font-medium" title={item.fileId}>
                      {item.name ?? item.fileId}
                    </TableCell>
                    <TableCell>{formatDocumentPhase(item.phase)}</TableCell>
                    <TableCell className="max-w-md break-words text-muted-foreground">{item.detail}</TableCell>
                    {onSkip ? (
                      <TableCell className="text-right">
                        {documentPhaseNeedsSkip(item.phase) ? (
                          <Button size="sm" variant="outline" onClick={() => onSkip(item.fileId)}>
                            Skip file
                          </Button>
                        ) : null}
                      </TableCell>
                    ) : null}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}

      {snapshot.events.length ? (
        <Card>
          <CardHeader>
            <CardTitle>Recent events</CardTitle>
          </CardHeader>
          <CardContent>
            <ol className="space-y-2 text-sm">
              {snapshot.events.slice(-30).reverse().map((event, index) => (
                <li key={`${event.timestamp}-${index}`}>
                  <span className="text-muted-foreground">{event.timestamp}</span> {event.stage} {event.status}
                  {event.fileId ? ` · ${event.fileId}` : ""}
                  {event.message ? ` — ${event.message}` : ""}
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}

/** "Version 2 of 3 · head", from what the runtime knows of the graph's versions. */
function versionLine(snapshot: RunSnapshot): string | null {
  const version = snapshot.raw.version;
  if (!version || typeof version !== "object") {
    return null;
  }
  const { number, count, head } = version as { number: number; count: number; head: boolean };
  if (!count) {
    return null;
  }
  if (!number) {
    return `${count} published version${count === 1 ? "" : "s"} of this graph`;
  }
  return `Version ${number} of ${count}${head ? " · head" : ""}`;
}

function watchPrefixFrom(snapshot: RunSnapshot): string {
  const raw = snapshot.raw as Record<string, unknown>;
  const source = raw.source && typeof raw.source === "object" ? (raw.source as Record<string, unknown>) : {};
  return String(raw.sourcePath ?? source.path ?? source.input_path ?? raw.input_path ?? "");
}

function exportBundle(
  snapshot: RunSnapshot,
  dataset: GraphDataset | undefined,
  quality: unknown,
) {
  const payload = {
    graphId: snapshot.graphId,
    graphName: snapshot.graphName,
    status: snapshot.status,
    exportedAt: new Date().toISOString(),
    counts: dataset ? graphCounts(dataset) : null,
    nodes: dataset?.nodes ?? [],
    edges: dataset?.edges ?? [],
    evidence: dataset?.evidence ?? [],
    quality,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${snapshot.graphId}-review-bundle.json`;
  link.click();
  URL.revokeObjectURL(url);
  toast.success("Downloaded review bundle");
}

function isStaleQueue(snapshot: RunSnapshot): boolean {
  const status = snapshot.status.toLowerCase();
  if (status !== "queued" && status !== "pending" && status !== "planning") {
    return false;
  }
  const stamp = Date.parse(snapshot.startedAt || snapshot.updatedAt || "");
  return Number.isFinite(stamp) && Date.now() - stamp > 60 * 60 * 1000;
}
