"use client";

import Image from "next/image";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Boxes, ChevronRight, KeyRound, Plus, Search, Server, ShieldAlert, Trash2 } from "lucide-react";
import { trpc } from "@/components/providers";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ARTIFACTS_UNAVAILABLE_STATUS,
  isActiveStatus,
  isSuccessStatus,
  type Capability,
  type RuntimeMode,
  type RunSnapshot,
} from "@/server/protocol/schema";
import { cn, formatRelativeTime } from "@/lib/utils";
import { catalogEmptyCopy, statusSentence } from "@/lib/status-sentence";
import { toast } from "sonner";

interface SidebarProps {
  runtime: RuntimeMode;
  availableRuntimes: RuntimeMode[];
  capabilities: Set<Capability>;
  page: string;
  selectedRunId: string | null;
  identified: boolean;
  principal: string;
  role: "operator" | "analyst" | "staff";
  suggestionMode: "off" | "on-request" | "auto-fill";
  onRuntimeChange: (runtime: RuntimeMode) => void;
  onNewGraph: () => void;
  onFleet: () => void;
  onClusters: () => void;
  onStaff: () => void;
  onKeys: () => void;
  onSelectRun: (runId: string) => void;
  onForgotten?: (runId: string) => void;
  className?: string;
  onNavigate?: () => void;
  showClose?: boolean;
}

export function Sidebar(props: SidebarProps) {
  const [search, setSearch] = useState("");
  const [storage, setStorage] = useState("all");
  const [scope, setScope] = useState<"all" | "mine">("all");
  const [identityOpen, setIdentityOpen] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [pendingForget, setPendingForget] = useState<string | null>(null);
  const [confirmBulkForget, setConfirmBulkForget] = useState(false);
  const [jumpKeys, setJumpKeys] = useState("⌘K");
  const utils = trpc.useUtils();
  const runs = trpc.runs.list.useQuery({ limit: 100 }, { refetchInterval: 4_000 });
  const forget = trpc.runs.forget.useMutation({
    onSuccess: async (_void, variables) => {
      toast.success("Removed from this catalog. Stored files were not deleted.");
      await runs.refetch();
      props.onForgotten?.(variables.runId);
    },
    onError: (error) => toast.error(error.message),
  });
  const forgetMany = trpc.runs.forgetMany.useMutation({
    onSuccess: async (result, variables) => {
      toast.success(`Forgot ${result.forgotten} graphs from this catalog.`);
      setSelected([]);
      await runs.refetch();
      for (const id of variables.runIds) {
        props.onForgotten?.(id);
      }
    },
    onError: (error) => toast.error(error.message),
  });
  const assume = trpc.auth.assume.useMutation({
    onSuccess: async () => {
      await utils.invalidate();
      setIdentityOpen(false);
    },
    onError: (error) => toast.error(error.message),
  });
  const setSuggestions = trpc.auth.setSuggestions.useMutation({
    onSuccess: async () => {
      await utils.auth.session.invalidate();
    },
  });
  const analyst = props.role === "analyst";
  const workspace = trpc.workspace.get.useQuery();

  useEffect(() => {
    setJumpKeys(/Mac|iPhone|iPad/.test(navigator.platform) ? "⌘K" : "Ctrl+K");
  }, []);

  const filtered = useMemo(() => {
    const rows = runs.data ?? [];
    if (analyst) {
      const production = (workspace.data?.perspectives ?? []).filter((item) => item.lifecycle === "production");
      return production.flatMap((item) => {
        const run = rows.find((row) => row.graphId === item.graphId);
        if (!run) {
          return [];
        }
        const haystack = `${item.name} ${run.graphId}`.toLowerCase();
        if (search && !haystack.includes(search.toLowerCase())) {
          return [];
        }
        return [{ ...run, graphName: item.name }];
      });
    }
    return rows.filter((run) => {
      if (scope === "mine") {
        if (!props.identified) {
          return false;
        }
        const owner = String(run.raw?.owner ?? "").toUpperCase();
        if (owner !== props.principal.toUpperCase()) {
          return false;
        }
      }
      const haystack = `${run.graphName ?? ""} ${run.graphId} ${run.status}`.toLowerCase();
      if (search && !haystack.includes(search.toLowerCase())) {
        return false;
      }
      if (storage === "local" && run.storageKind !== "local_files") {
        return false;
      }
      if (storage === "snowflake" && run.storageKind !== "snowflake") {
        return false;
      }
      return true;
    });
  }, [analyst, runs.data, search, storage, scope, props.identified, props.principal, workspace.data?.perspectives]);

  const empty = catalogEmptyCopy({
    loading: runs.isLoading,
    total: runs.data?.length ?? 0,
    visible: filtered.length,
    search,
    storageFilter: storage,
    mine: scope === "mine",
    identified: props.identified,
    analyst,
  });

  const go = (action: () => void) => {
    action();
    props.onNavigate?.();
  };

  return (
    <aside className={cn("flex h-full w-80 shrink-0 flex-col border-r border-border bg-sidebar", props.className)}>
      <div className="relative px-3 pt-3">
        <button
          type="button"
          aria-label="FlakeGraph home"
          title="Home"
          className="block w-full rounded-md text-left hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onClick={() => go(props.onNewGraph)}
        >
          <Image
            src="/flakegraph-logo.png"
            alt=""
            width={1460}
            height={394}
            sizes="20rem"
            className="h-auto w-full object-contain"
            priority
          />
        </button>
        {props.showClose ? (
          <Button
            size="sm"
            variant="ghost"
            aria-label="Close navigation"
            className="absolute right-1 top-1"
            onClick={() => props.onNavigate?.()}
          >
            Close
          </Button>
        ) : null}
      </div>
      <div className="space-y-3 px-4 pb-3 pt-3">
        <button
          type="button"
          aria-label="Jump"
          className="flex h-8 w-full items-center gap-2 rounded-md border border-border bg-background px-2.5 text-left text-[13px] text-muted-foreground transition-colors hover:bg-muted/70 hover:text-foreground"
          onClick={() => window.dispatchEvent(new Event("flakegraph:jump"))}
        >
          <Search className="size-3.5 shrink-0 opacity-70" />
          <span className="min-w-0 flex-1 truncate">Jump</span>
          <kbd className="rounded border border-border bg-muted px-1.5 py-px font-sans text-[10px] font-medium text-muted-foreground">
            {jumpKeys}
          </kbd>
        </button>
        <div className="overflow-hidden rounded-md border border-border">
          <button
            type="button"
            data-testid="identity-chip"
            aria-label="Session identity"
            className="flex w-full items-center gap-2 px-2.5 py-2 text-left hover:bg-muted/60"
            onClick={() => setIdentityOpen(true)}
          >
            <span
              className={cn(
                "size-1.5 shrink-0 rounded-full",
                props.identified ? "bg-emerald-500" : "bg-amber-400",
              )}
            />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-[13px] font-medium leading-tight">
                {props.identified ? props.principal : "Unidentified"}
              </span>
              <span className="mt-0.5 block truncate text-[11px] leading-tight text-muted-foreground">
                {environmentLabel(props.runtime)}
                {props.identified
                  ? ""
                  : props.runtime === "snowflake"
                    ? " — private graphs hidden"
                    : props.runtime === "kubernetes"
                      ? " — cluster catalog, not tenant isolation"
                      : " — laptop catalog, not tenant isolation"}
              </span>
            </span>
            <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
          </button>
          <div className="border-t border-border px-1.5 py-1.5">
            <p className="px-1 pb-1 text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              Where the job runs
            </p>
            {analyst ? (
              <p className="px-1 text-xs text-muted-foreground">{environmentLabel(props.runtime)}</p>
            ) : (
            <Select value={props.runtime} onValueChange={(value) => props.onRuntimeChange(value as RuntimeMode)}>
              <SelectTrigger aria-label="Runtime" className="h-8 border-0 bg-transparent shadow-none">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {props.availableRuntimes.map((runtime) => (
                  <SelectItem key={runtime} value={runtime}>
                    {labelRuntime(runtime)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            )}
          </div>
        </div>
        <Dialog open={identityOpen} onOpenChange={setIdentityOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Session identity</DialogTitle>
              <DialogDescription>
                Assume an ACL principal for this catalog. These are demo identities, not a product login.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-3 text-sm">
              <p>
                You are <span className="font-medium">{props.identified ? props.principal : "unidentified"}</span>{" "}
                ({props.role}) on {environmentLabel(props.runtime)}.
              </p>
              <p className="text-muted-foreground">
                {props.identified
                  ? "Mine lists graphs you own. All also includes unowned graphs and, on Snowflake, graphs shared to you."
                  : props.runtime === "snowflake"
                    ? "Private graphs owned by someone else stay hidden. Unowned graphs remain visible. Sign in, then use Mine for graphs you own."
                    : props.runtime === "kubernetes"
                      ? "This cluster catalog lists every graph this control plane can see. Identity does not hide rows here. Sign in, then use Mine for graphs you own."
                      : "This laptop catalog lists every local graph. Identity does not hide rows here. Sign in, then use Mine for graphs you own."}
              </p>
              <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">Assume principal</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  size="sm"
                  onClick={() =>
                    assume.mutate({ userName: "ALICE", roles: ["APP_OPERATOR"], role: "operator" })
                  }
                >
                  Sign in as ALICE
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => assume.mutate({ userName: "CAROL", roles: ["ANALYST"], role: "analyst" })}
                >
                  Sign in as CAROL
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => assume.mutate({ userName: "SRE", roles: ["FLAKEGRAPH_STAFF"], role: "staff" })}
                >
                  Staff
                </Button>
                <Button size="sm" variant="ghost" onClick={() => assume.mutate({ userName: "", roles: [], role: "operator" })}>
                  Sign out
                </Button>
              </div>
              <div className="space-y-1.5">
                <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">Suggestions</p>
                <div className="flex flex-wrap gap-1.5">
                  {(["off", "on-request", "auto-fill"] as const).map((mode) => (
                    <Button
                      key={mode}
                      size="sm"
                      variant={props.suggestionMode === mode ? "default" : "outline"}
                      onClick={() => setSuggestions.mutate({ mode })}
                    >
                      {suggestionLabel(mode)}
                    </Button>
                  ))}
                </div>
              </div>
              <div className="rounded-md border border-border bg-muted/30 p-3 text-sm">
                <p>
                  <span className="font-medium">Forget</span> — catalog only. Files stay.
                </p>
                <p>
                  <span className="font-medium">Delete graph</span>
                  {props.runtime === "snowflake"
                    ? " — removes Snowflake storage."
                    : " — Snowflake-only. Forget removes the catalog row; it does not delete stored files."}
                </p>
              </div>
            </div>
          </DialogContent>
        </Dialog>
        {analyst ? null : (
          <Button
            className="w-full"
            variant={props.page === "new" ? "default" : "outline"}
            aria-current={props.page === "new" ? "page" : undefined}
            onClick={() => go(props.onNewGraph)}
          >
            <Plus /> New graph
          </Button>
        )}
        <nav className="space-y-0.5">
          {props.capabilities.has("cluster") && !analyst ? (
            <>
              <NavButton active={props.page === "fleet"} onClick={() => go(props.onFleet)}>
                <Server className="size-3.5" /> Fleet
              </NavButton>
              <NavButton active={props.page === "clusters"} onClick={() => go(props.onClusters)}>
                <Boxes className="size-3.5" /> Clusters
              </NavButton>
            </>
          ) : null}
          {props.role === "staff" || props.role === "operator" ? (
            <NavButton active={props.page === "keys"} onClick={() => go(props.onKeys)}>
              <KeyRound className="size-3.5" /> SDK keys
            </NavButton>
          ) : null}
          {props.role === "staff" ? (
            <NavButton active={props.page === "staff"} onClick={() => go(props.onStaff)}>
              <ShieldAlert className="size-3.5" /> Incident
            </NavButton>
          ) : null}
        </nav>
      </div>
      <Separator />
      <div className="space-y-2 px-4 py-3">
        <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
          {analyst ? "Perspectives" : "Graphs"}
        </p>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-2 size-3.5 text-muted-foreground" />
          <Input
            aria-label="Search graphs"
            className="h-8 pl-8 text-[13px]"
            placeholder="Search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
        {analyst ? null : (
          <div className="flex gap-1.5">
            <div className="grid min-w-0 flex-1 grid-cols-2 rounded-md border border-border p-0.5" role="group" aria-label="Catalog scope">
              <button
                type="button"
                aria-pressed={scope === "all"}
                data-selected={scope === "all"}
                className={cn(
                  "h-7 rounded-sm border-0 text-xs font-medium appearance-none",
                  scope === "all" ? "bg-foreground text-background" : "bg-transparent text-muted-foreground hover:bg-accent",
                )}
                onClick={() => setScope("all")}
              >
                All
              </button>
              <button
                type="button"
                aria-pressed={scope === "mine"}
                data-selected={scope === "mine"}
                className={cn(
                  "h-7 rounded-sm border-0 text-xs font-medium appearance-none",
                  scope === "mine" ? "bg-foreground text-background" : "bg-transparent text-muted-foreground hover:bg-accent",
                )}
                onClick={() => setScope("mine")}
              >
                Mine
              </button>
            </div>
            <Select value={storage} onValueChange={setStorage}>
              <SelectTrigger aria-label="Storage filter" className="h-8 w-[7.75rem] shrink-0 px-2 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All storage</SelectItem>
                <SelectItem value="local">Local files</SelectItem>
                <SelectItem value="snowflake">Snowflake</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-0.5 px-3 pb-8">
          {props.capabilities.has("forget") && !analyst && selected.length > 0 ? (
            <div className="mb-2 flex items-center justify-between gap-2 px-1">
              <p className="text-xs text-muted-foreground">{selected.length} selected</p>
              <Button
                size="sm"
                variant={confirmBulkForget ? "destructive" : "outline"}
                disabled={forgetMany.isPending}
                onClick={() => {
                  if (!confirmBulkForget) {
                    setConfirmBulkForget(true);
                    return;
                  }
                  forgetMany.mutate({ runIds: selected });
                  setConfirmBulkForget(false);
                }}
              >
                {confirmBulkForget ? "Confirm forget" : "Forget selected"}
              </Button>
            </div>
          ) : null}
          {runs.isLoading && !runs.data ? (
            <div className="space-y-2 px-1" aria-busy="true" aria-label="Loading graphs">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </div>
          ) : null}
          {runs.error ? (
            <p className="px-1 text-sm text-destructive" role="alert">
              {runs.error.message}
            </p>
          ) : null}
          {empty && !runs.error ? (
            <div className="px-1 text-sm text-muted-foreground" data-testid="catalog-empty">
              <p className="font-medium text-foreground">{empty.title}</p>
              <p>{empty.detail}</p>
              {scope === "mine" ? (
                <Button className="mt-2" size="sm" variant="outline" onClick={() => setScope("all")}>
                  Show all graphs
                </Button>
              ) : null}
            </div>
          ) : null}
          {filtered.map((run) => (
            <RunRow
              key={run.runId}
              run={run}
              selected={props.selectedRunId === run.runId && props.page === "run"}
              canForget={props.capabilities.has("forget") && !analyst}
              canDelete={props.capabilities.has("delete_graph")}
              checked={selected.includes(run.runId)}
              onToggle={() =>
                setSelected((current) =>
                  current.includes(run.runId) ? current.filter((id) => id !== run.runId) : [...current, run.runId],
                )
              }
              onSelect={() => go(() => props.onSelectRun(run.runId))}
              onForget={() => {
                if (pendingForget !== run.runId) {
                  setPendingForget(run.runId);
                  return;
                }
                forget.mutate({ runId: run.runId });
                setPendingForget(null);
              }}
              confirmingForget={pendingForget === run.runId}
            />
          ))}
        </div>
      </ScrollArea>
    </aside>
  );
}

function NavButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={cn(
        "flex h-8 w-full items-center gap-2 rounded-md px-2.5 text-[13px] font-medium transition-colors",
        active ? "bg-accent text-foreground" : "text-muted-foreground hover:bg-muted/80 hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

function RunRow({
  run,
  selected,
  canForget,
  canDelete,
  checked,
  confirmingForget,
  onToggle,
  onSelect,
  onForget,
}: {
  run: RunSnapshot;
  selected: boolean;
  canForget: boolean;
  canDelete: boolean;
  checked: boolean;
  confirmingForget: boolean;
  onToggle: () => void;
  onSelect: () => void;
  onForget: () => void;
}) {
  const name = run.graphName || run.graphId;
  return (
    <div
      className={cn(
        "group flex items-start gap-2 rounded-md border border-transparent border-l-4 px-2 py-1.5",
        selected ? "bg-accent" : "hover:bg-accent/60",
        statusAccent(run),
      )}
    >
      {canForget ? (
        <input type="checkbox" className="mt-1" aria-label={`Select ${run.graphName || run.graphId}`} checked={checked} onChange={onToggle} />
      ) : null}
      <button type="button" className="min-w-0 flex-1 text-left" onClick={onSelect}>
        <span className="block truncate text-[13px] font-medium leading-tight">{run.graphName || run.graphId}</span>
        <p className="mt-0.5 truncate text-xs text-muted-foreground">{statusSentence(run, { nodeCount: asCount(run.raw?.nodeCount) })}</p>
        <p className="truncate text-[11px] text-muted-foreground">
          {run.storageKind === "snowflake" ? "Snowflake" : "Local files"} · {formatRelativeTime(run.updatedAt)}
        </p>
      </button>
      {canForget && !isActiveStatus(run.status) ? (
        confirmingForget ? (
          <Button
            size="sm"
            variant="destructive"
            aria-label={`Confirm forget ${name}`}
            title={
              canDelete
                ? "Remove from this catalog. Stored files stay until Delete graph."
                : "Remove from this catalog. This does not delete stored files."
            }
            className="h-7 shrink-0 px-2 text-xs"
            onClick={onForget}
          >
            Confirm
          </Button>
        ) : (
          <Button
            size="icon"
            variant="ghost"
            aria-label={`Forget ${name}`}
            title={
              canDelete
                ? "Remove from this catalog. Stored files stay until Delete graph."
                : "Remove from this catalog. This does not delete stored files."
            }
            className="size-7"
            onClick={onForget}
          >
            <Trash2 className="size-3.5" />
          </Button>
        )
      ) : null}
    </div>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  if (isSuccessStatus(normalized)) {
    return <Badge variant="success">succeeded</Badge>;
  }
  if (normalized === "cancelling") {
    return <Badge variant="warning">cancelling</Badge>;
  }
  if (isActiveStatus(normalized)) {
    return <Badge variant="secondary">{normalized}</Badge>;
  }
  if (normalized === ARTIFACTS_UNAVAILABLE_STATUS) {
    return <Badge variant="warning">unavailable</Badge>;
  }
  if (normalized === "interrupted") {
    return <Badge variant="warning">interrupted</Badge>;
  }
  if (normalized === "cancelled") {
    return <Badge variant="outline">cancelled</Badge>;
  }
  return <Badge variant="danger">{normalized}</Badge>;
}

function labelRuntime(runtime: RuntimeMode): string {
  if (runtime === "kubernetes") {
    return "Kubernetes";
  }
  if (runtime === "snowflake") {
    return "Snowflake";
  }
  return "Local";
}

function environmentLabel(runtime: RuntimeMode): string {
  if (runtime === "kubernetes") {
    return "Kubernetes";
  }
  if (runtime === "snowflake") {
    return "Snowflake";
  }
  return "Laptop · local files";
}

function suggestionLabel(mode: "off" | "on-request" | "auto-fill"): string {
  if (mode === "on-request") {
    return "On request";
  }
  if (mode === "auto-fill") {
    return "Auto-fill";
  }
  return "Off";
}

function statusAccent(run: RunSnapshot): string {
  const normalized = run.status.toLowerCase();
  if (asCount(run.raw?.nodeCount) === 0 && isSuccessStatus(normalized)) {
    return "border-l-destructive";
  }
  if (isSuccessStatus(normalized)) {
    return "border-l-emerald-500";
  }
  if (normalized === "interrupted" || normalized === ARTIFACTS_UNAVAILABLE_STATUS || normalized === "cancelling") {
    return "border-l-amber-500";
  }
  if (isActiveStatus(normalized)) {
    return "border-l-sky-500";
  }
  if (normalized === "cancelled") {
    return "border-l-border";
  }
  return "border-l-destructive";
}

function asCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
