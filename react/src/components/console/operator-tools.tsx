"use client";

import { useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type { RuntimeMode } from "@/server/protocol/schema";
import { PageHeader } from "@/components/console/page-header";

export function PromotionCard({
  fromRuntime,
  graphId,
  graphName,
  onApplied,
}: {
  fromRuntime: RuntimeMode;
  graphId: string;
  graphName: string;
  onApplied?: (runtime: RuntimeMode) => Promise<void> | void;
}) {
  const [target, setTarget] = useState<RuntimeMode>(fromRuntime === "kubernetes" ? "snowflake" : "kubernetes");
  const [keepId, setKeepId] = useState(true);
  const plan = trpc.graphs.promote.useQuery({ fromRuntime, toRuntime: target, keepGraphId: keepId });
  const apply = trpc.graphs.applyPromotion.useMutation({
    onSuccess: async () => {
      toast.success(`Remaps staged for ${target}. The compose form keeps this graph.`);
      await onApplied?.(target);
    },
    onError: (error) => toast.error(error.message),
  });
  return (
    <Card>
      <CardHeader>
        <CardTitle>Promote this graph</CardTitle>
        <CardDescription>
          {fromRuntime === "local"
            ? "This laptop run worked. Remap sources before Kubernetes or Snowflake. The form is not wiped."
            : fromRuntime === "kubernetes"
              ? "This Kubernetes run worked. Remap sources before Snowflake or another cluster. The form is not wiped."
              : "This Snowflake run worked. Remap sources if you send the job to another runtime. The form is not wiped."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        <div className="flex flex-wrap gap-2">
          {(["kubernetes", "snowflake"] as const).map((runtime) => (
            <Button key={runtime} size="sm" variant={target === runtime ? "default" : "outline"} onClick={() => setTarget(runtime)}>
              Run again on {runtime === "kubernetes" ? "Kubernetes" : "Snowflake"}
            </Button>
          ))}
          <Button
            size="sm"
            variant={keepId ? "secondary" : "ghost"}
            aria-pressed={keepId}
            onClick={() => setKeepId((value) => !value)}
          >
            {keepId ? "Keep graph ID" : "Mint a sibling"}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          {keepId
            ? "Compose will reuse this graph ID. Confirm the remaps, then Start."
            : "Compose will mint a new graph ID. The original graph stays on this catalog."}
        </p>
        <ul className="list-disc pl-5" data-testid="promotion-remaps">
          {(plan.data?.remaps ?? []).map((item) => (
            <li key={`${item.from}-${item.to}`}>
              {item.from} → {item.to}
            </li>
          ))}
        </ul>
        <Button
          size="sm"
          onClick={() =>
            apply.mutate({
              fromRuntime,
              toRuntime: target,
              keepGraphId: keepId,
              graphId,
              graphName,
            })
          }
        >
          Apply remaps
        </Button>
      </CardContent>
    </Card>
  );
}

export function StaffIncidentPage() {
  const incident = trpc.workspace.incident.useQuery();
  const workspace = trpc.workspace.get.useQuery();
  const [expected, setExpected] = useState("ALICE");
  const [actual, setActual] = useState("a1b2-uuid");
  const [owner, setOwner] = useState("BOB");
  const [grants, setGrants] = useState("ROLE ANALYST");
  const [bulkIntent, setBulkIntent] = useState<"recover" | "cancel" | null>(null);
  const save = trpc.workspace.identityIncident.useMutation({
    onSuccess: async () => {
      await workspace.refetch();
    },
  });
  const cancel = trpc.runs.cancel.useMutation({
    onSuccess: async () => {
      toast.success("Cancel requested");
      await incident.refetch();
    },
  });
  const recover = trpc.runs.recover.useMutation({
    onSuccess: async (message) => {
      toast.success(message);
      await incident.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  return (
    <div className="space-y-4">
      <PageHeader
        kicker="Staff"
        title="Staff incident"
        description={`Queue depth ${incident.data?.queueDepth ?? 0} · model server ${incident.data?.modelReady ? "ready" : "down"}`}
      />
      <Card>
        <CardHeader>
          <CardTitle>Identity mismatch template</CardTitle>
          <CardDescription>Expected principal, actual principal, graph owner, grants. Sharing bugs are identity bugs.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-2">
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">Expected principal</span>
            <Input aria-label="Expected principal" value={expected} onChange={(event) => setExpected(event.target.value)} />
          </label>
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">Actual principal</span>
            <Input aria-label="Actual principal" value={actual} onChange={(event) => setActual(event.target.value)} />
          </label>
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">Graph owner</span>
            <Input aria-label="Graph owner" value={owner} onChange={(event) => setOwner(event.target.value)} />
          </label>
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">Grants</span>
            <Input aria-label="Grants" value={grants} onChange={(event) => setGrants(event.target.value)} />
          </label>
          <div className="md:col-span-2">
            <Button
              onClick={() =>
                save.mutate({
                  expectedPrincipal: expected,
                  actualPrincipal: actual,
                  graphOwner: owner,
                  grants,
                })
              }
            >
              Build incident note
            </Button>
          </div>
          {workspace.data?.identityIncident?.template ? (
            <Textarea readOnly className="md:col-span-2 min-h-32 font-mono text-xs" value={workspace.data.identityIncident.template} />
          ) : null}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Bulk recover / cancel</CardTitle>
          <CardDescription>Confirmed writes only. Unknown callers stay rejected.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Button
            size="sm"
            variant="outline"
            disabled={(incident.data?.activeRunIds ?? []).length === 0}
            onClick={() => {
              const ids = incident.data?.activeRunIds ?? [];
              if (bulkIntent !== "recover") {
                setBulkIntent("recover");
                return;
              }
              for (const runId of ids) {
                recover.mutate({ runId });
              }
              setBulkIntent(null);
            }}
          >
            {bulkIntent === "recover" ? "Confirm recover" : "Recover active workers"}
          </Button>
          <Button
            size="sm"
            variant="outline"
            disabled={(incident.data?.activeRunIds ?? []).length === 0}
            onClick={() => {
              const ids = incident.data?.activeRunIds ?? [];
              if (bulkIntent !== "cancel") {
                setBulkIntent("cancel");
                return;
              }
              for (const runId of ids) {
                cancel.mutate({ runId });
              }
              setBulkIntent(null);
            }}
          >
            {bulkIntent === "cancel" ? "Confirm cancel" : "Cancel active jobs"}
          </Button>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Last 20 catalog events</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          {(incident.data?.events ?? []).length === 0 ? (
            <p className="text-muted-foreground">No catalog events yet.</p>
          ) : null}
          {(incident.data?.events ?? []).map((item) => (
            <p key={item.runId}>
              <span className="font-medium">{item.graphName}</span> · {item.status}
              {item.error ? ` · ${item.error}` : ""}
            </p>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

export function SnowflakeGrantsCard() {
  const session = trpc.auth.session.useQuery();
  const mark = trpc.workspace.markGrant.useMutation({
    onSuccess: async () => {
      await session.refetch();
    },
  });
  const grants = session.data?.grants ?? [];
  const blocked = grants.some((grant) => !grant.ok);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Snowflake account preflight</CardTitle>
        <CardDescription>Start stays blocked until warehouse, stage, and Cortex grants are green.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {blocked ? (
          <Alert variant="destructive">Account preflight is red. Copy SQL, apply the grant, then Mark granted.</Alert>
        ) : (
          <Alert>Account preflight is green.</Alert>
        )}
        {grants.map((grant) => (
          <div key={grant.object} className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <span>
              {grant.object} · {grant.ok ? "ok" : "missing"}
            </span>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" onClick={() => void navigator.clipboard.writeText(grant.sql)}>
                Copy SQL
              </Button>
              <Button size="sm" variant="secondary" onClick={() => mark.mutate({ object: grant.object, ok: !grant.ok })}>
                {grant.ok ? "Mark missing" : "Mark granted"}
              </Button>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
