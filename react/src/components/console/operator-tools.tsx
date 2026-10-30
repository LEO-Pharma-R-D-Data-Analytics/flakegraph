"use client";

import { useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { PageHeader } from "@/components/console/page-header";

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
      toast.success("Incident note built");
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  // Bulk actions report once, for the whole batch, rather than per run.
  const cancel = trpc.runs.cancel.useMutation({ onError: () => undefined });
  const recover = trpc.runs.recover.useMutation({ onError: () => undefined });
  const [bulkRunning, setBulkRunning] = useState<"recover" | "cancel" | null>(null);

  async function runBulk(intent: "recover" | "cancel") {
    const ids = incident.data?.activeRunIds ?? [];
    setBulkIntent(null);
    setBulkRunning(intent);
    const results = await Promise.allSettled(
      ids.map((runId) => (intent === "recover" ? recover.mutateAsync({ runId }) : cancel.mutateAsync({ runId }))),
    );
    setBulkRunning(null);
    const failed = results.filter((result) => result.status === "rejected").length;
    const verb = intent === "recover" ? "Recovered" : "Requested cancellation of";
    if (failed === 0) {
      toast.success(`${verb} ${ids.length} run${ids.length === 1 ? "" : "s"}`);
    } else {
      toast.error(`${verb} ${ids.length - failed} of ${ids.length} runs; ${failed} failed`);
    }
    await incident.refetch();
  }
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
              pending={save.isPending}
              onClick={() =>
                save.mutate({
                  expectedPrincipal: expected,
                  actualPrincipal: actual,
                  graphOwner: owner,
                  grants,
                })
              }
            >
              {save.isPending ? "Building…" : "Build incident note"}
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
        <CardContent className="flex flex-wrap items-center gap-2">
          {(["recover", "cancel"] as const).map((intent) => {
            const armed = bulkIntent === intent;
            const running = bulkRunning === intent;
            const count = (incident.data?.activeRunIds ?? []).length;
            const label = intent === "recover" ? "Recover active workers" : "Cancel active runs";
            return (
              <Button
                key={intent}
                size="sm"
                variant={armed ? "destructive" : "outline"}
                pending={running}
                disabled={count === 0 || (bulkRunning !== null && !running)}
                onClick={() => (armed ? void runBulk(intent) : setBulkIntent(intent))}
              >
                {running
                  ? intent === "recover"
                    ? "Recovering…"
                    : "Cancelling…"
                  : armed
                    ? `Confirm: ${intent} ${count} run${count === 1 ? "" : "s"}`
                    : label}
              </Button>
            );
          })}
          {bulkIntent ? (
            <Button size="sm" variant="ghost" onClick={() => setBulkIntent(null)}>
              Keep them running
            </Button>
          ) : null}
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
    onError: (error) => toast.error(error.message),
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
              <Button
                size="sm"
                variant="outline"
                onClick={() =>
                  void navigator.clipboard
                    .writeText(grant.sql)
                    .then(() => toast.success("SQL copied"), () => toast.error("Could not copy to the clipboard"))
                }
              >
                Copy SQL
              </Button>
              <Button
                size="sm"
                variant="secondary"
                pending={mark.isPending && mark.variables?.object === grant.object}
                disabled={mark.isPending}
                onClick={() => mark.mutate({ object: grant.object, ok: !grant.ok })}
              >
                {grant.ok ? "Mark missing" : "Mark granted"}
              </Button>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
