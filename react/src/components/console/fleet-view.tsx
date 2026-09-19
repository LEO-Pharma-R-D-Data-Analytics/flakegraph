"use client";

import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { PageHeader } from "@/components/console/page-header";
import { Skeleton } from "@/components/ui/skeleton";

export function FleetView() {
  const cluster = trpc.fleet.cluster.useQuery(undefined, { refetchInterval: 15_000 });
  const session = trpc.auth.session.useQuery();
  const selected = cluster.data;
  const grafanaUrl = session.data?.grafanaUrl ?? null;

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Infrastructure"
        title="Compute fleet"
        description="Registered Kubernetes nodes, colocated model servers, and FlakeGraph workers."
        actions={
          <>
            {grafanaUrl ? (
              <Button size="sm" variant="outline" asChild>
                <a href={grafanaUrl} target="_blank" rel="noreferrer">
                  Open Grafana
                </a>
              </Button>
            ) : null}
            <Button size="sm" variant="outline" onClick={() => void cluster.refetch()}>
              Refresh
            </Button>
          </>
        }
      />
      {cluster.isLoading && !selected ? (
        <div className="grid gap-3 md:grid-cols-2">
          <Skeleton className="h-36 w-full" />
          <Skeleton className="h-36 w-full" />
        </div>
      ) : cluster.error ? (
        <Alert variant="destructive">{cluster.error.message}</Alert>
      ) : !selected ? (
        <Alert>Connect kubectl to a Kubernetes cluster to view its registered fleet.</Alert>
      ) : (
        <>
          <p className="text-sm text-muted-foreground" data-testid="fleet-status-sentence">
            Context: {selected.context} · Namespace: {selected.namespace} · {selected.nodesReady}/{selected.nodesTotal} nodes ready
            {selected.nodesReady < selected.nodesTotal
              ? " · Pending · waiting for node capacity"
              : ""}
          </p>
          {selected.warnings.map((warning) => (
            <Alert key={warning}>{warning}</Alert>
          ))}
          <div className="grid gap-3 sm:grid-cols-[repeat(auto-fit,minmax(16rem,1fr))]">
            {selected.nodes.map((node) => {
              const onNode = selected.workloads.filter((workload) => workload.node === node.name);
              const workerCount = onNode.filter((workload) => workload.component.startsWith("worker")).length;
              return (
                <Card key={node.name}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <span>{node.name}</span>
                      <Badge variant={node.ready ? "success" : "danger"}>{node.ready ? "ready" : "not ready"}</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-1 text-sm text-muted-foreground">
                    <p>Class {node.nodeClass}</p>
                    <p>
                      GPU {node.gpuCount}
                      {node.gpuModel ? ` · ${node.gpuModel}` : ""}
                    </p>
                    <p>
                      {onNode.length === 0
                        ? "Idle · no workloads on this node"
                        : `Workers ${workerCount} · ${onNode.length} workload${onNode.length === 1 ? "" : "s"} on this node`}
                    </p>
                    <p>Model {node.model || "none"}</p>
                    <NodeAssignments nodeName={node.name} namespace={selected.namespace} />
                  </CardContent>
                </Card>
              );
            })}
          </div>
          <Card>
            <CardHeader>
              <CardTitle>Workloads ({selected.workloads.length})</CardTitle>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Component</TableHead>
                    <TableHead>Phase</TableHead>
                    <TableHead>Node</TableHead>
                    <TableHead>Pending reason</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {selected.workloads.map((workload) => (
                    <TableRow key={workload.name}>
                      <TableCell>{workload.name}</TableCell>
                      <TableCell>{workload.component}</TableCell>
                      <TableCell>{workload.phase}</TableCell>
                      <TableCell>{workload.node}</TableCell>
                      <TableCell>
                        {workload.ready
                          ? "—"
                          : pendingReason(workload.phase, selected.nodesReady, selected.nodesTotal)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function pendingReason(phase: string, nodesReady: number, nodesTotal: number): string {
  const normalized = phase.toLowerCase();
  if (normalized === "pending" || normalized === "queued") {
    if (nodesReady < nodesTotal) {
      return `Pending · waiting for node capacity (${nodesReady}/${nodesTotal} ready)`;
    }
    return "Pending · waiting for GPU quota or image pull";
  }
  if (!normalized) {
    return "not ready";
  }
  return `${phase} · not ready`;
}

function NodeAssignments({ nodeName, namespace }: { nodeName: string; namespace: string }) {
  const assignments = trpc.fleet.nodeAssignments.useQuery({ nodeName, namespace });
  if (!assignments.data?.length) {
    return null;
  }
  return (
    <ul className="pt-2 text-xs">
      {assignments.data.map((item) => (
        <li key={item.taskId}>
          {item.stage} {item.scopeId} ({item.runId})
        </li>
      ))}
    </ul>
  );
}
