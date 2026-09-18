"use client";

import { useForm } from "react-hook-form";
import { useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { ClusterProfile } from "@/server/protocol/schema";
import { PageHeader } from "@/components/console/page-header";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

export function ClusterCatalog() {
  const clusters = trpc.clusters.list.useQuery();
  const selected = trpc.clusters.selected.useQuery();
  const utils = trpc.useUtils();
  const [pendingRemove, setPendingRemove] = useState<string | null>(null);
  const form = useForm<ClusterProfile>({
    defaultValues: {
      name: "lab",
      namespace: "flakegraph",
      context: "",
      kubeconfig: "",
      description: "",
    },
  });
  const upsert = trpc.clusters.upsert.useMutation({
    onSuccess: async () => {
      toast.success("Cluster saved");
      await clusters.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const remove = trpc.clusters.delete.useMutation({
    onSuccess: async () => {
      toast.success("Cluster removed");
      await Promise.all([clusters.refetch(), utils.clusters.selected.invalidate()]);
    },
    onError: (error) => toast.error(error.message),
  });
  const select = trpc.clusters.select.useMutation({
    onSuccess: async () => {
      toast.success("Cluster selected");
      await selected.refetch();
    },
    onError: (error) => toast.error(error.message),
  });

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Kubernetes"
        title="Registered clusters"
        description="The kubeconfig itself stays on disk. The catalog only stores names, contexts, and namespaces."
      />
      <Card>
        <CardHeader>
          <CardTitle>Add cluster</CardTitle>
        </CardHeader>
        <CardContent>
          <form
            className="grid gap-3 md:grid-cols-2"
            onSubmit={form.handleSubmit((values) => upsert.mutate(values))}
          >
            <label className="grid gap-1.5 text-sm">
              <span className="font-medium">Cluster name</span>
              <Input aria-label="Cluster name" {...form.register("name", { required: true })} />
            </label>
            <label className="grid gap-1.5 text-sm">
              <span className="font-medium">Namespace</span>
              <Input aria-label="Cluster namespace" {...form.register("namespace", { required: true })} />
            </label>
            <label className="grid gap-1.5 text-sm">
              <span className="font-medium">Context</span>
              <Input aria-label="Cluster context" {...form.register("context")} />
            </label>
            <label className="grid gap-1.5 text-sm">
              <span className="font-medium">Kubeconfig path</span>
              <Input aria-label="Cluster kubeconfig" {...form.register("kubeconfig")} />
            </label>
            <label className="grid gap-1.5 text-sm md:col-span-2">
              <span className="font-medium">Description</span>
              <Input aria-label="Cluster description" {...form.register("description")} />
            </label>
            <div className="md:col-span-2">
              <Button type="submit" disabled={upsert.isPending}>
                Save cluster
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Registered</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          {clusters.isLoading && !clusters.data ? (
            <Skeleton className="h-24 w-full" />
          ) : (clusters.data ?? []).length === 0 ? (
            <p className="text-sm text-muted-foreground">No clusters yet. Save one above to select it for fleet jobs.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Namespace</TableHead>
                  <TableHead>Context</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(clusters.data ?? []).map((cluster) => (
                  <TableRow key={cluster.name}>
                    <TableCell>
                      <span className="inline-flex items-center gap-2">
                        {cluster.name}
                        {selected.data?.name === cluster.name ? <Badge variant="secondary">selected</Badge> : null}
                      </span>
                    </TableCell>
                    <TableCell>{cluster.namespace}</TableCell>
                    <TableCell>{cluster.context || "ambient"}</TableCell>
                    <TableCell className="space-x-2 text-right">
                      <Button size="sm" variant="outline" onClick={() => select.mutate({ name: cluster.name })}>
                        Select
                      </Button>
                      <Button
                        size="sm"
                        variant={pendingRemove === cluster.name ? "destructive" : "ghost"}
                        onClick={() => {
                          if (pendingRemove !== cluster.name) {
                            setPendingRemove(cluster.name);
                            return;
                          }
                          remove.mutate({ name: cluster.name });
                          setPendingRemove(null);
                        }}
                      >
                        {pendingRemove === cluster.name ? "Confirm remove" : "Remove"}
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
