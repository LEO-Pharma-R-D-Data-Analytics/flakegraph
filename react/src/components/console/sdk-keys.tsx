"use client";

import { useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Alert } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PageHeader } from "@/components/console/page-header";
import {
  CONTROL_PLANE_API,
  CONTROL_PLANE_PROCEDURES,
  askGraphExample,
  askStreamExample,
  getRunExample,
  listRunsExample,
  pythonClientExample,
} from "@/lib/control-plane-api";

export function SdkKeysPage() {
  const workspace = trpc.workspace.get.useQuery();
  const [name, setName] = useState("ci-eval");
  const [secret, setSecret] = useState<string | null>(null);
  const [revokeId, setRevokeId] = useState<string | null>(null);
  const create = trpc.workspace.createKey.useMutation({
    onSuccess: async (result) => {
      setSecret(result.secret);
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const revoke = trpc.workspace.revokeKey.useMutation({
    onSuccess: async () => {
      toast.success("Key revoked. Existing jobs using it will get 401.");
      setSecret(null);
      setRevokeId(null);
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const keys = workspace.data?.apiKeys ?? [];
  const origin = typeof window === "undefined" ? "http://127.0.0.1:3000" : window.location.origin;
  const listExample = listRunsExample(origin);
  const docsUrl = `${origin}${CONTROL_PLANE_API.docsPath}`;

  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Access"
        title="SDK keys"
        description="Machine credentials for CI, eval jobs, and scripts. There is no separate SDK package to install: programs call this control plane over HTTP. This page is the human guide. GET /api/docs is the machine-readable catalog. The browser session keeps using SSO."
      />

      <Card id="api-docs" data-testid="sdk-docs">
        <CardHeader>
          <CardTitle>Where the API docs are</CardTitle>
          <CardDescription>
            Keys authenticate callers. The API they call is this control plane, not a third-party SDK site.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <ol className="list-decimal space-y-2 pl-5 text-muted-foreground">
            <li>
              <span className="text-foreground">This page.</span> Worked examples and the procedure list below are the operator
              reference.
            </li>
            <li>
              <span className="text-foreground">JSON catalog.</span>{" "}
              <a className="font-mono text-xs text-foreground underline underline-offset-4" href={CONTROL_PLANE_API.docsPath}>
                GET {CONTROL_PLANE_API.docsPath}
              </a>{" "}
              returns endpoint, auth headers, and every supported procedure. Bookmark {docsUrl}.
            </li>
            <li>
              <span className="text-foreground">Checkout.</span> Procedure names live in{" "}
              <span className="font-mono text-xs text-foreground">{CONTROL_PLANE_API.checkoutProcedures}</span>. Request and
              graph payloads live in{" "}
              <span className="font-mono text-xs text-foreground">{CONTROL_PLANE_API.checkoutPayloads}</span>.
            </li>
            <li>
              <span className="text-foreground">Wire format.</span>{" "}
              <a
                className="underline underline-offset-4"
                href={CONTROL_PLANE_API.trpcHttpDocs}
                target="_blank"
                rel="noreferrer"
              >
                tRPC HTTP
              </a>{" "}
              at <span className="font-mono text-xs text-foreground">{`${origin}${CONTROL_PLANE_API.endpoint}/{procedure}`}</span>.
              Queries are GET with <span className="font-mono text-xs text-foreground">?input=</span>. Mutations are POST.
            </li>
            <li>
              <span className="text-foreground">Fleet SSO.</span> Program paths bypass the browser sign-in gate. See{" "}
              <span className="font-mono text-xs text-foreground">{CONTROL_PLANE_API.fleetGuide}</span>.
            </li>
          </ol>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" asChild>
              <a href={CONTROL_PLANE_API.docsPath}>Open JSON catalog</a>
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                void navigator.clipboard.writeText(docsUrl);
                toast.success("Docs URL copied");
              }}
            >
              Copy docs URL
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(20rem,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Keys on this control plane</CardTitle>
            <CardDescription>The secret is shown once. Store it in a secret manager; we only keep a preview after that.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
              <label className="grid min-w-0 flex-1 gap-1.5 text-sm">
                <span className="font-medium">Key name</span>
                <Input
                  aria-label="Key name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="ci-eval"
                />
              </label>
              <Button className="shrink-0" disabled={create.isPending} onClick={() => create.mutate({ name: name.trim() || "ci-eval" })}>
                Create a key
              </Button>
            </div>
            {secret ? (
              <Alert className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <span className="break-all font-mono text-xs">Copy now: {secret}</span>
                <Button
                  className="shrink-0"
                  size="sm"
                  variant="secondary"
                  onClick={() => {
                    void navigator.clipboard.writeText(secret);
                    toast.success("Secret copied");
                  }}
                >
                  Copy secret
                </Button>
              </Alert>
            ) : null}
            {keys.length === 0 ? (
              <p className="text-sm text-muted-foreground">No machine keys yet. Create one for CI or a script.</p>
            ) : (
              <ul className="divide-y divide-border rounded-md border border-border">
                {keys.map((key) => (
                  <li key={key.id} className="flex flex-wrap items-center justify-between gap-3 px-3 py-3">
                    <div className="min-w-0">
                      <p className="font-medium">{key.name}</p>
                      <p className="truncate font-mono text-xs text-muted-foreground">
                        {key.preview} · {key.createdAt.slice(0, 10)}
                      </p>
                    </div>
                    <Button
                      size="sm"
                      className="shrink-0"
                      variant={revokeId === key.id ? "destructive" : "ghost"}
                      onClick={() => {
                        if (revokeId !== key.id) {
                          setRevokeId(key.id);
                          return;
                        }
                        revoke.mutate({ id: key.id });
                      }}
                    >
                      {revokeId === key.id ? "Confirm revoke" : "Revoke"}
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>How to use a key</CardTitle>
            <CardDescription>Browsers send the session cookie. Machines send this secret instead.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p>
              Create a named key, copy the <span className="font-mono text-xs">fg_…</span> secret immediately, then export it as{" "}
              <span className="font-mono text-xs">FLAKEGRAPH_API_KEY</span> in the job that calls the control plane.
            </p>
            <ul className="list-disc space-y-1 pl-5 text-muted-foreground">
              <li>CI evals and scripts authenticate as the key, not as ALICE in a browser.</li>
              <li>
                Send <span className="font-mono text-xs text-foreground">Authorization: Bearer $FLAKEGRAPH_API_KEY</span> or{" "}
                <span className="font-mono text-xs text-foreground">x-flakegraph-api-key</span>. Browser SSO is ignored when a
                key is present.
              </li>
              <li>
                Add <span className="font-mono text-xs text-foreground">x-flakegraph-runtime: local</span> (or kubernetes /
                snowflake) so the call hits the same catalog as the sidebar.
              </li>
              <li>A 401 on this path means the key is missing or revoked. Do not bounce the operator through SSO.</li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <Card id="api-examples">
        <CardHeader>
          <CardTitle>What you can do with a key</CardTitle>
          <CardDescription>
            Typical jobs: list graphs in CI, poll a run until it finishes, ask a completed graph, or pull inspect JSON for an
            eval. Replace <span className="font-mono text-xs">fg_…</span> with the secret shown once above.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="list">
            <TabsList className="h-auto min-h-9 w-auto max-w-full flex-wrap justify-start">
              <TabsTrigger value="list">List graphs</TabsTrigger>
              <TabsTrigger value="status">Check a run</TabsTrigger>
              <TabsTrigger value="ask">Ask a graph</TabsTrigger>
              <TabsTrigger value="stream">Stream ask</TabsTrigger>
              <TabsTrigger value="python">Python</TabsTrigger>
            </TabsList>
            <TabsContent value="list">
              <ExampleBlock
                why="CI and dashboards call this to see which graphs exist on the catalog and whether the latest job succeeded."
                code={listExample}
                testId="sdk-key-example"
              />
            </TabsContent>
            <TabsContent value="status">
              <ExampleBlock
                why="Poll runs.get until status is succeeded, failed, or cancelled. runs.documents is the per-file progress the workspace uses."
                code={getRunExample(origin)}
                testId="sdk-status-example"
              />
            </TabsContent>
            <TabsContent value="ask">
              <ExampleBlock
                why="Eval jobs that only need the final JSON answer can still call graphs.ask. Prefer POST /api/ask when you want tokens and tool status."
                code={askGraphExample(origin)}
                testId="sdk-ask-example"
              />
            </TabsContent>
            <TabsContent value="stream">
              <ExampleBlock
                why="Agents, eval harnesses, and other TypeScript consumers should stream POST /api/ask. format=ndjson emits status, tool, text, citation, and done events."
                code={askStreamExample(origin)}
                testId="sdk-ask-stream-example"
              />
            </TabsContent>
            <TabsContent value="python">
              <ExampleBlock
                why="Same HTTP API from Python’s standard library. No FlakeGraph pip package is required."
                code={pythonClientExample(origin)}
                testId="sdk-python-example"
              />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <Card id="api-reference">
        <CardHeader>
          <CardTitle>Procedure catalog</CardTitle>
          <CardDescription>
            These are the procedures scripts usually call. The full router, including compose and staff tools, is{" "}
            {CONTROL_PLANE_API.checkoutProcedures}.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table data-testid="sdk-procedure-catalog">
            <TableHeader>
              <TableRow>
                <TableHead>Procedure</TableHead>
                <TableHead>Kind</TableHead>
                <TableHead>Use it to</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {CONTROL_PLANE_PROCEDURES.map((procedure) => (
                <TableRow key={procedure.name}>
                  <TableCell className="font-mono text-xs">{procedure.name}</TableCell>
                  <TableCell>
                    <Badge variant={procedure.kind === "query" ? "secondary" : "outline"}>{procedure.kind}</Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground">{procedure.purpose}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

function ExampleBlock({
  why,
  code,
  testId,
}: {
  why: string;
  code: string;
  testId: string;
}) {
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">{why}</p>
      <pre className="overflow-x-auto rounded-md border border-border bg-muted/40 p-3 font-mono text-xs" data-testid={testId}>
        {code}
      </pre>
      <Button
        size="sm"
        variant="outline"
        onClick={() => {
          void navigator.clipboard.writeText(code);
          toast.success("Example copied");
        }}
      >
        Copy example
      </Button>
    </div>
  );
}
