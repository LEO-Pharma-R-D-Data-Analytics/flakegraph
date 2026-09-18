"use client";

import { useState } from "react";
import { toast } from "sonner";
import { trpc } from "@/components/providers";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { Alert } from "@/components/ui/alert";
import { PageHeader } from "@/components/console/page-header";
import { useStorageItem, writeStorage } from "@/lib/browser-storage";
import { AskPanel } from "@/components/console/ask-panel";

export { AskPanel };

export function VersionsPanel({ graphId, canPublish = true }: { graphId: string; canPublish?: boolean }) {
  const workspace = trpc.workspace.get.useQuery();
  const publish = trpc.workspace.publish.useMutation({
    onSuccess: async () => {
      toast.success("Published a new graph version. The previous production version stays readable.");
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const versions = (workspace.data?.versions ?? []).filter((item) => item.graphId === graphId);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Versions</CardTitle>
        <CardDescription>Atomic head move. Older versions stay readable.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {canPublish ? (
          <Button size="sm" onClick={() => publish.mutate({ graphId, note: "Published from Versions" })}>
            Publish new version
          </Button>
        ) : (
          <p className="text-sm text-muted-foreground">Publishing a version is a builder action.</p>
        )}
        {versions.length === 0 ? (
          <p className="text-sm text-muted-foreground">No published versions yet.</p>
        ) : (
          <ul className="space-y-2 text-sm">
            {versions.map((item) => (
              <li key={item.id}>
                <span className="font-medium">{item.label}</span> · {item.lifecycle} · {item.note}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

export function ReviewPanel({
  graphId,
  runId,
  canExport = true,
}: {
  graphId: string;
  runId: string;
  canExport?: boolean;
}) {
  const workspace = trpc.workspace.get.useQuery();
  const [ceiling, setCeiling] = useState(1);
  const decide = trpc.workspace.review.useMutation({
    onSuccess: async () => {
      await workspace.refetch();
    },
    onError: (error) => toast.error(error.message),
  });
  const sample = trpc.graphs.sampleReviews.useMutation({
    onSuccess: async () => {
      toast.success("Sampled 1% of high-confidence edges.");
      await workspace.refetch();
    },
  });
  const items = (workspace.data?.reviews ?? []).filter((item) => item.graphId === graphId);
  const pins = (workspace.data?.pins ?? []).filter((item) => item.graphId === graphId);
  const visible = items.filter((item) => (item.confidence ?? 0) <= ceiling);
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Keep, Drop, Merge, and Relabel record a human decision on this queue. They never rewrite the stored graph.
        {canExport
          ? " Export downloads the current graph JSON; it does not apply these decisions."
          : " Snowflake has no export on this page. These decisions stay on this queue only."}
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <Button size="sm" variant="outline" onClick={() => sample.mutate({ runId })}>
          Sample 1% high-confidence
        </Button>
        <label className="grid min-w-56 gap-1 text-sm">
          <span>Queue ceiling ({ceiling.toFixed(2)})</span>
          <Slider aria-label="Review confidence ceiling" value={[ceiling]} max={1} step={0.05} onValueChange={(value) => setCeiling(value[0] ?? 1)} />
        </label>
      </div>
      {pins.length ? (
        <p className="text-sm text-muted-foreground">{pins.length} looks-wrong pins. Pins are not deletes.</p>
      ) : null}
      {visible.length === 0 ? (
        <p className="text-sm text-muted-foreground">No uncertain triples in this confidence window.</p>
      ) : null}
      {visible.map((item) => (
        <Card key={item.id}>
          <CardHeader>
            <CardTitle className="text-base">
              {item.triple ? (
                <>
                  {item.triple.source} <span className="font-normal text-muted-foreground">—{item.triple.relation}→</span>{" "}
                  {item.triple.target}
                </>
              ) : (
                item.edgeId
              )}
            </CardTitle>
            <CardDescription>
              {item.quote || "Low-confidence triple"} · confidence {(item.confidence ?? 0).toFixed(2)}
              {item.triple ? ` · ${item.edgeId}` : ""}
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {(["keep", "drop", "merge", "relabel"] as const).map((decision) => (
              <Button
                key={decision}
                size="sm"
                variant={item.decision === decision ? "default" : "outline"}
                onClick={() => decide.mutate({ id: item.id, decision })}
              >
                {decision[0]!.toUpperCase() + decision.slice(1)}
              </Button>
            ))}
            {item.decision ? (
              <p className="w-full text-xs text-muted-foreground">
                Recorded {item.decision}. The live graph is unchanged.
              </p>
            ) : null}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function WatchPanel({
  graphId,
  sourcePath = "",
  onProcessNow,
}: {
  graphId: string;
  sourcePath?: string;
  onProcessNow?: () => void;
}) {
  const workspace = trpc.workspace.get.useQuery();
  const [prefix, setPrefix] = useState(sourcePath);
  // The field follows the graph's own source until it is edited.
  const [prefixFrom, setPrefixFrom] = useState(sourcePath);
  if (prefixFrom !== sourcePath) {
    setPrefixFrom(sourcePath);
    setPrefix(sourcePath);
  }
  const toggle = trpc.workspace.toggleWatch.useMutation({ onSuccess: () => void workspace.refetch() });
  const apply = trpc.workspace.applyWatch.useMutation({
    onSuccess: async (state) => {
      const latest = [...state.versions].reverse().find((item) => item.graphId === graphId);
      toast.success(
        latest
          ? `Recorded ${latest.label} from new checksums. Workers are not started until you submit.`
          : "Recorded an incremental version. Workers are not started until you submit.",
      );
      await workspace.refetch();
    },
  });
  const create = trpc.workspace.createWatch.useMutation({
    onSuccess: async () => {
      toast.success("Watching prefix for new checksums.");
      await workspace.refetch();
    },
  });
  const watches = (workspace.data?.watches ?? []).filter((item) => item.graphId === graphId);
  return (
    <div className="space-y-3">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Watch new files</CardTitle>
          <CardDescription>
            Schedule is a graph property. Pause anytime. Ingest records a new version; it does not start workers until you
            process the files.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Input
            aria-label="Watch prefix"
            value={prefix}
            onChange={(event) => setPrefix(event.target.value)}
            placeholder="Path or stage prefix workers can LIST"
          />
          <Button size="sm" disabled={!prefix.trim()} onClick={() => create.mutate({ graphId, prefix: prefix.trim() })}>
            Watch this prefix
          </Button>
        </CardContent>
      </Card>
      {watches.map((watch) => (
        <Card key={watch.id}>
          <CardHeader>
            <CardTitle className="text-base">{watch.prefix}</CardTitle>
            <CardDescription>
              Last window {watch.lastWindow}. {watch.pendingFiles} new files. Gold drift {watch.goldDrift}.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" onClick={() => toggle.mutate({ id: watch.id })}>
              {watch.paused ? "Resume watch" : "Pause watch"}
            </Button>
            <Button size="sm" disabled={watch.paused || watch.pendingFiles === 0} onClick={() => apply.mutate({ id: watch.id })}>
              Ingest new files
            </Button>
            {onProcessNow ? (
              <Button size="sm" variant="outline" onClick={onProcessNow}>
                Process on compose
              </Button>
            ) : null}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function OntologyPanel({ onApply }: { onApply: (types: string[]) => void }) {
  const [intent, setIntent] = useState("people, schools, and techniques in these histories");
  const propose = trpc.ingestion.ontology.useMutation({
    onError: (error) => toast.error(error.message),
  });
  const proposal = propose.data;
  const describe = (name: string) => proposal?.descriptions[name];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Describe the graph you want</CardTitle>
        <CardDescription>The assistant proposes types. You confirm. Submit stays a human click.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <Textarea aria-label="Graph intent" value={intent} onChange={(event) => setIntent(event.target.value)} rows={3} />
        <Button onClick={() => propose.mutate({ intent })} disabled={intent.trim().length < 8 || propose.isPending}>
          {propose.isPending ? "Proposing…" : "Propose ontology"}
        </Button>
        {proposal ? (
          <div className="space-y-2 text-sm" data-testid="ontology-proposal">
            <p>{proposal.warning}</p>
            {proposal.source === "heuristic" ? (
              <p className="text-muted-foreground">
                No model is configured for the console, so these are the nouns of your description.
              </p>
            ) : null}
            <TermList label="Types" names={proposal.types} describe={describe} />
            <TermList label="Relations" names={proposal.relations} describe={describe} />
            {proposal.coverage ? (
              <p data-testid="ontology-coverage">
                Gold coverage {proposal.coverage.covered.length}/{proposal.coverage.goldTypes.length}. Missing{" "}
                {proposal.coverage.missing.join(", ") || "none"}.
              </p>
            ) : null}
            <Button size="sm" variant="secondary" onClick={() => onApply(proposal.types)}>
              Use these types
            </Button>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

/** The ontology a fleet's workers extract with; a run there cannot choose its own. */
export function FleetOntologyCard({
  entityTypes,
  relationTypes,
}: {
  entityTypes: Array<{ name: string; description: string }>;
  relationTypes: Array<{ name: string; description: string }>;
}) {
  const describe = (name: string) =>
    [...entityTypes, ...relationTypes].find((term) => term.name === name)?.description || undefined;
  return (
    <Card data-testid="fleet-ontology">
      <CardHeader>
        <CardTitle>What the fleet extracts</CardTitle>
        <CardDescription>
          The workers share one ontology, and it is part of what makes a run theirs to claim, so every graph built
          here uses these types. Change the fleet&apos;s profile to change them.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        <TermList label="Types" names={entityTypes.map((term) => term.name)} describe={describe} />
        <TermList label="Relations" names={relationTypes.map((term) => term.name)} describe={describe} />
      </CardContent>
    </Card>
  );
}

function TermList({
  label,
  names,
  describe,
}: {
  label: string;
  names: string[];
  describe: (name: string) => string | undefined;
}) {
  if (names.length === 0) {
    return null;
  }
  return (
    <div>
      <p className="font-medium">{label}</p>
      <ul className="mt-1 flex flex-wrap gap-1.5">
        {names.map((name) => (
          <li
            key={name}
            className="rounded-md border border-border bg-muted/40 px-2 py-0.5 font-mono text-xs"
            title={describe(name)}
          >
            {name}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function AnalystHome({
  onOpenRun,
}: {
  onOpenRun: (runId: string) => void;
}) {
  const workspace = trpc.workspace.get.useQuery();
  const runs = trpc.runs.list.useQuery({ limit: 100 });
  const perspectives = (workspace.data?.perspectives ?? []).filter((item) => item.lifecycle === "production");
  return (
    <div className="space-y-6">
      <PageHeader
        kicker="Analyst"
        title="Perspectives"
        description="Published production views. Builders should only publish a perspective after gold is green."
      />
      {perspectives.length === 0 ? (
        <p className="text-sm text-muted-foreground">No production perspectives yet. Ask a builder to publish one after gold.</p>
      ) : null}
      <div className="grid items-start gap-3 md:grid-cols-2">
        {perspectives.map((item) => {
          const run = (runs.data ?? []).find((row) => row.graphId === item.graphId);
          return (
            <Card key={item.id}>
              <CardHeader>
                <CardTitle>{item.name}</CardTitle>
                <CardDescription>{item.suggestedQuestions[0] ?? "Open this neighborhood"}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <Button
                  disabled={!run}
                  onClick={() => run && onOpenRun(run.runId)}
                >
                  Open {item.name}
                </Button>
                {!run ? (
                  <p className="text-sm text-muted-foreground">No succeeded run is on this catalog for that perspective yet.</p>
                ) : null}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}

export function WelcomeBack({
  identified,
  suggestionMode,
  currentRunId = null,
  analyst = false,
  onOpenRun,
}: {
  identified: boolean;
  suggestionMode: "off" | "on-request" | "auto-fill";
  currentRunId?: string | null;
  analyst?: boolean;
  onOpenRun: (runId: string) => void;
}) {
  const runs = trpc.runs.list.useQuery({ limit: 40 });
  const workspace = trpc.workspace.get.useQuery();
  const dismissed = useStorageItem("session", "flakegraph.welcome-dismissed") === "1";
  if (!identified || suggestionMode === "off" || dismissed) {
    return null;
  }
  const skipName = /forget|rename|bulk|copied|disposable|cancelling|active ocr|fleet martial|llm timeout|queued snowflake|empty share/i;
  const productionIds = new Set(
    (workspace.data?.perspectives ?? []).filter((item) => item.lifecycle === "production").map((item) => item.graphId),
  );
  const catalog = (runs.data ?? []).filter((run) => (analyst ? productionIds.has(run.graphId) : true));
  const active = catalog.find(
    (run) =>
      ["queued", "running", "pending", "cancelling"].includes(run.status.toLowerCase()) &&
      !skipName.test(`${run.graphName ?? ""} ${run.graphId}`),
  );
  const finished = catalog.find(
    (run) => run.status.toLowerCase() === "succeeded" && !skipName.test(`${run.graphName ?? ""} ${run.graphId}`),
  );
  const suggested = active ?? finished;
  if (!suggested || suggested.runId === currentRunId) {
    return null;
  }
  return (
    <Alert className="mb-4" data-testid="welcome-back">
      <p className="text-sm">
        Welcome back.
        {active ? ` ${active.graphName || active.graphId} is still ${active.status}.` : ""}
        {!active && finished ? ` ${finished.graphName || finished.graphId} finished and is ready to explore.` : ""}
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        {active ? (
          <Button size="sm" onClick={() => onOpenRun(active.runId)}>
            Resume {active.graphName || active.graphId}
          </Button>
        ) : (
          <Button size="sm" onClick={() => onOpenRun(finished!.runId)}>
            Open {finished?.graphName || finished?.graphId}
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            writeStorage("session", "flakegraph.welcome-dismissed", "1");
          }}
        >
          Dismiss
        </Button>
      </div>
    </Alert>
  );
}
