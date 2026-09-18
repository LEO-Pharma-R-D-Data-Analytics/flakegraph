import { appEnv } from "@/server/env";
import { resolveRuntime } from "@/server/runtime/resolve";
import { loadWorkspace } from "@/server/workspace";
import { qualityForGraph } from "@/server/quality";
import { inspectFromDataset, inspectHtml } from "@/server/inspect-report";
import { Effect } from "effect";
import { unidentifiedViewer } from "@/server/identity";
import { readRunRecord, runDirectory, runRecordExists } from "@/server/catalog";
import type { RuntimeMode } from "@/server/protocol/schema";

export async function GET(request: Request) {
  const runId = new URL(request.url).searchParams.get("run");
  if (!runId) {
    return new Response("Missing run", { status: 400 });
  }
  const workspace = await loadWorkspace(appEnv().stateRoot);
  const viewer = workspace.identity ?? unidentifiedViewer();
  // The run's own record says which runtime holds it; a run the console has
  // no record of is tried on each runtime in turn.
  const record = runRecordExists(runDirectory(appEnv().stateRoot, runId))
    ? await readRunRecord(runDirectory(appEnv().stateRoot, runId))
    : null;
  const candidates: RuntimeMode[] = record?.runtime
    ? [record.runtime as RuntimeMode]
    : ["local", "kubernetes", "snowflake"];
  let failure = "Inspect failed";
  for (const mode of candidates) {
    const runtime = resolveRuntime(mode, viewer);
    try {
      const snapshot = await Effect.runPromise(runtime.getRun(runId));
      const dataset = await Effect.runPromise(runtime.loadRunGraph(snapshot));
      const quality = await qualityForGraph({
        snapshot,
        dataset,
        repositoryRoot: appEnv().repositoryRoot,
      });
      const html = inspectHtml(inspectFromDataset(snapshot, dataset, quality));
      return new Response(html, { headers: { "content-type": "text/html; charset=utf-8" } });
    } catch (error) {
      failure = error instanceof Error ? error.message : failure;
    }
  }
  return new Response(failure, { status: 404 });
}
