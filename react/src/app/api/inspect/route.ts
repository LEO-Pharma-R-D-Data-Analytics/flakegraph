import { appEnv } from "@/server/env";
import { resolveRuntime } from "@/server/runtime/resolve";
import { loadWorkspace } from "@/server/workspace";
import { qualityForGraph } from "@/server/quality";
import { inspectFromDataset, inspectHtml } from "@/server/inspect-report";
import { Effect } from "effect";
import { unidentifiedViewer } from "@/server/identity";

export async function GET(request: Request) {
  const runId = new URL(request.url).searchParams.get("run");
  if (!runId) {
    return new Response("Missing run", { status: 400 });
  }
  const workspace = await loadWorkspace(appEnv().stateRoot);
  const runtime = resolveRuntime("local", workspace.identity ?? unidentifiedViewer());
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
  } catch {
    const snowflake = resolveRuntime("snowflake", workspace.identity ?? unidentifiedViewer());
    try {
      const snapshot = await Effect.runPromise(snowflake.getRun(runId));
      const dataset = await Effect.runPromise(snowflake.loadRunGraph(snapshot));
      const quality = await qualityForGraph({
        snapshot,
        dataset,
        repositoryRoot: appEnv().repositoryRoot,
      });
      const html = inspectHtml(inspectFromDataset(snapshot, dataset, quality));
      return new Response(html, { headers: { "content-type": "text/html; charset=utf-8" } });
    } catch (error) {
      return new Response(error instanceof Error ? error.message : "Inspect failed", { status: 404 });
    }
  }
}
