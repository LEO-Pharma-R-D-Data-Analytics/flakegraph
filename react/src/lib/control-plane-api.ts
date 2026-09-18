export const CONTROL_PLANE_API = {
  endpoint: "/api/trpc",
  askPath: "/api/ask",
  docsPath: "/api/docs",
  humanGuidePath: "/?page=keys",
  healthPath: "/api/health",
  wireFormat: "tRPC HTTP with SuperJSON envelopes ({ json: … }). Queries are GET with ?input=; mutations are POST with a JSON body. Ask streaming is a separate POST /api/ask that emits AI SDK UI chunks or application/x-ndjson.",
  trpcHttpDocs: "https://trpc.io/docs/rpc",
  checkoutProcedures: "react/src/server/trpc/router.ts",
  checkoutPayloads: "react/src/server/protocol/schema.ts",
  checkoutAsk: "react/src/server/ask/",
  fleetGuide: "docs/kubernetes-fleet.md (Callers that are programs)",
} as const;

export const CONTROL_PLANE_PROCEDURES = [
  { name: "runs.list", kind: "query", purpose: "List graphs on this catalog" },
  { name: "runs.get", kind: "query", purpose: "Status, stages, and errors for one job" },
  { name: "runs.documents", kind: "query", purpose: "Per-file OCR and extract progress" },
  { name: "runs.submit", kind: "mutation", purpose: "Start ingestion with the same payload as compose" },
  { name: "runs.cancel", kind: "mutation", purpose: "Cancel an in-flight job" },
  { name: "runs.retry", kind: "mutation", purpose: "Retry failed documents" },
  { name: "runs.recover", kind: "mutation", purpose: "Reconcile Kubernetes workers" },
  { name: "graphs.loadRun", kind: "query", purpose: "Entities, relations, and evidence" },
  { name: "graphs.quality", kind: "query", purpose: "QA gates and gold compare" },
  { name: "graphs.inspect", kind: "query", purpose: "Inspect JSON for evals and reports" },
  { name: "graphs.ask", kind: "mutation", purpose: "Ask a completed graph a question (JSON convenience; prefer POST /api/ask to stream)" },
  { name: "ingestion.preflight", kind: "mutation", purpose: "Validate a compose payload before Start" },
  { name: "fleet.cluster", kind: "query", purpose: "Kubernetes node and queue snapshot" },
] as const;

export function listRunsExample(origin: string): string {
  return `export FLAKEGRAPH_API_KEY=fg_…
curl -sS \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "x-flakegraph-runtime: local" \\
  '${origin}/api/trpc/runs.list?input={"json":{"limit":5}}'`;
}

export function getRunExample(origin: string): string {
  return `export FLAKEGRAPH_API_KEY=fg_…
# Use a runId from runs.list
curl -sS \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "x-flakegraph-runtime: local" \\
  '${origin}/api/trpc/runs.get?input={"json":{"runId":"run_martial_arts"}}'`;
}

export function askGraphExample(origin: string): string {
  return `export FLAKEGRAPH_API_KEY=fg_…
curl -sS -X POST \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "content-type: application/json" \\
  -H "x-flakegraph-runtime: local" \\
  -d '{"json":{"runId":"run_martial_arts","question":"Who developed judo?","mode":"local"}}' \\
  "${origin}/api/trpc/graphs.ask"`;
}

export function askStreamExample(origin: string): string {
  return `export FLAKEGRAPH_API_KEY=fg_…
# NDJSON: status, tool, text, citation, done
curl -N -X POST \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "content-type: application/json" \\
  -H "accept: application/x-ndjson" \\
  -H "x-flakegraph-runtime: local" \\
  -d '{"runId":"run_martial_arts","question":"Who developed judo?","mode":"local","format":"ndjson"}' \\
  "${origin}/api/ask"

# TypeScript consumer (AI SDK UI stream)
# useChat({ transport: new DefaultChatTransport({ api: "${origin}/api/ask", body: { runId, mode, format: "ui" } }) })`;
}

export function pythonClientExample(origin: string): string {
  return `import json, os, urllib.parse, urllib.request

origin = "${origin}"
headers = {
    "Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}",
    "x-flakegraph-runtime": "local",
}

def query(procedure, payload):
    encoded = urllib.parse.quote(json.dumps({"json": payload}))
    request = urllib.request.Request(
        f"{origin}/api/trpc/{procedure}?input={encoded}",
        headers=headers,
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)

def mutate(procedure, payload):
    request = urllib.request.Request(
        f"{origin}/api/trpc/{procedure}",
        data=json.dumps({"json": payload}).encode(),
        headers={**headers, "content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)

runs = query("runs.list", {"limit": 5})
run_id = runs["result"]["data"]["json"][0]["runId"]
print(query("runs.get", {"runId": run_id}))
print(mutate("graphs.ask", {
    "runId": run_id,
    "question": "Who developed judo?",
    "mode": "local",
}))

# Streaming Ask (status + tokens). Parse each line as JSON.
ask = urllib.request.Request(
    f"{origin}/api/ask",
    data=json.dumps({
        "runId": run_id,
        "question": "Who developed judo?",
        "mode": "local",
        "format": "ndjson",
    }).encode(),
    headers={**headers, "content-type": "application/json", "accept": "application/x-ndjson"},
    method="POST",
)
with urllib.request.urlopen(ask) as response:
    for line in response:
        print(json.loads(line))`;
}
