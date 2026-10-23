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

export const EXAMPLE_LANGUAGES = [
  { id: "typescript", label: "TypeScript" },
  { id: "python", label: "Python" },
  { id: "curl", label: "curl" },
] as const;

export type ExampleLanguage = (typeof EXAMPLE_LANGUAGES)[number]["id"];

export type TaskExamples = Record<ExampleLanguage, string>;

export function listRunsExamples(origin: string): TaskExamples {
  return {
    curl: `export FLAKEGRAPH_API_KEY=fg_…
curl -sS \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "x-flakegraph-runtime: local" \\
  '${origin}/api/trpc/runs.list?input={"json":{"limit":5}}'`,
    typescript: `const origin = "${origin}";
const headers = {
  Authorization: \`Bearer \${process.env.FLAKEGRAPH_API_KEY}\`,
  "x-flakegraph-runtime": "local",
};
const input = encodeURIComponent(JSON.stringify({ json: { limit: 5 } }));
const response = await fetch(\`\${origin}/api/trpc/runs.list?input=\${input}\`, { headers });
console.log(await response.json());`,
    python: `import json, os, urllib.parse, urllib.request

origin = "${origin}"
headers = {
    "Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}",
    "x-flakegraph-runtime": "local",
}
encoded = urllib.parse.quote(json.dumps({"json": {"limit": 5}}))
request = urllib.request.Request(
    f"{origin}/api/trpc/runs.list?input={encoded}",
    headers=headers,
)
with urllib.request.urlopen(request) as response:
    print(json.load(response))`,
  };
}

export function getRunExamples(origin: string): TaskExamples {
  return {
    curl: `export FLAKEGRAPH_API_KEY=fg_…
# Use a runId from runs.list
curl -sS \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "x-flakegraph-runtime: local" \\
  '${origin}/api/trpc/runs.get?input={"json":{"runId":"run_martial_arts"}}'`,
    typescript: `const origin = "${origin}";
const headers = {
  Authorization: \`Bearer \${process.env.FLAKEGRAPH_API_KEY}\`,
  "x-flakegraph-runtime": "local",
};
const input = encodeURIComponent(JSON.stringify({ json: { runId: "run_martial_arts" } }));
const response = await fetch(\`\${origin}/api/trpc/runs.get?input=\${input}\`, { headers });
console.log(await response.json());`,
    python: `import json, os, urllib.parse, urllib.request

origin = "${origin}"
headers = {
    "Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}",
    "x-flakegraph-runtime": "local",
}
encoded = urllib.parse.quote(json.dumps({"json": {"runId": "run_martial_arts"}}))
request = urllib.request.Request(
    f"{origin}/api/trpc/runs.get?input={encoded}",
    headers=headers,
)
with urllib.request.urlopen(request) as response:
    print(json.load(response))`,
  };
}

export function askGraphExamples(origin: string): TaskExamples {
  return {
    curl: `export FLAKEGRAPH_API_KEY=fg_…
curl -sS -X POST \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "content-type: application/json" \\
  -H "x-flakegraph-runtime: local" \\
  -d '{"json":{"runId":"run_martial_arts","question":"Who developed judo?","mode":"local"}}' \\
  "${origin}/api/trpc/graphs.ask"`,
    typescript: `const origin = "${origin}";
const headers = {
  Authorization: \`Bearer \${process.env.FLAKEGRAPH_API_KEY}\`,
  "content-type": "application/json",
  "x-flakegraph-runtime": "local",
};
const response = await fetch(\`\${origin}/api/trpc/graphs.ask\`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    json: { runId: "run_martial_arts", question: "Who developed judo?", mode: "local" },
  }),
});
console.log(await response.json());`,
    python: `import json, os, urllib.request

origin = "${origin}"
request = urllib.request.Request(
    f"{origin}/api/trpc/graphs.ask",
    data=json.dumps({
        "json": {
            "runId": "run_martial_arts",
            "question": "Who developed judo?",
            "mode": "local",
        }
    }).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}",
        "content-type": "application/json",
        "x-flakegraph-runtime": "local",
    },
    method="POST",
)
with urllib.request.urlopen(request) as response:
    print(json.load(response))`,
  };
}

export function askStreamExamples(origin: string): TaskExamples {
  return {
    curl: `export FLAKEGRAPH_API_KEY=fg_…
# NDJSON: status, tool, text, citation, done
curl -N -X POST \\
  -H "Authorization: Bearer $FLAKEGRAPH_API_KEY" \\
  -H "content-type: application/json" \\
  -H "accept: application/x-ndjson" \\
  -H "x-flakegraph-runtime: local" \\
  -d '{"runId":"run_martial_arts","question":"Who developed judo?","mode":"local","format":"ndjson"}' \\
  "${origin}/api/ask"`,
    typescript: `import { DefaultChatTransport } from "ai";
import { useChat } from "@ai-sdk/react";

// Browser / useChat consumer
useChat({
  transport: new DefaultChatTransport({
    api: "${origin}/api/ask",
    body: { runId: "run_martial_arts", mode: "local", format: "ui" },
  }),
});

// Script / eval harness: NDJSON status, tool, text, citation, done
const origin = "${origin}";
const response = await fetch(\`\${origin}/api/ask\`, {
  method: "POST",
  headers: {
    Authorization: \`Bearer \${process.env.FLAKEGRAPH_API_KEY}\`,
    "content-type": "application/json",
    accept: "application/x-ndjson",
    "x-flakegraph-runtime": "local",
  },
  body: JSON.stringify({
    runId: "run_martial_arts",
    question: "Who developed judo?",
    mode: "local",
    format: "ndjson",
  }),
});
for (const line of (await response.text()).split("\\n").filter(Boolean)) {
  console.log(JSON.parse(line));
}`,
    python: `import json, os, urllib.request

origin = "${origin}"
request = urllib.request.Request(
    f"{origin}/api/ask",
    data=json.dumps({
        "runId": "run_martial_arts",
        "question": "Who developed judo?",
        "mode": "local",
        "format": "ndjson",
    }).encode(),
    headers={
        "Authorization": f"Bearer {os.environ['FLAKEGRAPH_API_KEY']}",
        "content-type": "application/json",
        "accept": "application/x-ndjson",
        "x-flakegraph-runtime": "local",
    },
    method="POST",
)
with urllib.request.urlopen(request) as response:
    for line in response:
        print(json.loads(line))`,
  };
}
