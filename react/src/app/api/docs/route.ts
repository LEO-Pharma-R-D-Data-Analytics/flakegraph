import { CONTROL_PLANE_API, CONTROL_PLANE_PROCEDURES } from "@/lib/control-plane-api";

export async function GET(request: Request) {
  const origin = new URL(request.url).origin;
  return Response.json({
    service: "flakegraph-control-plane",
    origin,
    endpoint: `${origin}${CONTROL_PLANE_API.endpoint}`,
    ask: `${origin}${CONTROL_PLANE_API.askPath}`,
    humanGuide: `${origin}${CONTROL_PLANE_API.humanGuidePath}`,
    health: `${origin}${CONTROL_PLANE_API.healthPath}`,
    auth: {
      environmentVariable: "FLAKEGRAPH_API_KEY",
      headers: ["Authorization: Bearer $FLAKEGRAPH_API_KEY", "x-flakegraph-api-key: $FLAKEGRAPH_API_KEY"],
      runtimeHeader: "x-flakegraph-runtime: local | kubernetes | snowflake",
      note: "A presented key authenticates as a machine, not as a browser SSO session. 401 means the key is missing or revoked.",
    },
    wireFormat: CONTROL_PLANE_API.wireFormat,
    trpcHttpDocs: CONTROL_PLANE_API.trpcHttpDocs,
    sourceOfTruth: {
      procedures: CONTROL_PLANE_API.checkoutProcedures,
      payloads: CONTROL_PLANE_API.checkoutPayloads,
      ask: CONTROL_PLANE_API.checkoutAsk,
      fleetMachinePaths: CONTROL_PLANE_API.fleetGuide,
    },
    askStream: {
      path: `${origin}${CONTROL_PLANE_API.askPath}`,
      methods: ["GET", "POST"],
      formats: ["ui", "ndjson"],
      events: ["status", "tool", "text", "citation", "error", "done"],
      note: "POST /api/ask streams. graphs.ask is the non-streaming JSON convenience wrapper.",
    },
    procedures: CONTROL_PLANE_PROCEDURES,
  });
}
