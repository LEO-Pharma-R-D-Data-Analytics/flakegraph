import { askCatalog, handleAskRequest } from "@/server/ask/stream";
import { resolveAskModel } from "@/server/ask/model";

export const runtime = "nodejs";
export const maxDuration = 120;

export async function GET(request: Request) {
  return Response.json(askCatalog(new URL(request.url).origin));
}

export async function POST(request: Request) {
  resolveAskModel();
  return handleAskRequest(request);
}
