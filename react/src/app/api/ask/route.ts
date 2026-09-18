import { askCatalog, handleAskRequest } from "@/server/ask/stream";
import { resolveAskModel } from "@/server/ask/model";
import { authorizeRequest, refusal } from "@/server/auth";

export const runtime = "nodejs";
export const maxDuration = 120;

export async function GET(request: Request) {
  return Response.json(askCatalog(new URL(request.url).origin));
}

export async function POST(request: Request) {
  try {
    await authorizeRequest(request.headers);
  } catch (error) {
    return refusal(error) ?? Promise.reject(error);
  }
  resolveAskModel();
  return handleAskRequest(request);
}
