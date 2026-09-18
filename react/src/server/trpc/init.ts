import { initTRPC, TRPCError } from "@trpc/server";
import { Effect } from "effect";
import superjson from "superjson";
import { matchApiKey, presentedApiSecret } from "../api-keys";
import { appEnv } from "../env";
import { unidentifiedViewer, viewerFromHeaders } from "../identity";
import { loadWorkspace } from "../workspace";
import { ControlPlaneError, toTrpcCode } from "../protocol/errors";
import type { ControlPlane } from "../protocol/runtime";
import type { RuntimeMode, Viewer } from "../protocol/schema";
import { resolveRuntime, runtimeFromHeader } from "../runtime/resolve";

export interface TrpcContext {
  headers: Headers;
  runtimeName: RuntimeMode;
  controlPlane: ControlPlane;
  viewer: Viewer;
}

export async function createContext(opts: { headers: Headers }): Promise<TrpcContext> {
  const env = appEnv();
  const runtimeName = runtimeFromHeader(opts.headers.get("x-flakegraph-runtime"));
  const workspace = await loadWorkspace(env.stateRoot);
  const presentedSecret = presentedApiSecret(opts.headers);
  if (presentedSecret) {
    const key = matchApiKey(workspace.apiKeys, presentedSecret);
    if (!key) {
      throw new TRPCError({
        code: "UNAUTHORIZED",
        message: "API key missing or revoked. This path does not use SSO.",
      });
    }
    const viewer = unidentifiedViewer();
    const controlPlane = resolveRuntime(runtimeName, viewer);
    return {
      headers: opts.headers,
      runtimeName: controlPlane.runtime,
      controlPlane,
      viewer,
    };
  }
  const headerViewer = viewerFromHeaders(opts.headers, {
    trustIdentityHeaders: env.trustIdentityHeaders,
    snowflakeHosted: env.snowflakeHosted,
  });
  // A gate that established who the viewer is outranks a runtime's own
  // answer; an assumed identity is a rehearsal device for a control plane
  // nobody has signed in to, never an override of the gate.
  const assumed = env.trustIdentityHeaders ? null : workspace.identity;
  const viewer = assumed ?? headerViewer;
  const controlPlane = resolveRuntime(runtimeName, viewer);
  const runtimeViewer = headerViewer.userName ? headerViewer : await runEffect(controlPlane.viewer());
  return {
    headers: opts.headers,
    runtimeName: controlPlane.runtime,
    controlPlane,
    viewer: assumed ?? runtimeViewer,
  };
}

const t = initTRPC.context<TrpcContext>().create({
  transformer: superjson,
  errorFormatter({ shape, error }) {
    return {
      ...shape,
      data: {
        ...shape.data,
        controlPlaneCode: error.cause instanceof ControlPlaneError ? error.cause.code : undefined,
      },
    };
  },
});

export const router = t.router;
export const publicProcedure = t.procedure.use(async ({ next }) => {
  try {
    return await next();
  } catch (error) {
    if (error instanceof ControlPlaneError) {
      throw new TRPCError({
        code: toTrpcCode(error),
        message: error.message,
        cause: error,
      });
    }
    throw error;
  }
});

export function runEffect<A>(effect: Effect.Effect<A, ControlPlaneError>): Promise<A> {
  return Effect.runPromise(effect);
}
