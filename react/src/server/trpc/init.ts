import { initTRPC, TRPCError } from "@trpc/server";
import { Effect } from "effect";
import superjson from "superjson";
import { authorizeRequest, RequestRefused, type Authorized } from "../auth";
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
  const runtimeName = runtimeFromHeader(opts.headers.get("x-flakegraph-runtime"));
  let authorized: Authorized;
  try {
    authorized = await authorizeRequest(opts.headers);
  } catch (error) {
    if (error instanceof RequestRefused) {
      throw new TRPCError({ code: "UNAUTHORIZED", message: error.message });
    }
    throw error;
  }
  const controlPlane = resolveRuntime(runtimeName, authorized.viewer);
  if (authorized.machine || authorized.assumed) {
    return {
      headers: opts.headers,
      runtimeName: controlPlane.runtime,
      controlPlane,
      viewer: authorized.viewer,
    };
  }
  const runtimeViewer = authorized.viewer.userName
    ? authorized.viewer
    : await runEffect(controlPlane.viewer());
  return {
    headers: opts.headers,
    runtimeName: controlPlane.runtime,
    controlPlane,
    viewer: runtimeViewer,
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
