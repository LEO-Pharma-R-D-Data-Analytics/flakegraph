import { initTRPC, TRPCError } from "@trpc/server";
import { Cause, Effect, Exit, Option } from "effect";
import superjson from "superjson";
import { authorizeRequest, mayMutate, RequestRefused, SIGN_IN_HINT, type Authorized, type AuthorizationMode } from "../auth";
import { ControlPlaneError, toTrpcCode } from "../protocol/errors";
import type { GraphAccessPlane } from "../protocol/runtime";
import { notePerson } from "../access/controlled-plane";
import { appEnv } from "../env";
import type { RuntimeMode, Viewer } from "../protocol/schema";
import { resolveRuntime, runtimeFromHeader } from "../runtime/resolve";

export interface TrpcContext {
  headers: Headers;
  runtimeName: RuntimeMode;
  controlPlane: GraphAccessPlane;
  viewer: Viewer;
  /** How the caller was admitted; an anonymous caller reads and changes nothing. */
  authorization: AuthorizationMode;
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
  // Access is decided against whoever the request is finally served as: the
  // key's principal, the gate's or assumed identity, else the runtime's own.
  const viewer =
    authorized.machine || authorized.assumed || authorized.viewer.userName
      ? authorized.viewer
      : await runEffect(resolveRuntime(runtimeName, authorized.viewer).viewer());
  const controlPlane = resolveRuntime(runtimeName, viewer);
  await notePerson(appEnv().stateRoot, viewer).catch(() => undefined);
  return {
    headers: opts.headers,
    runtimeName: controlPlane.runtime,
    controlPlane,
    authorization: authorized.mode,
    viewer,
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

/**
 * A read-only query: served to whoever the context admitted.
 *
 * A middleware's `next()` resolves with the resolver's failure rather than
 * throwing it, so a control-plane error is read off the result and given
 * its own status; left alone, every refusal would answer 500.
 */
export const publicProcedure = t.procedure.use(async ({ next }) => {
  const result = await next();
  if (!result.ok) {
    const cause = controlPlaneCause(result.error);
    if (cause) {
      throw new TRPCError({ code: toTrpcCode(cause), message: cause.message, cause });
    }
  }
  return result;
});

/** The control-plane error under a tRPC error, whether thrown directly or out of an Effect. */
function controlPlaneCause(error: unknown): ControlPlaneError | null {
  let current: unknown = error;
  for (let depth = 0; depth < 5 && current; depth += 1) {
    if (current instanceof ControlPlaneError) {
      return current;
    }
    current = current instanceof Error ? current.cause : null;
  }
  return null;
}

/**
 * A procedure that changes something, spawns something, or names a path on
 * this host. Served only to a caller somebody vouched for: a gate, a key,
 * or the demo identities of a console that has no gate.
 */
export const protectedProcedure = publicProcedure.use(async ({ ctx, next }) => {
  if (!mayMutate({ mode: ctx.authorization })) {
    throw new TRPCError({ code: "UNAUTHORIZED", message: SIGN_IN_HINT });
  }
  return next();
});

/**
 * A procedure that acts as a named principal: minting a key that will act
 * as them, or assuming who they are. A console that serves an anonymous
 * demo session still has nobody to bind such a thing to.
 */
export const identifiedProcedure = protectedProcedure.use(async ({ ctx, next }) => {
  if (!ctx.viewer.userName) {
    throw new TRPCError({
      code: "UNAUTHORIZED",
      message: "Sign in first: this is bound to who you are, and nobody has said who that is.",
    });
  }
  return next();
});

/**
 * Run a control-plane effect and surface its failure as the error it was.
 * `Effect.runPromise` rejects with a FiberFailure that only resembles the
 * error, so the exit is read directly and the typed failure rethrown.
 */
export async function runEffect<A>(effect: Effect.Effect<A, ControlPlaneError>): Promise<A> {
  const exit = await Effect.runPromiseExit(effect);
  if (Exit.isSuccess(exit)) {
    return exit.value;
  }
  const failure = Cause.failureOption(exit.cause);
  if (Option.isSome(failure)) {
    throw failure.value;
  }
  throw Cause.squash(exit.cause);
}
