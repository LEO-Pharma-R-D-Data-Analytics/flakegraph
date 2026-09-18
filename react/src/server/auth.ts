import { matchApiKey, presentedApiSecret } from "./api-keys";
import { appEnv } from "./env";
import { unidentifiedViewer, viewerFromHeaders } from "./identity";
import type { Viewer } from "./protocol/schema";
import { loadWorkspace, type ApiKey, type WorkspaceState } from "./workspace";

/** A request the control plane refuses to serve, and the status that says so. */
export class RequestRefused extends Error {
  readonly status = 401;
}

export interface Authorized {
  viewer: Viewer;
  /** The machine key the request presented, when it did; a browser session presents none. */
  machine: ApiKey | null;
  /** True when the viewer was assumed from the workspace, a rehearsal device for an ungated console. */
  assumed: boolean;
}

/**
 * Decide who a request is from, or refuse it.
 *
 * A machine presents an API key and is served as a machine whether or not a
 * gate stands in front of the console; a key that matches nothing is refused.
 * A browser is whoever the gate says it is. Where the deployment requires a
 * sign-in, a request that is neither is refused: behind a gate that only
 * happens to a caller that reached the service without passing the gate, and
 * such a caller gets nothing.
 */
export async function authorizeRequest(headers: Headers, workspace?: WorkspaceState): Promise<Authorized> {
  const env = appEnv();
  const state = workspace ?? (await loadWorkspace(env.stateRoot));
  const presentedSecret = presentedApiSecret(headers);
  if (presentedSecret) {
    const key = matchApiKey(state.apiKeys, presentedSecret);
    if (!key) {
      throw new RequestRefused("API key missing or revoked. This path does not use SSO.");
    }
    return { viewer: unidentifiedViewer(), machine: key, assumed: false };
  }
  const headerViewer = viewerFromHeaders(headers, {
    trustIdentityHeaders: env.trustIdentityHeaders,
    snowflakeHosted: env.snowflakeHosted,
  });
  // A gate that established who the viewer is outranks a runtime's own
  // answer; an assumed identity is a rehearsal device for a control plane
  // nobody has signed in to, never an override of the gate.
  const assumed = env.trustIdentityHeaders ? null : state.identity;
  const viewer = assumed ?? headerViewer;
  if (env.requireSignIn && !viewer.userName) {
    throw new RequestRefused("Sign in through the gate, or present an API key.");
  }
  return { viewer, machine: null, assumed: Boolean(assumed) };
}

/** The 401 a route handler answers with when `authorizeRequest` refused. */
export function refusal(error: unknown): Response | null {
  if (error instanceof RequestRefused) {
    return Response.json({ error: error.message }, { status: error.status });
  }
  return null;
}
