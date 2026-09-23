import type { Viewer } from "../protocol/schema";

/** What a grant lets someone do with a graph that is not theirs. */
export type AccessLevel = "read" | "write";

/** A viewer's standing on one graph: its owner, a grantee at a level, or nothing. */
export type GraphRole = "owner" | AccessLevel;

export const ACCESS_LEVELS: readonly AccessLevel[] = ["read", "write"];

export interface GraphGrant {
  principal: string;
  level: AccessLevel;
  /** Who added the grant; `system` for grants written outside the console. */
  grantedBy: string;
  grantedAt: string;
}

export interface GraphAccessRecord {
  owner: string | null;
  grants: GraphGrant[];
}

/**
 * How strictly unclaimed graphs and unnamed viewers are treated.
 *
 * `strict` is a deployment that requires sign-in: every graph belongs to
 * someone, a graph nobody owns is open to nobody, and nobody is exempt.
 *
 * `laptop` is one person's console - no sign-in, demo identities on - where
 * whoever is at the keyboard can read every file anyway: a session that
 * assumed no identity is theirs, and sees and changes everything. A demo
 * identity assumed there sees exactly what that person would.
 *
 * Anywhere else a caller with no name (an anonymous reader) owns nothing: it
 * may read a graph nobody owns, and nothing that belongs to someone.
 */
export interface AccessPolicy {
  strict: boolean;
  laptop?: boolean;
}

/** One spelling per principal: identities arrive in whatever case their source used. */
export function principalKey(name: string | null | undefined): string {
  return (name ?? "").trim().toUpperCase();
}

/** The viewer's role on a graph, or null when the graph is not theirs to see. */
export function roleFor(record: GraphAccessRecord | null, viewer: Viewer, policy: AccessPolicy): GraphRole | null {
  const owner = principalKey(record?.owner);
  const me = principalKey(viewer.userName);
  if (!me && policy.laptop && !policy.strict) {
    return "owner";
  }
  if (!owner) {
    return policy.strict ? null : me ? "owner" : "read";
  }
  if (!me) {
    return null;
  }
  if (owner === me) {
    return "owner";
  }
  const grant = record?.grants.find((item) => principalKey(item.principal) === me);
  return grant ? grant.level : null;
}

const RANK: Record<GraphRole, number> = { read: 1, write: 2, owner: 3 };

/** Whether a role covers what an action needs. */
export function allows(role: GraphRole | null, needed: GraphRole): boolean {
  return role !== null && RANK[role] >= RANK[needed];
}

const PRINCIPAL = /^[A-Za-z0-9._%+'-]+(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})?$/;

/**
 * Check a principal someone typed. Behind a sign-in gate every identity is a
 * user principal name, so an address is required; a laptop's demo
 * identities are bare names.
 */
export function validPrincipal(input: string, policy: AccessPolicy): string | null {
  const value = input.trim();
  if (!value || value.length > 200 || !PRINCIPAL.test(value)) {
    return null;
  }
  if (policy.strict && !value.includes("@")) {
    return null;
  }
  return principalKey(value);
}

export function isAccessLevel(value: unknown): value is AccessLevel {
  return value === "read" || value === "write";
}
