import type { Viewer } from "./protocol/schema";

const VIEWER_NAME_HEADERS = [
  "x-forwarded-preferred-username",
  "x-forwarded-user",
  "x-forwarded-email",
  "x-auth-request-user",
  "x-auth-request-email",
];

export function unidentifiedViewer(): Viewer {
  return { userName: "", email: "", roles: [] };
}

export function viewerFromHeaders(
  headers: Headers,
  options: { trustIdentityHeaders: boolean; snowflakeHosted: boolean },
): Viewer {
  if (options.snowflakeHosted) {
    const tokenUser = headers.get("sf-context-current-user") ?? headers.get("Sf-Context-Current-User");
    const email = headers.get("sf-context-current-user-email") ?? "";
    if (tokenUser?.trim()) {
      return {
        userName: tokenUser.trim().toUpperCase(),
        email: email.trim(),
        roles: parseRoles(headers.get("sf-context-current-user-roles")),
      };
    }
  }
  if (!options.trustIdentityHeaders) {
    return unidentifiedViewer();
  }
  for (const header of VIEWER_NAME_HEADERS) {
    const value = headers.get(header)?.trim();
    if (value) {
      const email = headers.get("x-forwarded-email") ?? headers.get("x-auth-request-email") ?? "";
      return {
        userName: value.toUpperCase(),
        email,
        roles: parseRoles(headers.get("x-forwarded-groups") ?? headers.get("x-auth-request-groups")),
      };
    }
  }
  return unidentifiedViewer();
}

export function visibleGraphPredicate(
  viewer: Viewer,
  alias = "G",
): { sql: string; parameters: string[] } {
  if (!viewer.userName) {
    return { sql: `${alias}.OWNER IS NULL`, parameters: [] };
  }
  const roles = [...viewer.roles].sort();
  const parameters = [viewer.userName, viewer.userName];
  let roleArm = "";
  if (roles.length > 0) {
    const placeholders = roles.map(() => "?").join(", ");
    roleArm = ` OR (UPPER(A.GRANTEE_TYPE) = 'ROLE' AND UPPER(A.GRANTEE) IN (${placeholders}))`;
    parameters.push(...roles);
  }
  const shared =
    `EXISTS (SELECT 1 FROM KG_GRAPH_ACL A WHERE A.GRAPH_ID = ${alias}.GRAPH_ID ` +
    `AND ((UPPER(A.GRANTEE_TYPE) = 'USER' AND UPPER(A.GRANTEE) = ?)${roleArm}))`;
  return {
    sql: `(${alias}.OWNER IS NULL OR UPPER(${alias}.OWNER) = ? OR ${shared})`,
    parameters,
  };
}

export function administrableGraphPredicate(
  viewer: Viewer,
  alias = "G",
): { sql: string; parameters: string[] } {
  if (!viewer.userName) {
    return { sql: `${alias}.OWNER IS NULL`, parameters: [] };
  }
  return {
    sql: `(${alias}.OWNER IS NULL OR UPPER(${alias}.OWNER) = ?)`,
    parameters: [viewer.userName],
  };
}

export function viewerStagePrefix(viewer: Viewer, jobId: string): string {
  const who = sanitizePrincipal(viewer.userName || "shared");
  return `app/${who}/${jobId}`;
}

export function mayReadStagePath(viewer: Viewer, path: string): boolean {
  const segments = path.trim().replace(/^\/+|\/+$/g, "").split("/").filter(Boolean);
  if (segments.length === 0) {
    return false;
  }
  if (segments[0] !== "app") {
    return true;
  }
  const mine = viewerStagePrefix(viewer, "").replace(/\/+$/, "").split("/");
  return mine.every((segment, index) => segments[index] === segment);
}

function parseRoles(value: string | null): string[] {
  if (!value) {
    return [];
  }
  return value
    .split(/[,\s]+/)
    .map((role) => role.trim().toUpperCase())
    .filter(Boolean);
}

function sanitizePrincipal(value: string): string {
  const cleaned = value.replace(/[^A-Za-z0-9._-]/g, "_").replace(/^_+|_+$/g, "");
  if (!cleaned || [...cleaned].every((character) => character === ".")) {
    return "shared";
  }
  return cleaned;
}
