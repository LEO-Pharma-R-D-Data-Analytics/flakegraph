import path from "node:path";
import { existsSync } from "node:fs";
import type { RuntimeMode } from "./protocol/schema";

export interface AppEnv {
  repositoryRoot: string;
  stateRoot: string;
  defaultRuntime: RuntimeMode;
  kubernetesNamespace: string;
  kubernetesConfig: string | null;
  databaseUrl: string | null;
  flakegraphCommand: string[];
  trustIdentityHeaders: boolean;
  stubRuntimes: boolean;
  snowflakeHosted: boolean;
  requireSignIn: boolean;
}

let cached: AppEnv | null = null;

export function appEnv(): AppEnv {
  cached ??= loadEnv();
  return cached;
}

export function resetAppEnv(): void {
  cached = null;
}

export function loadEnv(overrides: Partial<NodeJS.ProcessEnv> = process.env): AppEnv {
  const cwd = process.cwd();
  const repositoryRoot = resolveRepositoryRoot(
    overrides.FLAKEGRAPH_REPOSITORY_ROOT ?? cwd,
  );
  const configuredState = overrides.FLAKEGRAPH_APP_STATE_ROOT?.trim();
  return {
    repositoryRoot,
    stateRoot: path.resolve(configuredState || path.join(repositoryRoot, ".flakegraph", "app")),
    defaultRuntime: parseRuntime(overrides.FLAKEGRAPH_APP_DEFAULT_RUNTIME, "local"),
    kubernetesNamespace: overrides.FLAKEGRAPH_APP_KUBERNETES_NAMESPACE?.trim() || "flakegraph",
    kubernetesConfig: overrides.FLAKEGRAPH_APP_KUBERNETES_CONFIG?.trim() || null,
    databaseUrl: overrides.DATABASE_URL?.trim() || overrides.FLAKEGRAPH_DATABASE_URL?.trim() || null,
    flakegraphCommand: parseCommand(overrides.FLAKEGRAPH_CLI),
    trustIdentityHeaders: isTruthy(overrides.FLAKEGRAPH_TRUST_IDENTITY_HEADERS),
    stubRuntimes: isTruthy(overrides.FLAKEGRAPH_STUB_RUNTIMES),
    snowflakeHosted: detectSnowflakeHost(overrides),
    requireSignIn: isTruthy(overrides.FLAKEGRAPH_APP_REQUIRE_SIGN_IN),
  };
}

export function parseRuntime(value: string | undefined, fallback: RuntimeMode): RuntimeMode {
  const normalized = value?.trim().toLowerCase();
  if (normalized === "kubernetes" || normalized === "k8s") {
    return "kubernetes";
  }
  if (normalized === "snowflake") {
    return "snowflake";
  }
  if (normalized === "local") {
    return "local";
  }
  return fallback;
}

function parseCommand(value: string | undefined): string[] {
  const trimmed = value?.trim();
  if (trimmed) {
    return trimmed.split(/\s+/);
  }
  return ["uv", "run", "flakegraph"];
}

function isTruthy(value: string | undefined): boolean {
  return ["1", "true", "yes", "on"].includes((value ?? "").trim().toLowerCase());
}

function detectSnowflakeHost(env: Partial<NodeJS.ProcessEnv>): boolean {
  return Boolean(
    env.SNOWFLAKE_ACCOUNT ||
      env.SNOWFLAKE_HOST ||
      env.SNOWFLAKE_QUERY_WAREHOUSE ||
      env.SF_CONTEXT_CURRENT_USER_TOKEN ||
      env.FLAKEGRAPH_SNOWFLAKE_HOSTED,
  );
}

export function resolveRepositoryRoot(start: string): string {
  let current = path.resolve(start);
  for (let i = 0; i < 8; i += 1) {
    if (existsSync(path.join(current, "configs", "app-defaults.yaml"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) {
      break;
    }
    current = parent;
  }
  const fromReact = path.resolve(start, "..");
  if (existsSync(path.join(fromReact, "configs", "app-defaults.yaml"))) {
    return fromReact;
  }
  return path.resolve(start);
}
