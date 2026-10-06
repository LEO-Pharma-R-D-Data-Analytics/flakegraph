import path from "node:path";
import { existsSync } from "node:fs";
import type { RuntimeMode } from "./protocol/schema";

export interface AppEnv {
  repositoryRoot: string;
  stateRoot: string;
  defaultRuntime: RuntimeMode;
  kubernetesNamespace: string;
  databaseUrl: string | null;
  flakegraphCommand: string[];
  /**
   * Believe the identity headers a sign-in gate writes. Sound only when
   * nothing but that gate can reach this server: a caller who reaches it
   * directly can write the same headers and become anyone.
   */
  trustIdentityHeaders: boolean;
  stubRuntimes: boolean;
  /** Running inside a Snowflake container, where Sf-Context headers name the viewer. */
  snowflakeHosted: boolean;
  /** Refuse an anonymous caller everywhere, even where demo identities or anonymous reads would serve one. */
  requireSignIn: boolean;
  /**
   * A laptop or rehearsal console with no gate: the viewer is whoever the
   * workspace says was assumed, and a session that assumed nobody is still
   * served. Production never sets this.
   */
  demoIdentities: boolean;
  /** Serve read-only catalog queries to a caller nobody identified. Closed by default. */
  anonymousRead: boolean;
  /**
   * Folders a run may read documents from, besides the checkout and the
   * console's own state. Paths a request names are confined to these roots.
   */
  sourceRoots: string[];
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
    stateRoot: path.resolve(
      /* turbopackIgnore: true */ configuredState || path.join(repositoryRoot, ".flakegraph", "app"),
    ),
    defaultRuntime: parseRuntime(overrides.FLAKEGRAPH_APP_DEFAULT_RUNTIME, "local"),
    kubernetesNamespace: overrides.FLAKEGRAPH_APP_KUBERNETES_NAMESPACE?.trim() || "flakegraph",
    databaseUrl: overrides.DATABASE_URL?.trim() || overrides.FLAKEGRAPH_DATABASE_URL?.trim() || null,
    flakegraphCommand: parseCommand(overrides.FLAKEGRAPH_CLI),
    trustIdentityHeaders: isTruthy(overrides.FLAKEGRAPH_TRUST_IDENTITY_HEADERS),
    stubRuntimes: isTruthy(overrides.FLAKEGRAPH_STUB_RUNTIMES),
    // Only an explicit flag: ambient SNOWFLAKE_* variables belong to the
    // Python writer and must not switch the console into trusting
    // Sf-Context headers from whoever sends them.
    snowflakeHosted: isTruthy(overrides.FLAKEGRAPH_SNOWFLAKE_HOSTED),
    requireSignIn: isTruthy(overrides.FLAKEGRAPH_APP_REQUIRE_SIGN_IN),
    demoIdentities: isTruthy(overrides.FLAKEGRAPH_DEMO_IDENTITIES),
    anonymousRead: isTruthy(overrides.FLAKEGRAPH_APP_ANONYMOUS_READ),
    sourceRoots: parsePathList(overrides.FLAKEGRAPH_APP_SOURCE_ROOTS),
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

function parsePathList(value: string | undefined): string[] {
  return (value ?? "")
    .split(path.delimiter)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => path.resolve(item));
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
