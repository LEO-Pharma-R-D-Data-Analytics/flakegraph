import "server-only";
import { appEnv, parseRuntime } from "../env";
import type { RuntimeMode, Viewer } from "../protocol/schema";
import type { ControlPlane } from "../protocol/runtime";
import { createKubernetesRuntime } from "./kubernetes";
import { createLocalRuntime } from "./local";
import { createSnowflakeRuntime } from "./snowflake";

export function resolveRuntime(runtime: RuntimeMode, viewer?: Viewer): ControlPlane {
  const env = appEnv();
  const selected = env.snowflakeHosted ? "snowflake" : runtime;
  if (selected === "kubernetes") {
    return createKubernetesRuntime(viewer ?? null);
  }
  if (selected === "snowflake") {
    return createSnowflakeRuntime(viewer);
  }
  return createLocalRuntime(viewer ?? null);
}

export function runtimeFromHeader(header: string | null): RuntimeMode {
  const env = appEnv();
  if (env.snowflakeHosted) {
    return "snowflake";
  }
  return parseRuntime(header ?? undefined, env.defaultRuntime);
}

export function availableRuntimes(): RuntimeMode[] {
  if (appEnv().snowflakeHosted) {
    return ["snowflake"];
  }
  return ["local", "kubernetes", "snowflake"];
}
