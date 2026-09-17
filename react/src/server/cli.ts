import { Command } from "@effect/platform";
import { NodeContext } from "@effect/platform-node";
import { Effect } from "effect";
import { spawn, type ChildProcess } from "node:child_process";
import { createWriteStream } from "node:fs";
import { appEnv } from "./env";

export interface CommandResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

export async function runFlakegraph(
  args: string[],
  options: { cwd: string; env?: Record<string, string> } = { cwd: appEnv().repositoryRoot },
): Promise<CommandResult> {
  const [binary, ...prefix] = appEnv().flakegraphCommand;
  const command = Command.make(binary, ...prefix, ...args).pipe(
    Command.workingDirectory(options.cwd),
    Command.env(options.env ?? {}),
  );
  const result = await Effect.runPromise(
    Command.string(command).pipe(
      Effect.map((stdout) => ({ exitCode: 0, stdout, stderr: "" })),
      Effect.catchAll((error) =>
        Effect.succeed({
          exitCode: 1,
          stdout: "",
          stderr: error instanceof Error ? error.message : String(error),
        }),
      ),
      Effect.provide(NodeContext.layer),
    ),
  );
  return result;
}

export function spawnFlakegraph(
  args: string[],
  options: {
    cwd: string;
    env: Record<string, string>;
    stdoutPath: string;
    stderrPath: string;
  },
): ChildProcess {
  const [binary, ...prefix] = appEnv().flakegraphCommand;
  const stdout = createWriteStream(options.stdoutPath, { flags: "w" });
  const stderr = createWriteStream(options.stderrPath, { flags: "w" });
  const child = spawn(binary, [...prefix, ...args], {
    cwd: options.cwd,
    env: options.env,
    stdio: ["ignore", "pipe", "pipe"],
    detached: true,
  });
  child.stdout?.pipe(stdout);
  child.stderr?.pipe(stderr);
  child.unref();
  return child;
}

export function lastJsonObject(text: string): Record<string, unknown> | null {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      const value = JSON.parse(lines[index]!) as unknown;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return value as Record<string, unknown>;
      }
    } catch {
      continue;
    }
  }
  const trimmed = text.trim();
  if (!trimmed) {
    return null;
  }
  try {
    const value = JSON.parse(trimmed) as unknown;
    return value && typeof value === "object" && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}
