import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { Command } from "@effect/platform";
import { appEnv } from "./env";

export interface CommandResult {
  exitCode: number | null;
  stdout: string;
  stderr: string;
}

export function flakegraphCommand(args: string[]): Command.Command {
  const [binary, ...prefix] = appEnv().flakegraphCommand;
  return Command.make(binary, ...prefix, ...args);
}

export async function runFlakegraph(
  args: string[],
  options: { cwd: string; env?: Record<string, string> },
): Promise<CommandResult> {
  const command = flakegraphCommand(args).pipe(
    Command.workingDirectory(options.cwd),
    Command.env(options.env ?? {}),
  );
  if (command._tag !== "StandardCommand") {
    throw new Error("Expected a standard flakegraph command");
  }
  return await new Promise((resolve) => {
    const child = spawn(command.command, [...command.args], {
      cwd: options.cwd,
      env: { ...process.env, ...(options.env ?? {}) },
    });
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      resolve({ exitCode: 1, stdout, stderr: error.message });
    });
    child.on("close", (code) => {
      resolve({ exitCode: code, stdout, stderr });
    });
  });
}

export function spawnFlakegraph(
  args: string[],
  options: {
    cwd: string;
    env: Record<string, string>;
    stdoutPath: string;
    stderrPath: string;
  },
) {
  const command = flakegraphCommand(args);
  if (command._tag !== "StandardCommand") {
    throw new Error("Expected a standard flakegraph command");
  }
  const stdout = createWriteStream(options.stdoutPath, { flags: "w" });
  const stderr = createWriteStream(options.stderrPath, { flags: "w" });
  const child = spawn(command.command, [...command.args], {
    cwd: options.cwd,
    env: { ...process.env, ...options.env },
    stdio: ["ignore", "pipe", "pipe"],
    detached: true,
  });
  child.stdout?.pipe(stdout);
  child.stderr?.pipe(stderr);
  child.unref();
  return child;
}

export function lastJsonObject(text: string): Record<string, unknown> | null {
  const candidates = [
    ...text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .reverse(),
    text.trim(),
  ];
  for (const candidate of candidates) {
    try {
      const value = JSON.parse(candidate) as unknown;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return value as Record<string, unknown>;
      }
    } catch {
      continue;
    }
  }
  return null;
}
