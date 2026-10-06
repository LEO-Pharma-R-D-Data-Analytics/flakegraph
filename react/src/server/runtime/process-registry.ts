import type { ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";

export interface ManagedProcess {
  runId: string;
  graphId: string;
  process: ChildProcess;
  eventsPath: string;
  outputPath: string;
  stdoutPath: string;
  configPath: string;
  startedAt: string;
  storageKind: "local_files" | "snowflake";
  storageLocation: string;
}

class ProcessRegistry {
  private readonly processes = new Map<string, ManagedProcess>();

  start(managed: ManagedProcess): ManagedProcess {
    const existing = this.processes.get(managed.runId);
    if (existing && existing.process.exitCode === null && !existing.process.killed) {
      throw new Error(`Run is already active: ${managed.runId}`);
    }
    this.processes.set(managed.runId, managed);
    return managed;
  }

  get(runId: string): ManagedProcess | undefined {
    return this.processes.get(runId);
  }

  cancel(runId: string): ManagedProcess {
    const managed = this.processes.get(runId);
    if (!managed) {
      throw new Error(`Unknown local run: ${runId}`);
    }
    terminateProcess(managed.process);
    return managed;
  }

  forget(runId: string): void {
    const managed = this.processes.get(runId);
    if (managed && managed.process.exitCode === null) {
      terminateProcess(managed.process);
    }
    this.processes.delete(runId);
  }
}

export const processRegistry = new ProcessRegistry();

export function terminateProcess(process: ChildProcess): void {
  if (process.exitCode !== null || process.killed) {
    return;
  }
  try {
    if (process.pid) {
      process.kill("SIGTERM");
    }
  } catch {
    // The child may have already exited between the poll and the signal.
  }
}

export function pidIsRunning(pid: unknown): boolean {
  const value = typeof pid === "number" ? pid : Number(pid);
  if (!Number.isInteger(value) || value <= 0) {
    return false;
  }
  try {
    process.kill(value, 0);
    return true;
  } catch {
    return existsSync(`/proc/${value}`);
  }
}
