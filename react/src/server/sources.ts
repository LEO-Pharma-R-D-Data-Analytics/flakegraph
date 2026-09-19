import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readdir, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { stringify as stringifyYaml } from "yaml";
import { runFlakegraph } from "./cli";
import { sourceSettings } from "./config";
import type { SourceKind, SourceObject } from "./protocol/schema";
import { SUPPORTED_SUFFIXES } from "./providers";

/**
 * List a bucket or container the way the pipeline itself would read it.
 *
 * The console has no object-storage client of its own; it writes the same
 * `files` settings a run would use and asks `flakegraph sources list` to name
 * what is there. The listing therefore matches what the run will ingest, and
 * credentials stay where the pipeline already finds them.
 */
export async function listRemoteObjects(
  sourceKind: SourceKind,
  source: Record<string, unknown>,
  options: { cwd: string; stateRoot: string; env?: Record<string, string>; limit?: number },
): Promise<SourceObject[]> {
  const directory = path.join(options.stateRoot, "browse");
  await mkdir(directory, { recursive: true });
  const configPath = path.join(directory, `${randomUUID()}.yaml`);
  await writeFile(configPath, stringifyYaml(sourceSettings(sourceKind, source)), "utf8");
  try {
    const result = await runFlakegraph(
      ["sources", "list", "--config", configPath, "--limit", String(options.limit ?? 1_000)],
      { cwd: options.cwd, env: options.env ?? {} },
    );
    if (result.exitCode !== 0) {
      throw new Error(result.stderr.trim() || result.stdout.trim() || `Listing the ${sourceKind} source failed`);
    }
    return parseListing(result.stdout);
  } finally {
    await rm(configPath, { force: true });
  }
}

function parseListing(stdout: string): SourceObject[] {
  const start = stdout.indexOf("[");
  const end = stdout.lastIndexOf("]");
  if (start < 0 || end < start) {
    throw new Error("Source listing returned no JSON array");
  }
  const rows = JSON.parse(stdout.slice(start, end + 1)) as Array<Record<string, unknown>>;
  return rows.map((row) => ({
    uri: String(row.uri ?? ""),
    name: String(row.name ?? ""),
    sizeBytes: typeof row.size_bytes === "number" ? row.size_bytes : Number(row.size_bytes ?? 0),
    modifiedAt: typeof row.modified_at === "string" ? row.modified_at : null,
    checksum: typeof row.checksum === "string" ? row.checksum : null,
  }));
}

export async function listLocalObjects(inputPath: string, limit = 1_000): Promise<SourceObject[]> {
  if (!existsSync(inputPath)) {
    throw new Error(`Input path does not exist: ${inputPath}`);
  }
  const objects: SourceObject[] = [];
  for await (const file of localFiles(inputPath)) {
    if (!SUPPORTED_SUFFIXES.has(path.extname(file).toLowerCase())) {
      continue;
    }
    const info = await stat(file);
    objects.push({
      uri: pathToFileUrl(file),
      name: path.basename(file),
      sizeBytes: info.size,
      modifiedAt: info.mtime.toISOString(),
      checksum: null,
    });
    if (objects.length >= limit) {
      break;
    }
  }
  return objects;
}

async function* localFiles(inputPath: string): AsyncGenerator<string> {
  const info = await stat(inputPath);
  if (info.isFile()) {
    yield inputPath;
    return;
  }
  const stack = [inputPath];
  while (stack.length > 0) {
    const directory = stack.pop()!;
    const entries = (await readdir(directory, { withFileTypes: true })).sort((left, right) =>
      left.name.localeCompare(right.name),
    );
    for (const entry of entries) {
      const candidate = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        stack.push(candidate);
      } else if (entry.isFile()) {
        yield candidate;
      }
    }
  }
}

function pathToFileUrl(file: string): string {
  const resolved = path.resolve(file);
  return `file://${resolved.split(path.sep).map(encodeURIComponent).join("/")}`;
}
