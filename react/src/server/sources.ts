import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import { existsSync } from "node:fs";
import type { SourceObject } from "./protocol/schema";
import { SUPPORTED_SUFFIXES } from "./providers";

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
