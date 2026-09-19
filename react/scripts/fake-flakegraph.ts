#!/usr/bin/env bun
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { parse as parseYaml } from "yaml";
import { goldToDataset, writeDataset } from "../src/server/gold";
import type { GoldGraph } from "../src/server/gold";

const [, , command, ...rest] = process.argv;

async function main() {
  if (command === "preflight") {
    await preflight(argValue("--config"));
    return;
  }
  if (command === "worker") {
    await worker(argValue("--config"));
    return;
  }
  if (command === "distributed") {
    await distributed(rest[0] ?? "", argValue("--config"), argValue("--run-id"));
    return;
  }
  if (command === "sources" && rest[0] === "list") {
    await sourcesList(argValue("--config"), Number(argValue("--limit") || 1000));
    return;
  }
  console.error(`Unknown flakegraph command: ${command}`);
  process.exit(1);
}

function argValue(flag: string): string {
  const index = process.argv.indexOf(flag);
  return index >= 0 ? process.argv[index + 1] ?? "" : "";
}

async function preflight(configPath: string) {
  const config = await readConfig(configPath);
  const inputPath = String((config.files as Record<string, unknown> | undefined)?.input_path ?? "");
  if (inputPath.includes("missing-corpus")) {
    process.stdout.write(`${JSON.stringify({ ok: false, errors: ["Input path does not exist"] })}\n`);
    process.exit(1);
  }
  process.stdout.write(
    `${JSON.stringify({ ok: true, errors: [], warnings: [], checks: [{ name: "source", ok: true }] })}\n`,
  );
}

// A bucket listing shaped like the real command's: the fake bucket holds
// three objects under the requested prefix, one of them unsupported.
async function sourcesList(configPath: string, limit: number) {
  const config = await readConfig(configPath);
  const kind = String((config.files as Record<string, unknown> | undefined)?.source ?? "");
  if (kind !== "s3" && kind !== "azure_blob") {
    process.stderr.write(`${kind} sources cannot be listed without fetching\n`);
    process.exit(2);
  }
  const backend = (config[kind] ?? {}) as Record<string, unknown>;
  const container = String(backend.bucket ?? backend.container ?? "");
  if (!container) {
    process.stderr.write(`${kind} file source requires a bucket\n`);
    process.exit(1);
  }
  if (container === "slow") {
    // A credential chain that never answers.
    await new Promise((resolve) => setTimeout(resolve, 60_000));
  }
  if (container === "missing") {
    process.stderr.write("NoSuchBucket: The specified bucket does not exist\n");
    process.exit(1);
  }
  const prefix = String(backend.prefix ?? "");
  const scheme = kind === "s3" ? "s3" : "az";
  const rows = [
    { name: "judo.md", size_bytes: 512 },
    { name: "karate.pdf", size_bytes: 40_960 },
  ].map((row) => ({
    uri: `${scheme}://${container}/${prefix}${row.name}`,
    name: row.name,
    size_bytes: row.size_bytes,
    modified_at: "2026-09-18T10:00:00+00:00",
    checksum: "etag-" + row.name,
  }));
  process.stdout.write(`${JSON.stringify(rows.slice(0, limit), null, 2)}\n`);
}

async function worker(configPath: string) {
  const config = await readConfig(configPath);
  const outputPath = String((config.writer as Record<string, unknown> | undefined)?.output_path ?? "");
  const graphId = String((config.job as Record<string, unknown> | undefined)?.graph_id ?? "graph");
  const inputPath = String((config.files as Record<string, unknown> | undefined)?.input_path ?? "");
  const fail = inputPath.includes("fail-worker") || process.env.FAKE_FLAKEGRAPH_FAIL === "1";
  const delay = Number(process.env.FAKE_FLAKEGRAPH_DELAY_MS ?? 150);
  const files = inputPath.includes("deep_learning") ? 8 : 10;
  emit("file_source", "started", { files_seen: files });
  await sleep(delay);
  emit("file_source", "completed", { files_seen: files, files_total: files });
  for (let index = 0; index < Math.min(files, 3); index += 1) {
    emit("ocr", "running", { files_total: files }, `doc_${index}`);
    await sleep(delay);
    emit("ocr", "completed", { files_total: files }, `doc_${index}`);
  }
  if (fail) {
    emit("graph_extraction", "failed", {}, null, "Simulated worker failure");
    process.exit(1);
  }
  emit("write", "completed", { files_total: files });
  await mkdir(outputPath, { recursive: true });
  const repo = process.env.FLAKEGRAPH_REPOSITORY_ROOT || process.cwd();
  const goldPath = inputPath.includes("deep_learning")
    ? path.resolve(repo, "data/deep_learning_papers/gold.json")
    : path.resolve(repo, "data/martial_arts/gold.json");
  try {
    const gold = JSON.parse(await readFile(goldPath, "utf8")) as GoldGraph;
    await writeDataset(outputPath, goldToDataset(gold, graphId));
  } catch {
    await writeDataset(outputPath, goldToDataset({ name: graphId, entities: [{ id: "n1", name: "Node", type: "THING" }], relations: [] }, graphId));
  }
}

async function distributed(sub: string, configPath: string, runId: string) {
  const id = runId || "distributed-run";
  if (sub === "submit") {
    process.stdout.write(`${JSON.stringify({ id, status: "queued", config: configPath })}\n`);
    return;
  }
  if (sub === "status") {
    process.stdout.write(`${JSON.stringify({ id, status: "running" })}\n`);
    return;
  }
  if (sub === "cancel") {
    process.stdout.write(`${JSON.stringify({ id, status: "cancelled" })}\n`);
    return;
  }
  if (sub === "retry") {
    process.stdout.write(`${JSON.stringify({ id, status: "queued" })}\n`);
    return;
  }
  process.stdout.write(`${JSON.stringify({ ok: true, sub })}\n`);
}

async function readConfig(configPath: string): Promise<Record<string, unknown>> {
  const loaded = parseYaml(await readFile(configPath, "utf8")) ?? {};
  return loaded as Record<string, unknown>;
}

function emit(
  stage: string,
  status: string,
  counts: Record<string, unknown> = {},
  fileId: string | null = null,
  message: string | null = null,
) {
  const record = {
    event: "kg_processor.progress",
    timestamp: new Date().toISOString(),
    stage,
    status,
    file_id: fileId,
    message,
    elapsed_ms: 25,
    counts,
    metadata: {},
  };
  process.stderr.write(`${JSON.stringify(record)}\n`);
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

await main();
