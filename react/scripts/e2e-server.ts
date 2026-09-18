#!/usr/bin/env bun
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";

const port = process.env.PORT ?? "3100";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repo = path.resolve(root, "..");
const stateRoot = process.env.FLAKEGRAPH_APP_STATE_ROOT ?? path.join(tmpdir(), "flakegraph-e2e");

// FLAKEGRAPH_ASK_NO_MODEL runs the journeys the way CI does: no hosted
// credentials picked up from a developer checkout, and no local engine.
if (process.env.FLAKEGRAPH_ASK_NO_MODEL === "1") {
  process.env.FLAKEGRAPH_ASK_DISABLE_OLLAMA = "1";
  process.env.FLAKEGRAPH_ASK_IGNORE_HUB_SECRETS = "1";
} else {
  applyAskSecrets(root, repo);
}

if (!process.env.FLAKEGRAPH_KEEP_STATE) {
  await rm(stateRoot, { recursive: true, force: true });
}

const seed = spawn("bun", ["scripts/seed.ts"], {
  cwd: root,
  stdio: "inherit",
  env: {
    ...process.env,
    FLAKEGRAPH_APP_STATE_ROOT: stateRoot,
    FLAKEGRAPH_REPOSITORY_ROOT: repo,
  },
});
const seedCode = await new Promise<number>((resolve) => seed.on("close", (code) => resolve(code ?? 1)));
if (seedCode !== 0) {
  process.exit(seedCode);
}

const child = spawn("bun", ["run", "dev", "--port", port, "--hostname", "127.0.0.1"], {
  cwd: root,
  stdio: "inherit",
  env: {
    ...process.env,
    PORT: port,
    FLAKEGRAPH_APP_STATE_ROOT: stateRoot,
    FLAKEGRAPH_REPOSITORY_ROOT: repo,
    FLAKEGRAPH_STUB_RUNTIMES: "1",
    FLAKEGRAPH_CLI: `bun ${path.join(root, "scripts/fake-flakegraph.ts")}`,
  },
});

child.on("exit", (code) => process.exit(code ?? 0));

function applyAskSecrets(reactRoot: string, repoRoot: string) {
  const candidates = [
    process.env.FLAKEGRAPH_ASK_SECRETS_FILE,
    path.resolve(reactRoot, "../../hub-app/.env.secrets"),
    path.resolve(repoRoot, "../hub-app/.env.secrets"),
    "/Users/mathiasgruber/Documents/github/hub-app/.env.secrets",
  ].filter((item): item is string => Boolean(item));
  for (const file of candidates) {
    if (!existsSync(file)) {
      continue;
    }
    const parsed = parseEnvFile(readFileSync(file, "utf8"));
    for (const [key, value] of Object.entries(parsed)) {
      if (!process.env[key]) {
        process.env[key] = value;
      }
    }
    process.env.FLAKEGRAPH_ASK_SECRETS_FILE ??= file;
    process.env.FLAKEGRAPH_ASK_API_KEY ??= parsed.OPENAI_API_KEY ?? parsed.AISERVICES_API_KEY;
    process.env.FLAKEGRAPH_ASK_BASE_URL ??= parsed.OPENAI_API_BASE_URL ?? parsed.AZURE_OPENAI_ENDPOINT;
    process.env.FLAKEGRAPH_ASK_MODEL ??= parsed.AZURE_OPENAI_DEPLOYMENT ?? "gpt-4.1-mini-2025-04-14";
    process.env.FLAKEGRAPH_ASK_API_VERSION ??= parsed.AZURE_OPENAI_API_VERSION ?? "2024-12-01-preview";
    return;
  }
}

function parseEnvFile(contents: string): Record<string, string> {
  const env: Record<string, string> = {};
  for (const line of contents.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || !trimmed.includes("=")) {
      continue;
    }
    const index = trimmed.indexOf("=");
    const key = trimmed.slice(0, index).replace(/^export\s+/, "").trim();
    const value = trimmed.slice(index + 1).trim().replace(/^['"]|['"]$/g, "");
    if (key && value && !value.startsWith("placeholder")) {
      env[key] = value;
    }
  }
  return env;
}
