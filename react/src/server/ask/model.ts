import "server-only";

import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { createAzure } from "@ai-sdk/azure";
import { createOpenAI } from "@ai-sdk/openai";
import type { LanguageModel } from "ai";
import {
  DEFAULT_AZURE_API_VERSION,
  DEFAULT_AZURE_DEPLOYMENT,
  DEFAULT_OLLAMA_BASE_URL,
  DEFAULT_OLLAMA_MODEL,
  HUB_SECRETS_CANDIDATES,
} from "./constants";

export interface AskModel {
  languageModel: LanguageModel;
  provider: "azure" | "openai" | "ollama";
  modelId: string;
  billed: boolean;
}

let cached: AskModel | null | undefined;
// Whether the resolved model answered a probe. A local engine is assumed
// from its address alone, and one that is not running would otherwise be
// "configured" and fail every call; the probe settles it once per process.
let probed: Promise<void> | null = null;

export function resetAskModel(): void {
  cached = undefined;
  probed = null;
}

export function resolveAskModel(): AskModel | null {
  if (cached !== undefined) {
    return cached;
  }
  cached = loadAskModel();
  return cached;
}

/**
 * Resolve the model and confirm a local engine is actually there.
 *
 * Every request entry point awaits this before the synchronous readers run;
 * a hosted provider is taken at its word, an Ollama address is asked for its
 * model list with a short timeout and dropped when it does not answer.
 */
export async function probeAskModel(): Promise<AskModel | null> {
  const model = resolveAskModel();
  if (!model || model.provider !== "ollama") {
    return model;
  }
  probed ??= (async () => {
    const base = ollamaBaseUrl();
    try {
      const response = await fetch(`${base}/api/tags`, { signal: AbortSignal.timeout(OLLAMA_PROBE_TIMEOUT_MS) });
      if (!response.ok) {
        cached = null;
      }
    } catch {
      cached = null;
    }
  })();
  await probed;
  return cached ?? null;
}

const OLLAMA_PROBE_TIMEOUT_MS = 1_500;

function ollamaBaseUrl(): string {
  const env = { ...loadOptionalSecretFile(), ...process.env };
  const base = first(env, ["FLAKEGRAPH_ASK_OLLAMA_BASE_URL", "OLLAMA_HOST"]) ?? DEFAULT_OLLAMA_BASE_URL;
  return base.replace(/\/$/, "").replace(/\/v1$/, "");
}

export function askModelConfigured(): boolean {
  return resolveAskModel() != null;
}

function loadAskModel(): AskModel | null {
  const env = { ...loadOptionalSecretFile(), ...process.env };
  const askKey = first(env, [
    "FLAKEGRAPH_ASK_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AISERVICES_API_KEY",
    "OPENAI_API_KEY",
  ]);
  const askBase = first(env, ["FLAKEGRAPH_ASK_BASE_URL", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_BASE_URL"]);
  const askModel = first(env, ["FLAKEGRAPH_ASK_MODEL", "AZURE_OPENAI_DEPLOYMENT"]) ?? DEFAULT_AZURE_DEPLOYMENT;
  const apiVersion = first(env, ["FLAKEGRAPH_ASK_API_VERSION", "AZURE_OPENAI_API_VERSION"]) ?? DEFAULT_AZURE_API_VERSION;

  if (askKey && askBase && isAzureHost(askBase)) {
    const baseURL = azureOpenAiBase(askBase);
    const azure = createAzure({
      baseURL,
      apiKey: askKey,
      apiVersion,
      useDeploymentBasedUrls: true,
    });
    return {
      languageModel: azure.chat(askModel),
      provider: "azure",
      modelId: askModel,
      billed: true,
    };
  }

  if (askKey && askBase && !isAzureHost(askBase)) {
    const openai = createOpenAI({ baseURL: askBase.replace(/\/$/, ""), apiKey: askKey });
    return {
      languageModel: openai.chat(askModel),
      provider: "openai",
      modelId: askModel,
      billed: true,
    };
  }

  const ollamaBase = first(env, ["FLAKEGRAPH_ASK_OLLAMA_BASE_URL", "OLLAMA_HOST"]) ?? DEFAULT_OLLAMA_BASE_URL;
  const ollamaModel = first(env, ["FLAKEGRAPH_ASK_OLLAMA_MODEL"]) ?? DEFAULT_OLLAMA_MODEL;
  if (ollamaReachable(ollamaBase)) {
    const openai = createOpenAI({
      baseURL: ollamaBase.replace(/\/$/, "").replace(/\/v1$/, "") + "/v1",
      apiKey: "ollama",
    });
    return {
      languageModel: openai.chat(ollamaModel),
      provider: "ollama",
      modelId: ollamaModel,
      billed: false,
    };
  }

  return null;
}

function isAzureHost(url: string): boolean {
  return /cognitiveservices\.azure\.com|openai\.azure\.com|services\.ai\.azure\.com/i.test(url);
}

function azureOpenAiBase(url: string): string {
  const trimmed = url.replace(/\/$/, "");
  if (trimmed.endsWith("/openai/v1")) {
    return trimmed.replace(/\/v1$/, "");
  }
  if (trimmed.endsWith("/openai")) {
    return trimmed;
  }
  return `${trimmed}/openai`;
}

function ollamaReachable(base: string): boolean {
  if (process.env.FLAKEGRAPH_ASK_DISABLE_OLLAMA === "1") {
    return false;
  }
  try {
    const url = new URL(base.includes("://") ? base : `http://${base}`);
    return url.hostname === "127.0.0.1" || url.hostname === "localhost";
  } catch {
    return false;
  }
}

function first(env: Record<string, string | undefined>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = env[key]?.trim();
    if (value && !value.startsWith("placeholder")) {
      return value;
    }
  }
  return undefined;
}

function loadOptionalSecretFile(): Record<string, string> {
  if (process.env.FLAKEGRAPH_ASK_IGNORE_HUB_SECRETS === "1") {
    return {};
  }
  const configured = process.env.FLAKEGRAPH_ASK_SECRETS_FILE?.trim();
  const candidates = [
    configured,
    ...HUB_SECRETS_CANDIDATES,
    path.resolve(process.cwd(), "../hub-app/.env.secrets"),
    path.resolve(process.cwd(), "../../hub-app/.env.secrets"),
  ].filter((item): item is string => Boolean(item));
  for (const file of candidates) {
    if (!existsSync(file)) {
      continue;
    }
    return parseEnvFile(readFileSync(file, "utf8"));
  }
  return {};
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
    if (key) {
      env[key] = value;
    }
  }
  return env;
}
