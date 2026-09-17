import { createHash } from "node:crypto";
import { readFile, stat, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";
import {
  type IngestionRequest,
  type ProviderSelection,
  writerProvider,
  storageLocation,
} from "./protocol/schema";
import { embeddingDimension } from "./providers";

const PARALLELISM_SETTINGS = [
  "extraction_parallelism",
  "resolution_parallelism",
  "community_report_parallelism",
  "description_merge_parallelism",
] as const;

const ENVIRONMENT_NAME = /^[A-Z_][A-Z0-9_]*$/;
const SENSITIVE_MARKERS = [
  "password",
  "token",
  "secret",
  "api_key",
  "private_key",
  "connection_string",
  "database_url",
];
const SENSITIVE_EXACT_KEYS = new Set(["dsn"]);
const NON_SECRET_REFERENCE_SUFFIXES = [
  "_environment_variable",
  "_path",
  "_name",
  "_header",
  "_prefix",
];
const ABSOLUTE_URL = /^[a-z][a-z0-9+.-]*:\/\//i;
const SAS_SIGNATURE = /[?&]sig=/i;
const SAS_QUERY_PARAMETER =
  /([?&](?:sig|se|sp|spr|sr|st|sv|skoid|sktid|skt|ske|sks|skv)=)[^&#\s]+/gi;
const SECRET_QUERY_PARAMETER =
  /([?&](?:access[_-]?key(?:[_-]?id)?|access[_-]?token|api[_-]?key|apikey|auth|auth[_-]?token|authorization|aws[_-]?access[_-]?key[_-]?id|client[_-]?secret|credential|key|passwd|password|private[_-]?token|secret|secret[_-]?access[_-]?key|session[_-]?token|sig|signature|subscription[_-]?key|token|x-amz-credential|x-amz-security-token|x-amz-signature)=)[^&#\s]+/gi;
const URL_PASSWORD = /(\b[a-z][a-z0-9+.-]*:\/\/[^\s:/@]*:)[^\s/@]+(@)/gi;
const REDACTED_VALUE = "${REDACTED}";

const baseConfigCache = new Map<string, { mtime: bigint; size: number; value: Record<string, unknown> }>();

export async function loadBaseConfig(file: string | null): Promise<Record<string, unknown>> {
  if (!file) {
    return {};
  }
  const resolved = path.resolve(file);
  const info = await stat(resolved);
  const cacheKey = resolved;
  const cached = baseConfigCache.get(cacheKey);
  if (cached && cached.mtime === info.mtimeNs && cached.size === info.size) {
    return structuredClone(cached.value);
  }
  const loaded = parseYaml(await readFile(resolved, "utf8")) ?? {};
  if (!loaded || typeof loaded !== "object" || Array.isArray(loaded)) {
    throw new Error("FlakeGraph configuration root must be a mapping");
  }
  const value = loaded as Record<string, unknown>;
  baseConfigCache.set(cacheKey, { mtime: info.mtimeNs, size: info.size, value });
  return structuredClone(value);
}

export async function buildRunConfig(request: IngestionRequest): Promise<Record<string, unknown>> {
  rejectLiteralSecrets(request.source, "source");
  const config = await loadBaseConfig(request.baseConfigPath);
  inlineOntologyProfile(config, ontologyProfilePath(config, request.baseConfigPath));
  rejectLiteralSecrets(config, "base configuration");
  const embedding = providerConfig(request.embedding);
  embedding.dimension = embeddingDimension(
    request.embedding.provider,
    request.embedding.model,
    request.embedding.dimension,
  );
  const overrides: Record<string, unknown> = {
    runtime: { runtime: runtimeValue(request) },
    job: {
      job_id: request.jobId,
      graph_id: request.graphId,
      use_file_queue: request.runtime === "snowflake",
    },
    files: { include_globs: request.includeGlobs },
    ocr: ocrProviderConfig(request.ocr),
    llm: providerConfig(request.llm),
    embedding,
    writer: {
      provider: writerProvider(request.output.kind),
      output_path: request.output.workspacePath,
    },
    cache: {
      provider: request.cacheProvider,
      path: path.join(path.dirname(request.output.workspacePath), "cache"),
    },
  };
  if (request.output.kind === "snowflake") {
    deepMerge(overrides, snowflakeOutputConfig(request));
  }
  deepMerge(overrides, { graph: providerParallelismSettings(request.providerParallelism) });
  if (request.runtime === "snowflake" && Object.keys(request.runtimeOptions).length > 0) {
    const runtimeOptions = Object.fromEntries(
      Object.entries(request.runtimeOptions).filter(
        ([, value]) => typeof value !== "string" || value.trim(),
      ),
    );
    if (Object.keys(runtimeOptions).length > 0) {
      deepMerge(overrides, { snowflake: runtimeOptions });
    }
  }
  if (request.ocr.provider === "generic_http") {
    overrides.generic_http_ocr = externalOcrConfig(request.ocr);
  }
  deepMerge(overrides, sourceConfig(request));
  deepMerge(config, overrides);
  sanitizeProviderSections(config, request);
  rejectLiteralSecrets(config);
  return config;
}

export async function writeRunConfig(request: IngestionRequest, destination: string): Promise<string> {
  await mkdir(path.dirname(destination), { recursive: true });
  const config = await buildRunConfig(request);
  await writeFile(destination, stringifyYaml(config, { sortMapEntries: false }), "utf8");
  return destination;
}

export function redactedConfig(config: Record<string, unknown>): Record<string, unknown> {
  const redact = (value: unknown, key = ""): unknown => {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return Object.fromEntries(
        Object.entries(value as Record<string, unknown>).map(([itemKey, item]) => [
          itemKey,
          redact(item, itemKey),
        ]),
      );
    }
    if (Array.isArray(value)) {
      return value.map((item) => redact(item, key));
    }
    if (sensitiveConfigKey(key)) {
      return value ? REDACTED_VALUE : value;
    }
    if (typeof value === "string") {
      return redactedUrl(value);
    }
    return value;
  };
  return redact(config) as Record<string, unknown>;
}

export function redactedUrl(value: string): string {
  if (!ABSOLUTE_URL.test(value)) {
    return value;
  }
  let redacted = SAS_SIGNATURE.test(value) ? value.replace(SAS_QUERY_PARAMETER, "$1***") : value;
  redacted = redacted.replace(SECRET_QUERY_PARAMETER, "$1***");
  return redacted.replace(URL_PASSWORD, "$1***$2");
}

export function sensitiveConfigKey(key: string): boolean {
  const normalized = key.trim().toLowerCase();
  if (NON_SECRET_REFERENCE_SUFFIXES.some((suffix) => normalized.endsWith(suffix))) {
    return false;
  }
  return SENSITIVE_EXACT_KEYS.has(normalized) || SENSITIVE_MARKERS.some((marker) => normalized.includes(marker));
}

export async function effectiveRequestFingerprint(
  request: IngestionRequest,
  effectiveConfig?: Record<string, unknown>,
): Promise<string> {
  const config = effectiveConfig ?? (await buildRunConfig(request));
  const identity = {
    effective_config: config,
    request: {
      runtime: request.runtime,
      job_id: request.jobId,
      graph_id: request.graphId,
      source_kind: request.sourceKind,
      source: request.source,
      base_config_path: request.baseConfigPath,
      include_globs: request.includeGlobs,
      cache_provider: request.cacheProvider,
      runtime_options: request.runtimeOptions,
      ocr: selectionIdentity(request.ocr),
      llm: selectionIdentity(request.llm),
      embedding: selectionIdentity(request.embedding),
      output_kind: request.output.kind,
      output_location: storageLocation(request.output),
    },
  };
  return createHash("sha256").update(stringifyYaml(identity, { sortMapEntries: true })).digest("hex");
}

export function environmentForRequest(
  request: IngestionRequest,
  base: NodeJS.ProcessEnv = process.env,
): Record<string, string> {
  const environment = Object.fromEntries(
    Object.entries(base).filter((entry): entry is [string, string] => typeof entry[1] === "string"),
  );
  for (const selection of [request.ocr, request.llm, request.embedding]) {
    const name = selection.apiKeyEnvironmentVariable;
    if (name && !(name in environment)) {
      throw new Error(`Credential environment variable is not set: ${name}`);
    }
  }
  const snowflakeVar = request.output.snowflake?.credentialEnvironmentVariable;
  if (snowflakeVar && !(snowflakeVar in environment)) {
    throw new Error(`Credential environment variable is not set: ${snowflakeVar}`);
  }
  return environment;
}

export function providerParallelismSettings(parallelism: number): Record<string, number> {
  const bounded = Math.max(1, Math.min(Math.trunc(parallelism), 64));
  return Object.fromEntries(PARALLELISM_SETTINGS.map((key) => [key, bounded]));
}

function sourceConfig(request: IngestionRequest): Record<string, unknown> {
  const source = request.source;
  if (request.sourceKind === "upload" || request.sourceKind === "local_path") {
    return { files: { source: "local", input_path: String(source.path ?? "") } };
  }
  if (request.sourceKind === "azure_blob") {
    return {
      files: { source: "azure_blob" },
      azure_blob: {
        account_url: source.accountUrl ?? source.account_url,
        container: source.container,
        prefix: source.prefix,
        download_path: path.join(path.dirname(request.output.workspacePath), "source-cache"),
      },
    };
  }
  if (request.sourceKind === "s3") {
    return {
      files: { source: "s3" },
      s3: {
        bucket: source.bucket,
        prefix: source.prefix,
        endpoint_url: source.endpointUrl ?? source.endpoint_url,
        region: source.region,
        download_path: path.join(path.dirname(request.output.workspacePath), "source-cache"),
      },
    };
  }
  if (request.sourceKind === "snowflake_stage") {
    return {
      files: { source: "snowflake_stage", stage_prefix: source.prefix },
      snowflake: { stage: source.stage },
    };
  }
  throw new Error(`Unsupported source kind: ${request.sourceKind}`);
}

function providerConfig(selection: ProviderSelection): Record<string, unknown> {
  const result: Record<string, unknown> = { provider: selection.provider, ...selection.options };
  if (selection.model) {
    result.model = selection.model;
  }
  if (selection.endpoint) {
    result.endpoint = selection.endpoint;
  }
  if (selection.apiKeyEnvironmentVariable) {
    const name = selection.apiKeyEnvironmentVariable.trim();
    if (!ENVIRONMENT_NAME.test(name)) {
      throw new Error(`Invalid credential environment variable name: ${name}`);
    }
    result.api_key = `\${${name}}`;
  }
  return result;
}

function ocrProviderConfig(selection: ProviderSelection): Record<string, unknown> {
  return providerConfig(selection);
}

function externalOcrConfig(selection: ProviderSelection): Record<string, unknown> {
  const config: Record<string, unknown> = {};
  if (selection.endpoint) {
    config.endpoint = selection.endpoint;
  }
  if (selection.apiKeyEnvironmentVariable) {
    config.api_key = `\${${selection.apiKeyEnvironmentVariable}}`;
  }
  return config;
}

function snowflakeOutputConfig(request: IngestionRequest): Record<string, unknown> {
  const snowflake = request.output.snowflake;
  if (!snowflake) {
    throw new Error("Snowflake destination requires account coordinates");
  }
  const connection: Record<string, unknown> = {
    account: snowflake.account,
    user: snowflake.user,
    database: snowflake.database,
    schema: snowflake.schema,
    warehouse: snowflake.warehouse,
    role: snowflake.role,
    host: snowflake.host,
    authenticator: snowflake.authenticator,
    bulk_stage: snowflake.bulkStage,
  };
  if (snowflake.credentialEnvironmentVariable) {
    connection[snowflake.credentialField || "password"] = `\${${snowflake.credentialEnvironmentVariable}}`;
  }
  return { snowflake: connection };
}

function sanitizeProviderSections(config: Record<string, unknown>, request: IngestionRequest): void {
  config.llm = providerSection(config.llm, providerConfig(request.llm), {
    allowed: new Set(
      request.llm.provider === "azure_openai" ? ["timeout_seconds", "api_version"] : ["timeout_seconds"],
    ),
  });
  const embedding = providerConfig(request.embedding);
  embedding.dimension = embeddingDimension(
    request.embedding.provider,
    request.embedding.model,
    request.embedding.dimension,
  );
  const embeddingAllowed = new Set(["batch_size"]);
  if (request.embedding.provider === "sentence_transformers") {
    embeddingAllowed.add("device");
  }
  if (request.embedding.provider === "azure_openai") {
    embeddingAllowed.add("api_version");
  }
  config.embedding = providerSection(config.embedding, embedding, { allowed: embeddingAllowed });
  const ocrAllowed = new Set(["language", "page_range", "timeout_seconds"]);
  const prefixes: string[] = [];
  if (request.ocr.provider === "fallback") {
    prefixes.push("fallback_", "mineru_", "tesseract_");
    ocrAllowed.add("model_cache_dir");
  } else if (request.ocr.provider === "mineru_internal") {
    prefixes.push("mineru_");
    ocrAllowed.add("model_cache_dir");
  } else if (request.ocr.provider === "tesseract_internal") {
    prefixes.push("tesseract_");
  } else if (request.ocr.provider === "snowflake_cortex") {
    prefixes.push("snowflake_");
  }
  config.ocr = providerSection(config.ocr, ocrProviderConfig(request.ocr), {
    allowed: ocrAllowed,
    prefixes,
    excluded: new Set(["mineru_api_url", "mineru_api_key"]),
  });
  if (request.ocr.provider === "generic_http") {
    const existing = asRecord(config.generic_http_ocr);
    const { endpoint: _endpoint, api_key: _apiKey, ...neutral } = existing;
    config.generic_http_ocr = { ...neutral, ...externalOcrConfig(request.ocr) };
  } else {
    delete config.generic_http_ocr;
  }
}

function providerSection(
  existing: unknown,
  selected: Record<string, unknown>,
  options: { allowed: Set<string>; prefixes?: string[]; excluded?: Set<string> },
): Record<string, unknown> {
  const blocked = options.excluded ?? new Set<string>();
  const retained: Record<string, unknown> = {};
  const current = asRecord(existing);
  for (const [key, value] of Object.entries(current)) {
    if (blocked.has(key)) {
      continue;
    }
    if (options.allowed.has(key) || options.prefixes?.some((prefix) => key.startsWith(prefix))) {
      retained[key] = value;
    }
  }
  return { ...retained, ...selected };
}

function runtimeValue(request: IngestionRequest): string {
  if (request.runtime === "kubernetes") {
    return "kubernetes";
  }
  if (request.runtime === "snowflake") {
    return "snowflake";
  }
  return "local";
}

function ontologyProfilePath(config: Record<string, unknown>, baseConfigPath: string | null): string | null {
  const ontology = asRecord(config.ontology);
  const profilePath = ontology.profile_path;
  if (typeof profilePath !== "string" || !profilePath) {
    return null;
  }
  if (path.isAbsolute(profilePath) || !baseConfigPath) {
    return profilePath;
  }
  return path.resolve(path.dirname(baseConfigPath), profilePath);
}

function inlineOntologyProfile(config: Record<string, unknown>, profilePath: string | null): void {
  if (!profilePath) {
    return;
  }
  const ontology = asRecord(config.ontology);
  if (ontology.profile && typeof ontology.profile === "object") {
    return;
  }
  ontology.profile_path = profilePath;
  config.ontology = ontology;
}

function rejectLiteralSecrets(value: unknown, pathLabel = "configuration"): void {
  const walk = (item: unknown, current: string): void => {
    if (Array.isArray(item)) {
      item.forEach((entry, index) => walk(entry, `${current}[${index}]`));
      return;
    }
    if (!item || typeof item !== "object") {
      if (typeof item === "string" && looksLikeLiteralSecret(current, item)) {
        throw new Error(`Refusing to persist a literal secret at ${current}`);
      }
      return;
    }
    for (const [key, nested] of Object.entries(item as Record<string, unknown>)) {
      walk(nested, `${current}.${key}`);
    }
  };
  walk(value, pathLabel);
}

function looksLikeLiteralSecret(key: string, value: string): boolean {
  const last = key.split(".").pop() ?? key;
  if (!sensitiveConfigKey(last)) {
    return false;
  }
  if (value.startsWith("${") && value.endsWith("}")) {
    return false;
  }
  return value.trim().length > 0;
}

function selectionIdentity(selection: ProviderSelection): Record<string, unknown> {
  return {
    provider: selection.provider,
    model: selection.model,
    endpoint: selection.endpoint,
    api_key_environment_variable: selection.apiKeyEnvironmentVariable,
    dimension: selection.dimension,
    options: selection.options,
  };
}

function deepMerge(target: Record<string, unknown>, source: Record<string, unknown>): Record<string, unknown> {
  for (const [key, value] of Object.entries(source)) {
    const existing = target[key];
    if (isPlainObject(existing) && isPlainObject(value)) {
      target[key] = deepMerge({ ...existing }, value);
    } else {
      target[key] = value;
    }
  }
  return target;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function asRecord(value: unknown): Record<string, unknown> {
  return isPlainObject(value) ? { ...value } : {};
}
