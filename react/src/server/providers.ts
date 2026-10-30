import type { ProviderSelection } from "./protocol/schema";

/** The form uses each deployed transport's endpoint, including separate OCR contracts. */
export function fleetSelection(
  profile: { config: Record<string, unknown>; parsingEndpoint?: string | null },
  section: "ocr" | "llm" | "embedding",
  fallback: ProviderSelection,
): ProviderSelection {
  const record = profile.config[section];
  if (!record || typeof record !== "object" || Array.isArray(record)) return fallback;
  const values = record as Record<string, unknown>;
  if (typeof values.provider !== "string" || !values.provider) return fallback;
  const generic = profile.config.generic_http_ocr as Record<string, unknown> | undefined;
  const endpoint = section === "ocr" && values.provider === "generic_http" ? generic?.endpoint :
    section === "ocr" && values.provider === "mineru_api" ? profile.parsingEndpoint ?? values.mineru_api_url : values.endpoint;
  return {
    ...providerSelection(section, values.provider),
    model: typeof values.model === "string" ? values.model : null,
    endpoint: typeof endpoint === "string" && !endpoint.includes("${") ? endpoint : null,
    dimension: typeof values.dimension === "number" ? values.dimension : null,
  };
}

export interface ProviderOption {
  name: string;
  label: string;
  needsModel?: boolean;
  needsEndpoint?: boolean;
  needsApiKey?: boolean;
  defaultModel?: string | null;
  defaultEndpoint?: string | null;
  defaultDimension?: number | null;
}

/**
 * The first entry of each list is the laptop default: what a fresh checkout
 * can run with nothing installed beyond the pipeline itself. Built-in text
 * extraction needs no OCR engine; Ollama serves a small model on any
 * machine; the sentence-transformers model downloads on first use. A fleet
 * overrides all three through its profile, so these never reach a cluster.
 */
export const OCR_PROVIDERS: ProviderOption[] = [
  { name: "builtin_text", label: "Built-in document text only" },
  { name: "fallback", label: "Adaptive layout (native text, then MinerU)" },
  { name: "mineru_internal", label: "MinerU (inside worker)" },
  { name: "mineru_api", label: "MinerU API", needsEndpoint: true, needsApiKey: true },
  { name: "tesseract_internal", label: "Tesseract (inside worker)" },
  { name: "generic_http", label: "Generic HTTP OCR", needsEndpoint: true, needsApiKey: true },
  { name: "snowflake_cortex", label: "Snowflake Cortex Parse Document" },
];

export const LLM_PROVIDERS: ProviderOption[] = [
  {
    name: "ollama",
    label: "Ollama",
    needsModel: true,
    needsEndpoint: true,
    defaultModel: "qwen3:4b",
    defaultEndpoint: "http://localhost:11434",
  },
  {
    name: "vllm_local",
    label: "vLLM",
    needsModel: true,
    needsEndpoint: true,
    defaultModel: "Qwen/Qwen3-8B",
    defaultEndpoint: "http://localhost:8000/v1",
  },
  {
    name: "openai_compatible",
    label: "OpenAI-compatible API",
    needsModel: true,
    needsEndpoint: true,
    needsApiKey: true,
  },
  {
    name: "azure_openai",
    label: "Azure OpenAI",
    needsModel: true,
    needsEndpoint: true,
    needsApiKey: true,
  },
  { name: "snowflake_cortex", label: "Snowflake Cortex", needsModel: true },
];

export const EMBEDDING_PROVIDERS: ProviderOption[] = [
  {
    name: "sentence_transformers",
    label: "Sentence Transformers",
    needsModel: true,
    defaultModel: "Qwen/Qwen3-Embedding-0.6B",
    defaultDimension: 1024,
  },
  {
    name: "openai_compatible",
    label: "OpenAI-compatible API",
    needsModel: true,
    needsEndpoint: true,
    needsApiKey: true,
    defaultDimension: 1536,
  },
  {
    name: "azure_openai",
    label: "Azure OpenAI",
    needsModel: true,
    needsEndpoint: true,
    needsApiKey: true,
    defaultDimension: 1536,
  },
  {
    name: "snowflake_cortex",
    label: "Snowflake Cortex",
    needsModel: true,
    defaultDimension: 768,
  },
];

export const CORTEX_EMBEDDING_MODELS_BY_WIDTH: Record<number, string> = {
  768: "snowflake-arctic-embed-m",
  1024: "snowflake-arctic-embed-l-v2.0",
};

export const SUPPORTED_SUFFIXES = new Set([
  ".pdf",
  ".txt",
  ".md",
  ".markdown",
  ".html",
  ".htm",
  ".docx",
  ".png",
  ".jpg",
  ".jpeg",
  ".tif",
  ".tiff",
  ".bmp",
  ".webp",
  ".pptx",
  ".xlsx",
]);

export function embeddingDimension(
  provider: string,
  model: string | null,
  dimension: number | null,
): number {
  if (dimension && dimension > 0) {
    return dimension;
  }
  const option = EMBEDDING_PROVIDERS.find((item) => item.name === provider);
  if (option?.defaultDimension) {
    return option.defaultDimension;
  }
  if (provider === "snowflake_cortex" && model) {
    const match = Object.entries(CORTEX_EMBEDDING_MODELS_BY_WIDTH).find(([, name]) => name === model);
    if (match) {
      return Number(match[0]);
    }
  }
  return 1024;
}

export function selectionFromOption(option: ProviderOption): ProviderSelection {
  return {
    provider: option.name,
    model: option.needsModel ? option.defaultModel ?? null : null,
    endpoint: option.needsEndpoint ? option.defaultEndpoint ?? null : null,
    apiKeyEnvironmentVariable: null,
    dimension: option.defaultDimension ?? null,
    options: {},
  };
}

export function providerSelection(kind: "ocr" | "llm" | "embedding", name: string): ProviderSelection {
  const options = kind === "ocr" ? OCR_PROVIDERS : kind === "llm" ? LLM_PROVIDERS : EMBEDDING_PROVIDERS;
  const option = options.find((item) => item.name === name);
  return option ? selectionFromOption(option) : defaultProvider(kind);
}

export function defaultProvider(kind: "ocr" | "llm" | "embedding"): ProviderSelection {
  if (kind === "ocr") {
    return selectionFromOption(OCR_PROVIDERS[0]);
  }
  if (kind === "llm") {
    return selectionFromOption(LLM_PROVIDERS[0]);
  }
  return selectionFromOption(EMBEDDING_PROVIDERS[0]);
}
