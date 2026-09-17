export interface ProviderOption {
  name: string;
  label: string;
  needsModel?: boolean;
  needsEndpoint?: boolean;
  needsApiKey?: boolean;
  defaultDimension?: number | null;
}

export const OCR_PROVIDERS: ProviderOption[] = [
  { name: "fallback", label: "Adaptive text + MinerU" },
  { name: "builtin_text", label: "Built-in document text" },
  { name: "mineru_internal", label: "MinerU (inside worker)" },
  { name: "mineru_api", label: "MinerU API", needsEndpoint: true, needsApiKey: true },
  { name: "tesseract_internal", label: "Tesseract (inside worker)" },
  { name: "generic_http", label: "Generic HTTP OCR", needsEndpoint: true, needsApiKey: true },
  { name: "snowflake_cortex", label: "Snowflake Cortex Parse Document" },
];

export const LLM_PROVIDERS: ProviderOption[] = [
  { name: "ollama", label: "Ollama", needsModel: true, needsEndpoint: true },
  { name: "vllm_local", label: "vLLM", needsModel: true, needsEndpoint: true },
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
    defaultDimension: 384,
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
  return 384;
}

export function defaultProvider(kind: "ocr" | "llm" | "embedding") {
  if (kind === "ocr") {
    return { provider: "fallback", model: null, endpoint: null, apiKeyEnvironmentVariable: null, dimension: null, options: {} };
  }
  if (kind === "llm") {
    return {
      provider: "vllm_local",
      model: "unsloth/Qwen3.8-27B-NVFP4",
      endpoint: "http://localhost:8000/v1",
      apiKeyEnvironmentVariable: null,
      dimension: null,
      options: {},
    };
  }
  return {
    provider: "sentence_transformers",
    model: "sentence-transformers/all-MiniLM-L6-v2",
    endpoint: null,
    apiKeyEnvironmentVariable: null,
    dimension: 384,
    options: {},
  };
}
