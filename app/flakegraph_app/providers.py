"""Provider choices and defaults presented by the control-plane UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderOption:
    """One concise UI description of a processing-core adapter."""

    name: str
    label: str
    needs_model: bool = False
    needs_endpoint: bool = False
    needs_api_key: bool = False
    default_dimension: int | None = None


OCR_PROVIDERS = (
    ProviderOption("fallback", "Adaptive text + MinerU"),
    ProviderOption("builtin_text", "Built-in document text"),
    ProviderOption("mineru_internal", "MinerU (inside worker)"),
    ProviderOption("mineru_api", "MinerU API", needs_endpoint=True, needs_api_key=True),
    ProviderOption("tesseract_internal", "Tesseract (inside worker)"),
    ProviderOption("generic_http", "Generic HTTP OCR", needs_endpoint=True, needs_api_key=True),
    ProviderOption("snowflake_cortex", "Snowflake Cortex Parse Document"),
)

LLM_PROVIDERS = (
    ProviderOption("ollama", "Ollama", needs_model=True, needs_endpoint=True),
    ProviderOption("vllm_local", "vLLM", needs_model=True, needs_endpoint=True),
    ProviderOption(
        "openai_compatible",
        "OpenAI-compatible API",
        needs_model=True,
        needs_endpoint=True,
        needs_api_key=True,
    ),
    ProviderOption(
        "azure_openai",
        "Azure OpenAI",
        needs_model=True,
        needs_endpoint=True,
        needs_api_key=True,
    ),
    ProviderOption("snowflake_cortex", "Snowflake Cortex", needs_model=True),
)

EMBEDDING_PROVIDERS = (
    ProviderOption(
        "sentence_transformers",
        "Sentence Transformers",
        needs_model=True,
        default_dimension=384,
    ),
    ProviderOption(
        "openai_compatible",
        "OpenAI-compatible API",
        needs_model=True,
        needs_endpoint=True,
        needs_api_key=True,
        default_dimension=1536,
    ),
    ProviderOption(
        "azure_openai",
        "Azure OpenAI",
        needs_model=True,
        needs_endpoint=True,
        needs_api_key=True,
        default_dimension=1536,
    ),
    ProviderOption(
        "snowflake_cortex",
        "Snowflake Cortex",
        needs_model=True,
        default_dimension=768,
    ),
)

# Known defaults preserve compatibility for programmatic callers while making
# every generated app profile carry an explicit vector-shape contract. Unknown
# or custom model identifiers must provide ``ProviderSelection.dimension``.
EMBEDDING_MODEL_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "snowflake-arctic-embed-m-v1.5": 768,
    "snowflake-arctic-embed-l-v2.0": 1024,
}

# A Snowflake VECTOR column fixes its width when the table is created, so the
# destination decides which embedding models can be used at all. Offering a model
# the graph tables cannot store leaves an operator with a run they cannot submit.
CORTEX_EMBEDDING_MODELS_BY_WIDTH = {
    768: "snowflake-arctic-embed-m-v1.5",
    1024: "snowflake-arctic-embed-l-v2.0",
}


def cortex_embedding_model_for_width(width: int | None) -> str | None:
    """Return the Cortex embedding model whose vectors fit a stored width."""

    return CORTEX_EMBEDDING_MODELS_BY_WIDTH.get(width) if width else None

DEFAULTS = {
    "llm": {
        "ollama": ("http://localhost:11434", "hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_K_M"),
        "vllm_local": ("http://localhost:8000/v1", "unsloth/Qwen3.8-27B-NVFP4"),
        "openai_compatible": ("", ""),
        "azure_openai": ("", ""),
        "snowflake_cortex": ("", "mistral-large2"),
    },
    "embedding": {
        "sentence_transformers": ("", "sentence-transformers/all-MiniLM-L6-v2"),
        "openai_compatible": ("", ""),
        "azure_openai": ("", ""),
        "snowflake_cortex": ("", "snowflake-arctic-embed-m-v1.5"),
    },
}


def option_by_name(options: tuple[ProviderOption, ...], name: str) -> ProviderOption:
    """Resolve one provider option or fail on an impossible UI state."""

    for option in options:
        if option.name == name:
            return option
    raise ValueError(f"Unknown provider option: {name}")


def embedding_dimension(provider: str, model: str | None, explicit: int | None) -> int:
    """Resolve the reviewed vector dimension for one embedding selection.

    The UI always sends an explicit value. The known-model fallback supports
    older programmatic callers without silently guessing the shape of custom or
    hosted models whose dimensions are deployment-specific.
    """

    if explicit is not None:
        if explicit <= 0:
            raise ValueError("Embedding dimension must be positive")
        return explicit
    if model and model in EMBEDDING_MODEL_DIMENSIONS:
        return EMBEDDING_MODEL_DIMENSIONS[model]
    raise ValueError(f"Embedding dimension is required for {provider} model {model or '<unset>'}")
