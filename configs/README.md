# Configuration Profiles

This directory contains runnable FlakeGraph profiles. Each YAML file selects a
runtime, file source, OCR provider, LLM provider, embedding provider, writer,
and optional cache/job behavior. Values written as `${NAME}` are supplied from
environment variables at runtime.

## Smoke And Test Profiles

| Config | Purpose | Used by |
| --- | --- | --- |
| `local-smoke.yaml` | Fast credential-free local smoke run using `builtin_text`, fake LLM extraction, hash embeddings, and local artifacts. | README quick smoke, `.github/docker-compose.smoke.yaml`, CI Docker smoke |
| `local-cache-smoke.yaml` | Same lightweight providers as `local-smoke.yaml`, with the local JSON cache enabled. | Cache tests and local cache debugging |
| `local-manifest-smoke.yaml` | Reads `data/samples/manifest.jsonl` through the manifest file source with fake/hash providers. | Manifest integration tests |
| `local-mineru-smoke.yaml` | Runs one-page MinerU OCR against the sample PDF, then fake/hash providers. | Optional `mineru` compose smoke profile |
| `local-tesseract-smoke.yaml` | Runs Tesseract OCR preflight against the sample PDF with fake/hash providers. | Optional `tesseract` compose smoke profile |
| `local-test.yaml` | Broader credential-free local profile over `data/samples` for development and tests. | Local developer checks |

## Local Real-Provider Profiles

| Config | Purpose | Used by |
| --- | --- | --- |
| `local-mineru-oss.yaml` | Default local open-source OCR/embedding path: MinerU internal OCR, OpenAI-compatible LLM, sentence-transformers embeddings, local artifacts. | README quick start and Docker example |
| `local-mineru-openai.yaml` | MinerU internal OCR with OpenAI-compatible LLM and OpenAI-compatible embeddings. | Local provider validation |
| `local-mineru-api.yaml` | External MinerU API OCR with OpenAI-compatible LLM and embedding endpoints. | Optional `mineru-api` compose profile |
| `local-generic-http-ocr.yaml` | Generic HTTP OCR adapter contract with OpenAI-compatible LLM and local sentence-transformers embeddings. | Local adapter validation |
| `local-azure-foundry.yaml` | Azure OpenAI LLM and embeddings over local sample files with built-in text extraction. | Azure provider validation |
| `local-azure-smoke.yaml` | Small Azure OpenAI live-provider smoke profile over `data/samples/smoke.txt`. | `tests/integration/test_azure_openai_live.py` |
| `local-azure-blob-mineru-oss.yaml` | Azure Blob file source, MinerU internal OCR, OpenAI-compatible LLM, sentence-transformers embeddings, local artifacts. | Azure Blob/local OCR validation |
| `local-vllm-mineru-oss.yaml` | MinerU internal OCR, local vLLM-compatible LLM endpoint, sentence-transformers embeddings, local artifacts. | Optional `vllm` compose profile |

## Snowflake And Deployment Profiles

| Config | Purpose | Used by |
| --- | --- | --- |
| `local-snowflake-direct.yaml` | Local worker reads local files and writes directly to Snowflake tables with Azure OpenAI providers. | Snowflake writer integration path |
| `local-snowflake-bulk.yaml` | Local worker reads local files and writes through Snowflake bulk staging/merge with Azure OpenAI providers. | Snowflake bulk writer integration path |
| `onprem-azure-blob-vllm-mineru-oss.yaml` | On-prem worker profile for Azure Blob input, MinerU OCR, local vLLM LLM, local embeddings, Snowflake bulk writer/cache, and file queue leasing. | Kubernetes job rendering default |
| `snowflake-cortex.yaml` | Snowpark Container Services profile using Snowflake stages, Cortex OCR/LLM/embeddings, Snowflake cache, and Snowflake bulk writer. | Snowflake SQL/spec rendering, README Snowflake commands, live Snowflake tests |

## Compose Profiles

The CI/local smoke compose file lives at
`.github/docker-compose.smoke.yaml`. It intentionally sits outside this folder
because it is a harness for running selected configs in Docker, not an
application config itself.

Current compose mappings:

| Compose service | Config |
| --- | --- |
| `flakegraph` | `local-smoke.yaml` |
| `flakegraph-mineru` | `local-mineru-smoke.yaml` |
| `flakegraph-mineru-api` | `local-mineru-api.yaml` |
| `flakegraph-tesseract` | `local-tesseract-smoke.yaml` |
| `flakegraph-vllm` | `local-vllm-mineru-oss.yaml` |

## Naming Rules

- `local-*` profiles run on the developer machine or inside a local container.
- `*-smoke` profiles are intentionally small and suitable for quick validation.
- `snowflake-*` profiles use Snowflake-native services or deployment artifacts.
- `onprem-*` profiles are for non-Snowflake compute that still writes to
  Snowflake.
