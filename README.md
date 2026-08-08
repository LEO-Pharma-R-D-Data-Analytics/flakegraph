<p align="center">
  <img src="app/assets/flakegraph-logo.png" alt="FlakeGraph" width="620">
</p>

# FlakeGraph

FlakeGraph turns documents into evidence-backed knowledge graphs for Snowflake.
It extracts text, identifies entities and relationships, resolves identities
across documents, validates facts against their sources, and writes a graph to
local artifacts or Snowflake tables.

The processing core depends on provider interfaces rather than vendor SDKs.
File sources, OCR, LLMs, embeddings, caches, and graph writers can therefore be
selected independently in configuration.

[![FlakeGraph processing pipeline from source documents through OCR, two-pass extraction, corpus finalization, and versioned graph publication across local, Kubernetes, and Snowflake runtimes](docs/assets/flakegraph-pipeline.svg)](docs/assets/flakegraph-pipeline.svg)

[Read how each processing stage works](docs/algorithm.md), including the
entity-first relation extraction contract and the differences between local,
Kubernetes, and Snowflake execution.

## Quick Start

The bundled example uses Python 3.14, `uv`, Ollama, Qwen3.6,
sentence-transformers, and the public martial-arts dataset. The same commands
work on macOS and Linux, including DGX Spark. Install
[Ollama](https://docs.ollama.com/quickstart), then pull the default model:

```bash
ollama pull qwen3.6:35b-a3b-q4_K_M
```

Start the FlakeGraph application:

```bash
# Clear values exported for another FlakeGraph profile; KG_* variables override YAML.
unset KG_INPUT_PATH KG_LLM_ENDPOINT KG_LLM_MODEL KG_LLM_API_KEY

# MinerU supports Python 3.10-3.13, so install its CLI in an isolated tool
# environment instead of constraining FlakeGraph's Python 3.14 dependencies.
uv tool install --python 3.13 "mineru[pipeline]==3.4.4"
uv sync --extra app --extra local-embeddings
uv run streamlit run app/streamlit_app.py
```

Open the displayed URL, upload documents or select a source, run preflight, and
start ingestion. Submitted graphs appear in the sidebar like a conversation
history, with search and storage filters. Select local artifacts or a remote
Snowflake schema as the output independently from where processing runs. Active
graphs show OCR, extraction, finalization, and writes; completed graphs open the
entity, relation, community, and evidence explorer directly.

The default Q4 model is about 24 GB and is intended for systems with at least
32 GB of usable unified or GPU memory. On a smaller machine, select the compact
6.6 GB Qwen3.5 fallback before running the same profile:

```bash
ollama pull qwen3.5:9b
export KG_LLM_MODEL=qwen3.5:9b
```

Use `ollama ps` during processing to confirm that the model is fully accelerated.
Select `data/martial_arts/files` in the app to process the complete public sample
corpus. Graph views contain source text and evidence, so handle them with the
same care as the input documents. See [Application](app/README.md) for local,
Kubernetes, and Snowflake behavior. The CLI remains available for headless and
automated runs.

## Providers

Configuration selects adapters for each boundary:

| Boundary | Included adapters |
| --- | --- |
| Files | Uploads, local paths, manifests, Azure Blob Storage, S3-compatible buckets, Snowflake stages |
| OCR | Adaptive provider fallback, built-in text extraction, MinerU, Tesseract, generic HTTP, Snowflake Cortex |
| LLM | Ollama and other OpenAI-compatible APIs, Azure OpenAI, vLLM, Snowflake Cortex |
| Entity extraction | LLM or optional local GLiNER |
| Embeddings | Sentence-transformers, OpenAI-compatible APIs, Azure OpenAI, Snowflake Cortex |
| Cache | Local JSON or Snowflake |
| Writer | Local artifacts, direct Snowflake, Snowflake bulk load |

```bash
uv run flakegraph config providers
uv run flakegraph config print --config configs/local-mineru-oss.yaml
```

Provider implementations live under `src/kg_processor/adapters/` and implement
interfaces from `src/kg_processor/ports/`. See
[Architecture](docs/architecture.md) for the extension contract.

## Execution Modes

The same processing semantics are available in three runtimes:

| Mode | Use case | Coordination and storage |
| --- | --- | --- |
| Local | Development, evaluation, and single-host processing | One process, local cache and artifacts |
| Kubernetes | On-premises GPU fleets and large corpora | PostgreSQL leases, KEDA worker scaling, object storage, Spark finalization |
| Snowflake | Snowflake-native providers and graph storage | Stages, Cortex, Snowflake tables, optional SPCS container |

For a Kubernetes fleet, use the Helm chart and
[Kubernetes deployment guide](docs/kubernetes-fleet.md). For Snowflake objects,
grants, and deployment SQL, see [Snowflake setup](docs/snowflake-setup.md).

## Docker

Build the production image with MinerU support:

```bash
docker build --platform linux/amd64 -t flakegraph:mineru-oss .
```

Run it with an OpenAI-compatible LLM:

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/out:/app/out" \
  -v flakegraph-mineru-cache:/home/kgprocessor/.cache/mineru \
  -e KG_LLM_ENDPOINT \
  -e KG_LLM_MODEL \
  -e KG_LLM_API_KEY \
  -e KG_INPUT_PATH=data/martial_arts/files/martial-arts-overview.pdf \
  flakegraph:mineru-oss worker --config configs/local-mineru-oss.yaml
```

## Snowflake

Generate setup artifacts from the account-neutral Snowflake profile:

```bash
uv run flakegraph snowflake setup-sql --config configs/snowflake-cortex.yaml
uv run flakegraph snowflake access-check --config configs/snowflake-cortex.yaml
uv run flakegraph snowflake service-spec --config configs/snowflake-cortex.yaml
uv run flakegraph snowflake execute-job-sql --config configs/snowflake-cortex.yaml
```

## Project Layout

```text
src/kg_processor/
  domain/       provider-independent graph and document models
  ports/        interfaces implemented by providers and infrastructure
  application/  processing, validation, distribution, and inspection services
  adapters/     provider and persistence implementations
  config/       typed settings, provider registry, and preflight checks

app/            Streamlit control plane and runtime backends
configs/        reusable provider profiles and ontologies
data/           self-contained public benchmark datasets
deploy/         container launchers and Kubernetes Helm chart
docs/           architecture and deployment guides
tests/          unit, integration, packaging, and deployment contracts
```

## Tests

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy src
uv run pytest
```

The martial-arts dataset includes a gold graph and published measurements. See
[its dataset guide](data/martial_arts/README.md) and
[benchmark results](data/martial_arts/BENCHMARKS.md).

## Documentation

- [How FlakeGraph builds a graph](docs/algorithm.md)
- [Streamlit application](app/README.md)
- [Architecture](docs/architecture.md)
- [Configuration profiles](configs/README.md)
- [Kubernetes fleet deployment](docs/kubernetes-fleet.md)
- [Snowflake setup](docs/snowflake-setup.md)
- [Benchmark datasets](data/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

FlakeGraph is released under the [Apache License 2.0](LICENSE). Dependencies,
models, provider services, and image variants retain their own licenses and
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
