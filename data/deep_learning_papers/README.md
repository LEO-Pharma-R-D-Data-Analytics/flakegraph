# Famous Deep Learning Papers Benchmark

This benchmark uses the 49 primary papers listed by the
[Bau Lab reading list](https://papers.baulab.info/). The repository contains the
download workflow, annotations, configurations, and compact measurements, but
does not redistribute the papers. Copyright and usage terms remain with each
paper's respective authors and publisher.

## Download

```bash
uv run python data/deep_learning_papers/download.py
```

The command downloads the versioned 49-paper source catalog embedded in
`gold.json` into the ignored `files/` directory, validates PDF signatures and
reviewed checksums, and writes an ignored portable `manifest.jsonl` for dataset
inventory and manifest-source workflows. The bundled fleet profile reads the
downloaded `files/` directory directly. Existing valid files are reused; pass
`--refresh` to download them again. `--discover-index` reads the mutable live
table and is intended only for a deliberate benchmark update and review.

The downloader resolves two known stale links in the source index to the
corresponding Bottou and Mikolov PDFs hosted by the same site.

## Contents

| Path | Purpose |
| --- | --- |
| `download.py` | Discovers, validates, and downloads the non-redistributed corpus. |
| `ontology.yaml` | Scientific-paper entity and relation vocabulary. |
| `configs/kubernetes-vllm.yaml` | Queue-backed Qwen and MinerU fleet profile. |
| `gold.json` | Selected reference graph and short evidence anchors. |
| `BENCHMARKS.md` | Controlled GPT-5.6 Sol and Qwen measurements. |
| `results/` | Compact public benchmark records without source content. |
| `files/` | Ignored downloaded PDFs. |
| `manifest.jsonl` | Ignored generated source manifest and checksums. |

## Annotation Contract

The gold graph is a selected typed reference, not an exhaustive transcription
of every paper. Each entity must be explicitly identified in a source and match
its ontology definition; bounded datasets, tasks, methods, and experimental
systems need not be proper nouns. Each relation must satisfy `ontology.yaml` and
include a phrase of at most ten words plus its physical PDF page. The current
reference contains 657 entities, 742 directed relations, and 742 independently
located evidence observations. Citation proximity alone does not establish
`BUILDS_ON`, and generic unnamed baselines are not canonical entities.

Changes require source-page review, endpoint and ontology validation, checksum
verification for all 49 documents, and a full benchmark rerun. Provider output
may reveal a missing valid fact, but it is never copied into the gold graph
without independent confirmation in the cited paper.
