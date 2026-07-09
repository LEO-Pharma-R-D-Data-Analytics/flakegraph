# Test Data

This folder contains the reusable document corpus used for local OCR,
file-source, manifest, Docker, and graph-processing validation.

The fixtures are original martial-arts history documents created for this
repository. They intentionally contain no third-party prose, images, or private
business data, so the sample corpus can be reviewed and published with the
source tree. Keep the files small enough for CI/developer use, but varied enough
to cover the formats the processor is expected to ingest.

## Fixture Matrix

| File | Format | Used To Exercise |
| --- | --- | --- |
| `samples/smoke.txt` | text | Fast deterministic pipeline and Docker smoke checks. |
| `samples/martial-arts-overview.pdf` | PDF | Repeatable single-document OCR, MinerU, and local production smoke checks. |
| `samples/martial-arts-lineages.pdf` | PDF | Larger multi-page PDF text extraction and chunking behavior. |
| `samples/martial-arts-interview.docx` | DOCX | Office document text extraction and normalization. |
| `samples/martial-arts-schools.pptx` | PPTX | Slide text extraction and block normalization. |
| `samples/martial-arts-timeline.html` | HTML | Text-native web document parsing. |
| `samples/manifest.jsonl` | JSONL manifest | Full-corpus file-source validation with pinned checksums, sizes, MIME types, and source URIs. |

## Provenance

All sample document text was written from scratch for FlakeGraph. The fixtures
are educational sample data, not scholarly source material, and should be used
only to exercise ingestion, OCR, chunking, and graph-extraction paths.

## Manifest Contract

`samples/manifest.jsonl` is the source of truth for the reusable sample corpus.
Every listed file has:

- a `file://data/samples/...` source URI
- a SHA-256 checksum
- a byte size
- a MIME type

`tests/unit/test_sample_data_contract.py` verifies that the manifest matches the
files on disk and that this README mentions each reusable fixture. When a sample
is added, removed, or replaced, update both the manifest and this matrix in the
same change.

The default local test config uses explicit `builtin_text` OCR because MinerU is
not assumed to be installed on every development machine. Production local OCR
uses `mineru_internal` inside the production image.
