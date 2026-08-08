# Third-Party Notices

This file summarizes important third-party license and terms considerations for
FlakeGraph. It is informational and is not a substitute for legal review.

FlakeGraph source code is licensed under the Apache License 2.0. That license
applies to this repository's application code, configuration examples, prompt
templates, documentation, and tests. The original benchmark dataset under
`data/martial_arts/` is separately dedicated under CC0-1.0. That dedication
does not re-license third-party packages, provider services, downloaded model
weights, or operating system packages installed into a Docker image.
See `data/martial_arts/LICENSE.md` for the dataset dedication.

## Dependency Installation

The source repository does not vendor Python packages or model weights. Runtime
dependencies are resolved from `pyproject.toml` and `uv.lock`; Docker images
install those dependencies during the image build.

If you redistribute a built image or packaged artifact, include the license and
notice files required by the installed Python packages, model artifacts, and
system packages in that artifact.

## Notable Runtime Components

- `mineru[pipeline]` is installed as an isolated Python 3.13 tool by the quick
  start and default Docker build rather than as part of FlakeGraph's Python 3.14
  dependency environment. MinerU is licensed under the MinerU Open Source License,
  based on Apache 2.0 with additional terms. Review the attribution obligation
  and commercial thresholds before offering MinerU-backed services. MinerU's
  current dependency constraints include an older Transformers line, so use only
  trusted MinerU model artifacts and caches; FlakeGraph strips provider and cloud
  credentials from the isolated subprocess environment as defense in depth.
  Source: https://github.com/opendatalab/MinerU/blob/master/LICENSE.md
- `sentence-transformers` is used by the local embedding profile. The default
  model, `sentence-transformers/all-MiniLM-L6-v2`, is published on Hugging Face
  with an Apache-2.0 license. Production worker and Spark images preload its
  pinned revision so ephemeral executors do not download mutable weights during
  a run. Review the selected model card when changing models.
  Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- The README uses Ollama with the official Qwen3.6 35B-A3B Q4 model tag. The
  upstream `Qwen/Qwen3.6-35B-A3B` model is Apache-2.0 licensed. Ollama and model
  artifacts are obtained separately and are not redistributed by FlakeGraph;
  review the selected Ollama artifact and model metadata before redistribution.
  Sources: https://github.com/ollama/ollama/blob/main/LICENSE and
  https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Fleet and benchmark profiles use vLLM with
  `nvidia/Qwen3.6-35B-A3B-NVFP4`. NVIDIA publishes this quantized checkpoint
  under Apache-2.0 and identifies Alibaba's Qwen3.6 model as its base. Review
  the pinned model card and upstream notices before redistributing weights.
  Source: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
- The optional chart-managed server uses NVIDIA's vLLM container. vLLM source
  is Apache-2.0, while the NGC container also contains NVIDIA and third-party
  runtime components governed by the container's accompanying notices and NGC
  terms. FlakeGraph references the image by digest but does not redistribute it.
  Source: https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/
- `torch` is used by the local embedding profile. FlakeGraph selects CPU wheels
  for Linux builds; the separately installed MinerU tool resolves its own Torch
  and Torchvision dependencies. Review PyTorch package licenses and notices when
  redistributing images.
- `snowflake-connector-python` is used for Snowflake connectivity and is
  Apache-2.0 licensed. Snowflake Cortex and Snowpark Container Services usage is
  governed by the target Snowflake account's terms and configuration.
- `psycopg` provides PostgreSQL connectivity for distributed task coordination
  and metadata. It is distributed under the GNU Lesser General Public License
  3.0 with exceptions described by the upstream project. Source:
  https://www.psycopg.org/psycopg3/docs/basic/license.html
- `boto3`, `botocore`, and their transitive AWS SDK dependencies provide the
  S3-compatible artifact adapter. They are installed as ordinary Python
  dependencies and retain their upstream Apache-2.0 licenses and notices.
- Apache Spark, PySpark, Hadoop AWS, and GraphFrames provide the optional
  partitioned graph-finalization runtime. `Dockerfile.spark` copies runtime
  components from the pinned Apache Spark image and resolves JVM packages during
  the image build. Preserve their Apache-2.0 license and notice requirements when
  redistributing that image.
- `plotly` is MIT licensed and powers the interactive graph surface in generated
  static explorer files. Those self-contained HTML files embed Plotly.js so they
  can be opened without a server or network connection. Source:
  https://github.com/plotly/plotly.py
- `igraph` and `leidenalg` implement deterministic Leiden community detection.
  They are installed as normal runtime dependencies and remain under their
  respective GNU GPL licenses. Distributors must review the resulting obligations
  for their packaged artifact. Sources: https://igraph.org/c/html/latest/igraph-License.html
  and https://github.com/vtraag/leidenalg/blob/main/LICENSE
- `pypdfium2` bundles Google PDFium native binaries used by the built-in PDF text
  fallback. PDFium is BSD-3-Clause licensed and includes third-party components;
  preserve the license files and `PDFIUM_THIRD_PARTY` notices shipped by
  `pypdfium2` when redistributing FlakeGraph packages or images. Source:
  https://github.com/pypdfium2-team/pypdfium2#licensing
- `GLiNER` is an optional local entity-extraction dependency installed only with
  `--extra extract-gliner`. Model weights have their own model-card terms and
  are not redistributed by FlakeGraph. Source: https://github.com/urchade/GLiNER
- GLiREL is not bundled because its published code/model license includes
  non-commercial and share-alike restrictions. It can only be connected as an
  externally reviewed custom relation adapter. Source:
  https://github.com/jackboyla/GLiREL
- `tesseract-ocr` is installed only when `KG_INSTALL_TESSERACT=true`; Tesseract
  is Apache-2.0 licensed. Source: https://tesseract-ocr.github.io/tessdoc/
- `poppler-utils` is installed only when `KG_INSTALL_TESSERACT=true`; distro
  packages include GPL/LGPL/MIT licensed components. Review this optional image
  variant before redistribution.
- KEDA provides queue-driven Kubernetes autoscaling and is installed separately
  from its pinned upstream Helm chart. KEDA is Apache-2.0 licensed and is not
  embedded in the FlakeGraph Python package or container images. Source:
  https://github.com/kedacore/keda
- CloudNativePG and the NVIDIA Kubernetes device plugin are optional deployment
  infrastructure referenced by the fleet guide; they are not embedded into the
  FlakeGraph package. Review and pin their upstream licenses, images, and charts
  before operating or redistributing a complete platform bundle.
- SeaweedFS is an optional S3-compatible deployment referenced by the fleet
  guide. It is not installed by the FlakeGraph chart or embedded in either image;
  operators remain responsible for its Apache-2.0 license, image, chart, and
  storage configuration.

## Provider Services And Models

FlakeGraph can call OpenAI-compatible APIs, Azure OpenAI, vLLM endpoints,
Snowflake Cortex, generic HTTP OCR services, MinerU API services, and local
models. The configured provider or model owner controls the applicable terms,
data usage policy, allowed outputs, region availability, and commercial
permissions.

Before production use, approve every configured provider endpoint and model
identifier for licensing, data residency, security, and acceptable-use posture.
